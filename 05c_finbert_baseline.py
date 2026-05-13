"""
Step 5c: FinBERT Fine-Tuned Baseline

Fine-tune ProsusAI/finbert on the 100-company earnings call transcripts
using the same nested cross-validation design as XGBoost (5-fold outer,
3-fold inner for learning rate selection), making results directly comparable.

Input text: management prepared remarks, truncated to 512 tokens (BERT limit).
This captures the most signal-rich part of the transcript (forward guidance,
earnings commentary) while fitting within BERT's context window.

Tasks: binary only (UP/DOWN). Ternary is excluded because with only 10 FLAT
samples across 5 folds (2 per fold), stable fine-tuning is not feasible.

Output:
  - data/results/finbert_results.json        (metrics, fold scores)
  - data/results/finbert_predictions.csv     (per-company predictions)
  - figures/finbert_confusion_binary.png     (confusion matrix)

CPU-optimised settings (MAX_SEQ_LEN=512, EPOCHS_OUTER=10, EPOCHS_INNER=1, BATCH=4):
  ~2-3 hours on 16GB RAM CPU, ~30-60 mins on GPU.

Run AFTER 03_label_construction.py and 04_feature_extraction.py.
No API calls. No internet needed after model download.
"""
import json
import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

import config
from utils.transcript_parser import (
    parse_structured_content, parse_raw_content, strip_boilerplate,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FINBERT_MODEL   = "ProsusAI/finbert"
MAX_SEQ_LEN     = 512          # BERT maximum (was 256 — too short for signal)
BATCH_SIZE      = 4            # reduce to 2 if OOM on GPU with 512 tokens
EPOCHS_INNER    = 1            # for inner CV (faster, just for LR selection)
EPOCHS_OUTER    = 10            # for outer CV (more training with ~72 samples)
LR_CANDIDATES   = [1e-5, 2e-5, 3e-5, 5e-5]   # expanded inner CV grid search
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.01
RANDOM_SEED     = config.RANDOM_SEED

# Use GPU if available, otherwise CPU with all cores
torch.set_num_threads(os.cpu_count() or 4)   # use all CPU cores for matrix ops
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_transcripts_and_labels() -> pd.DataFrame:
    """
    Load 100-company transcripts and labels.
    Returns DataFrame with columns: symbol, text, label_binary.
    text = management prepared remarks (truncated to MAX_SEQ_LEN tokens).
    """
    companies  = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    labels_df  = pd.read_csv(config.LABELS_DIR / "labels_100.csv")
    labels_map = labels_df.set_index("symbol")[["label_binary", "label_ternary"]].to_dict("index")

    rows = []
    for _, row in companies.iterrows():
        ticker = row["ticker"]
        path   = config.RAW_TRANSCRIPTS_DIR / f"{ticker}.json"
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        parsed = parse_structured_content(data.get("structured_content"))
        if parsed is None:
            parsed = parse_raw_content(data.get("content", ""))
        if parsed is None:
            continue

        # Use management text as primary input — most signal-rich
        # Fall back to full text if management text is empty
        mgmt_text = parsed.get("management_text", "").strip()
        if len(mgmt_text) < 100:
            mgmt_text = parsed.get("full_text", "").strip()
        if len(mgmt_text) < 50:
            continue

        # Strip boilerplate (safe-harbor disclaimers) so BERT sees actual content
        mgmt_text = strip_boilerplate(mgmt_text)

        label_info = labels_map.get(ticker, {})
        label_binary = label_info.get("label_binary")

        rows.append({
            "symbol":       ticker,
            "text":         mgmt_text,
            "label_binary": label_binary,
        })

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} transcripts")
    print(f"  Binary label distribution: {df['label_binary'].value_counts().to_dict()}")
    return df


# ── Dataset ───────────────────────────────────────────────────────────────────

class EarningsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get(
                "token_type_ids",
                torch.zeros_like(self.encodings["input_ids"])
            )[idx],
            "labels": self.labels[idx],
        }


# ── Model training ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    """Evaluate model. Returns (predictions, probabilities)."""
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
    return np.array(all_preds), np.array(all_probs)


def build_model_and_optimizer(n_classes, lr, n_train_steps):
    """Load FinBERT, add classification head, return model + optimizer + scheduler."""
    from transformers import (
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    import torch.optim as optim

    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    model = model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    n_warmup = int(WARMUP_RATIO * n_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=n_warmup,
        num_training_steps=n_train_steps,
    )
    return model, optimizer, scheduler


# ── Nested CV ─────────────────────────────────────────────────────────────────

def run_nested_cv(texts, y, le, n_classes):
    """
    Nested cross-validation matching XGBoost design:
      Outer: 5-fold stratified — performance estimation
      Inner: 3-fold stratified — learning rate selection (grid search)

    Returns dict with OOF predictions, probabilities, fold results.
    """
    from transformers import AutoTokenizer

    print(f"  Loading tokenizer from {FINBERT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)

    outer_cv = StratifiedKFold(
        n_splits=config.OUTER_CV_FOLDS, shuffle=True,
        random_state=RANDOM_SEED
    )
    inner_cv = StratifiedKFold(
        n_splits=config.INNER_CV_FOLDS, shuffle=True,
        random_state=RANDOM_SEED
    )

    all_preds = np.full(len(y), -1, dtype=int)
    all_probs = np.zeros((len(y), n_classes))
    fold_results = []

    def make_weighted_forward(model, class_weights, device):
        """Factory to avoid closure-over-loop-variable bug."""
        orig = model.forward
        def _weighted_forward(*args, **kwargs):
            labels = kwargs.pop("labels", None)
            out = orig(*args, **kwargs)
            if labels is not None:
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
                out.loss = loss_fn(out.logits, labels.to(device))
            return out
        return _weighted_forward

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(texts, y)):
        print(f"\n    Outer fold {fold_idx + 1}/{config.OUTER_CV_FOLDS}...")

        X_train_texts = [texts[i] for i in train_idx]
        X_test_texts  = [texts[i] for i in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # ── Inner CV: grid search over learning rates ──────────────────
        best_lr   = LR_CANDIDATES[0]
        best_val_f1 = -1.0

        for lr in LR_CANDIDATES:
            lr_val_f1s = []
            for inner_train_idx, inner_val_idx in inner_cv.split(
                X_train_texts, y_train
            ):
                Xi_train = [X_train_texts[i] for i in inner_train_idx]
                Xi_val   = [X_train_texts[i] for i in inner_val_idx]
                yi_train = y_train[inner_train_idx]
                yi_val   = y_train[inner_val_idx]

                # Compute class weights for imbalanced binary
                if n_classes == 2:
                    n_neg = int(np.sum(yi_train == 0))
                    n_pos = int(np.sum(yi_train == 1))
                    spw   = n_neg / max(n_pos, 1)
                    class_weights = torch.tensor(
                        [spw, 1.0] if n_neg > n_pos else [1.0, 1.0/spw],
                        dtype=torch.float
                    ).to(DEVICE)
                else:
                    class_weights = None

                ds_train = EarningsDataset(Xi_train, yi_train, tokenizer)
                ds_val   = EarningsDataset(Xi_val,   yi_val,   tokenizer)
                dl_train = torch.utils.data.DataLoader(
                    ds_train, batch_size=BATCH_SIZE, shuffle=True
                )
                dl_val = torch.utils.data.DataLoader(
                    ds_val, batch_size=BATCH_SIZE, shuffle=False
                )

                n_steps = len(dl_train) * EPOCHS_INNER
                model, optimizer, scheduler = build_model_and_optimizer(
                    n_classes, lr, n_steps
                )

                # Override loss with weighted CE if needed
                if class_weights is not None:
                    model.forward = make_weighted_forward(model, class_weights, DEVICE)

                for _ in range(EPOCHS_INNER):
                    train_epoch(model, dl_train, optimizer, scheduler, DEVICE)

                preds_val, _ = evaluate(model, dl_val, DEVICE)
                val_f1 = f1_score(
                    yi_val, preds_val, average="macro", zero_division=0
                )
                lr_val_f1s.append(val_f1)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            mean_val_f1 = float(np.mean(lr_val_f1s))
            print(f"      LR={lr:.0e} | inner val F1={mean_val_f1:.4f}")
            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                best_lr     = lr

        print(f"      Best LR: {best_lr:.0e} (val F1={best_val_f1:.4f})")

        # ── Outer fold: train with best LR, evaluate on test ──────────
        # Compute class weights on full outer training set
        if n_classes == 2:
            n_neg = int(np.sum(y_train == 0))
            n_pos = int(np.sum(y_train == 1))
            spw   = n_neg / max(n_pos, 1)
            class_weights = torch.tensor(
                [spw, 1.0] if n_neg > n_pos else [1.0, 1.0/spw],
                dtype=torch.float
            ).to(DEVICE)
        else:
            class_weights = None

        ds_train = EarningsDataset(X_train_texts, y_train, tokenizer)
        ds_test  = EarningsDataset(X_test_texts,  y_test,  tokenizer)
        dl_train = torch.utils.data.DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True
        )
        dl_test = torch.utils.data.DataLoader(
            ds_test, batch_size=BATCH_SIZE, shuffle=False
        )

        n_steps = len(dl_train) * EPOCHS_OUTER
        model, optimizer, scheduler = build_model_and_optimizer(
            n_classes, best_lr, n_steps
        )

        if class_weights is not None:
            model.forward = make_weighted_forward(model, class_weights, DEVICE)

        for epoch in range(EPOCHS_OUTER):
            loss = train_epoch(model, dl_train, optimizer, scheduler, DEVICE)
            print(f"      Epoch {epoch+1}/{EPOCHS_OUTER} | loss={loss:.4f}")

        preds_test, probs_test = evaluate(model, dl_test, DEVICE)

        # Store OOF predictions
        all_preds[test_idx] = preds_test
        all_probs[test_idx] = probs_test

        fold_f1  = f1_score(y_test, preds_test, average="macro", zero_division=0)
        fold_acc = accuracy_score(y_test, preds_test)
        print(f"      Outer fold F1={fold_f1:.4f} | Acc={fold_acc:.4f}")

        fold_results.append({
            "fold":     fold_idx,
            "f1_macro": fold_f1,
            "accuracy": fold_acc,
            "best_lr":  best_lr,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Overall OOF metrics ────────────────────────────────────────────
    valid_mask  = all_preds >= 0
    label_names = list(le.classes_)

    oof_f1  = f1_score(y[valid_mask], all_preds[valid_mask],
                       average="macro", zero_division=0)
    oof_acc = accuracy_score(y[valid_mask], all_preds[valid_mask])
    oof_kap = cohen_kappa_score(y[valid_mask], all_preds[valid_mask])
    oof_cm  = confusion_matrix(y[valid_mask], all_preds[valid_mask])
    oof_rep = classification_report(
        y[valid_mask], all_preds[valid_mask],
        target_names=label_names, output_dict=True, zero_division=0,
    )

    # AUC from probability scores
    try:
        if n_classes == 2:
            oof_auc = float(roc_auc_score(
                y[valid_mask], all_probs[valid_mask, 1]
            ))
        else:
            from sklearn.preprocessing import label_binarize
            yt_bin  = label_binarize(y[valid_mask], classes=range(n_classes))
            present = [i for i in range(n_classes) if i in y[valid_mask]]
            oof_auc = float(roc_auc_score(
                yt_bin[:, present], all_probs[valid_mask][:, present],
                average="macro", multi_class="ovr",
            ))
    except Exception as e:
        print(f"  AUC warning: {e}")
        oof_auc = float("nan")

    print(f"\n    Overall OOF results:")
    print(f"      Macro F1 : {oof_f1:.4f}")
    print(f"      Accuracy : {oof_acc:.4f}")
    print(f"      AUC-ROC  : {oof_auc:.4f}")
    print(f"      Kappa    : {oof_kap:.4f}")
    print(classification_report(
        y[valid_mask], all_preds[valid_mask],
        target_names=label_names, zero_division=0,
    ))

    return {
        "f1_macro":   oof_f1,
        "accuracy":   oof_acc,
        "auc_roc":    oof_auc,
        "cohen_kappa": oof_kap,
        "report":     oof_rep,
        "confusion_matrix": oof_cm.tolist(),
        "fold_results": fold_results,
        "predictions": all_preds[valid_mask].tolist(),
        "probabilities": all_probs[valid_mask].tolist(),
        "valid_mask": valid_mask.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 5c: FINBERT FINE-TUNED BASELINE")
    print("=" * 70)
    print(f"\n  Device : {DEVICE}")
    print(f"  Model  : {FINBERT_MODEL}")
    print(f"  Max seq: {MAX_SEQ_LEN} tokens")
    print(f"  Input  : management prepared remarks")
    print(f"  Task   : binary (UP/DOWN) only")

    # ── Check dependencies ────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise ImportError(
            "transformers not installed.\n"
            "Run: pip install transformers torch datasets"
        )

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/4] Loading transcripts and labels...")
    df = load_transcripts_and_labels()

    # Binary task only
    task    = "binary"
    df_bin  = df[df["label_binary"].notna()].copy().reset_index(drop=True)
    print(f"  Binary samples: {len(df_bin)}")
    print(f"  Distribution  : {df_bin['label_binary'].value_counts().to_dict()}")

    le  = LabelEncoder()
    y   = le.fit_transform(df_bin["label_binary"].values)
    texts = df_bin["text"].tolist()
    n_classes = len(le.classes_)
    print(f"  Classes: {list(le.classes_)}")  # ['DOWN', 'UP'] after encoding

    # ── Run nested CV ─────────────────────────────────────────────────
    print(f"\n[2/4] Running nested CV "
          f"({config.OUTER_CV_FOLDS}-fold outer, "
          f"{config.INNER_CV_FOLDS}-fold inner)...")
    print(f"  LR candidates : {LR_CANDIDATES}")
    print(f"  Inner epochs  : {EPOCHS_INNER}")
    print(f"  Outer epochs  : {EPOCHS_OUTER}")
    print(f"  Batch size    : {BATCH_SIZE}")

    results = run_nested_cv(texts, y, le, n_classes)

    # ── Save predictions ──────────────────────────────────────────────
    print("\n[3/4] Saving predictions and results...")

    valid_mask = np.array(results["valid_mask"])
    symbols    = df_bin["symbol"].values[valid_mask]
    preds_enc  = np.array(results["predictions"])
    pred_labels = le.inverse_transform(preds_enc)
    probs_arr  = np.array(results["probabilities"])
    confidences = probs_arr.max(axis=1)

    pred_df = pd.DataFrame({
        "symbol":            symbols,
        "finbert_pred_binary": pred_labels,
        "finbert_conf_binary": confidences,
    })
    pred_path = config.RESULTS_DIR / "finbert_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions: {pred_path}")

    # Save full results JSON
    save_results = {
        "task":         task,
        "model":        FINBERT_MODEL,
        "input":        "management_text",
        "max_seq_len":  MAX_SEQ_LEN,
        "epochs_outer": EPOCHS_OUTER,
        "epochs_inner": EPOCHS_INNER,
        "lr_candidates": LR_CANDIDATES,
        "f1_macro":     results["f1_macro"],
        "accuracy":     results["accuracy"],
        "auc_roc":      results["auc_roc"],
        "cohen_kappa":  results["cohen_kappa"],
        "report":       results["report"],
        "confusion_matrix": results["confusion_matrix"],
        "fold_results": results["fold_results"],
    }
    json_path = config.RESULTS_DIR / "finbert_results.json"
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"  Results JSON: {json_path}")

    # ── Confusion matrix figure ───────────────────────────────────────
    label_names = list(le.classes_)
    cm = np.array(results["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"FinBERT Fine-tuned (Binary)\n"
        f"F1={results['f1_macro']:.3f}  AUC={results['auc_roc']:.3f}"
    )
    plt.tight_layout()
    fig_path = config.FIGURES_DIR / "finbert_confusion_binary.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix: {fig_path}")

    # ── Summary comparison ────────────────────────────────────────────
    print("\n[4/4] Summary comparison with other baselines...")

    xgb_path = config.RESULTS_DIR / "xgb_results.json"
    finbert_f1  = results["f1_macro"]
    finbert_auc = results["auc_roc"]
    finbert_kap = results["cohen_kappa"]

    print(f"\n  {'Method':<30} {'F1':>6} {'AUC':>6} {'Kappa':>7}  Notes")
    print(f"  {'-'*65}")

    if xgb_path.exists():
        with open(xgb_path) as f:
            xgb_res = json.load(f)
        for key in ["binary_XGB-full", "binary_XGB-sentiment-only"]:
            if key in xgb_res:
                r = xgb_res[key]
                name = key.replace("binary_", "")
                print(f"  {name:<30} {r['f1_macro']:>6.3f}   N/A "
                      f"{r.get('cohen_kappa', float('nan')):>7.3f}  N=90, nested CV")

    fc_path = config.RESULTS_DIR / "xgb_fullcorpus_results.json"
    if fc_path.exists():
        with open(fc_path) as f:
            fc_res = json.load(f)
        for key in ["binary_XGB-fullcorpus"]:
            if key in fc_res:
                r = fc_res[key]
                print(f"  {'XGB-fullcorpus':<30} {r['f1_macro']:>6.3f} "
                      f"{r.get('auc_roc', float('nan')):>6.3f} "
                      f"{r.get('cohen_kappa', float('nan')):>7.3f}  "
                      f"N_train={r['n_train']:,}, held-out")

    print(f"  {'FinBERT-finetuned':<30} {finbert_f1:>6.3f} "
          f"{finbert_auc:>6.3f} {finbert_kap:>7.3f}  N=90, nested CV")

    # Fold-level variance
    fold_f1s = [fr["f1_macro"] for fr in results["fold_results"]]
    print(f"\n  FinBERT fold F1 scores: {[f'{x:.3f}' for x in fold_f1s]}")
    print(f"  Mean={np.mean(fold_f1s):.3f} | Std={np.std(fold_f1s):.3f} | "
          f"Min={np.min(fold_f1s):.3f} | Max={np.max(fold_f1s):.3f}")

    print(f"\n{'=' * 70}")
    print("STEP 5c COMPLETE")
    print(f"{'=' * 70}")
    print("\nTo include FinBERT in 08_evaluation.py:")
    print("  Load finbert_results.json and finbert_predictions.csv")
    print("  The 08_evaluation.py will pick these up automatically")
    print("  if you add FinBERT to the XGB section of the evaluation.")
    print(f"\nFinBERT binary F1  = {finbert_f1:.4f}")
    print(f"FinBERT binary AUC = {finbert_auc:.4f}")
    print(f"FinBERT binary kap = {finbert_kap:.4f}")


if __name__ == "__main__":
    main()