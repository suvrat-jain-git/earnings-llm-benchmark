"""
Step 5b: XGBoost Full-Corpus Baseline

Train XGBoost on ALL ~33,000 labelled transcripts (TF-IDF + BoW features),
evaluate on the 100-company held-out set. This gives a genuinely strong
classical baseline — training on thousands of samples instead of 80.

Design:
  - Features: 80-dim (50 TF-IDF SVD + 30 BoW SVD), from features_full_sparse.parquet
  - Labels:   labels_full.csv for training, labels_100.csv for evaluation
  - Split:    100-company eval set is STRICTLY held out from training
  - Tuning:   Optuna 50 trials on a random 20% validation split of the
              training corpus (no nested CV — too expensive at 33k samples)
  - Evaluation: standard metrics on the 100-company held-out test set

Why only TF-IDF + BoW (not full 254 features)?
  Speaker-level features (POS, NER, L-M sentiment, readability) were only
  extracted for the 100-company set because running spaCy on 33,000 long
  transcripts takes many hours. The 80 sparse features are available for
  all transcripts and provide a fair large-corpus comparison.

Output:
  - data/results/xgb_fullcorpus_results.json  (metrics)
  - data/results/xgb_fullcorpus_predictions_binary.csv
  - data/results/xgb_fullcorpus_predictions_ternary.csv
  - figures/xgb_fullcorpus_confusion_binary.png
  - figures/xgb_fullcorpus_confusion_ternary.png

Run AFTER 04_feature_extraction.py and 03_label_construction.py.
"""
import json
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


# -- Helpers -------------------------------------------------------------------

def compute_class_weight(y: np.ndarray) -> float:
    """scale_pos_weight = n_negative / n_positive for binary XGBoost."""
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    return n_neg / max(n_pos, 1)


def tune_and_train(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   n_classes: int, n_trials: int = 50) -> xgb.XGBClassifier:
    """
    Tune XGBoost with Optuna on a held-out validation split,
    then retrain on full train+val with best params.
    Returns the retrained model.
    """
    objective_type = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if n_classes > 2 else "logloss"

    def optuna_objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        if n_classes > 2:
            m = xgb.XGBClassifier(
                objective=objective_type,
                num_class=n_classes,
                eval_metric=eval_metric,
                
                random_state=config.RANDOM_SEED,
                verbosity=0,
                **params,
            )
        else:
            spw = compute_class_weight(y_train)
            m = xgb.XGBClassifier(
                objective=objective_type,
                eval_metric=eval_metric,
                
                random_state=config.RANDOM_SEED,
                verbosity=0,
                scale_pos_weight=spw,
                **params,
            )

        m.fit(X_train, y_train, verbose=False)
        preds = m.predict(X_val)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    print(f"    Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    print(f"    Best val F1: {study.best_value:.4f} | params: {best_params}")

    # Retrain on train + val combined with best params
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    if n_classes > 2:
        model = xgb.XGBClassifier(
            objective=objective_type,
            num_class=n_classes,
            eval_metric=eval_metric,
            
            random_state=config.RANDOM_SEED,
            verbosity=0,
            **best_params,
        )
    else:
        spw = compute_class_weight(y_full)
        model = xgb.XGBClassifier(
            objective=objective_type,
            eval_metric=eval_metric,
            
            random_state=config.RANDOM_SEED,
            verbosity=0,
            scale_pos_weight=spw,
            **best_params,
        )

    model.fit(X_full, y_full, verbose=False)
    return model, best_params


def plot_confusion(cm: np.ndarray, labels: list, title: str, path: str):
    """Save a labelled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 5b: XGBOOST FULL-CORPUS BASELINE")
    print("=" * 70)

    # -- Load features ----------------------------------------------------
    print("\n[1/5] Loading features and labels...")

    # Full-corpus sparse features (TF-IDF SVD + BoW SVD only)
    sparse_path = config.FEATURES_DIR / "features_full_sparse.parquet"
    if not sparse_path.exists():
        raise FileNotFoundError(
            f"Missing: {sparse_path}\n"
            f"Run 04_feature_extraction.py first."
        )
    full_feats = pd.read_parquet(sparse_path)
    print(f"  Full sparse features: {full_feats.shape}")

    # Full labels
    labels_full_path = config.LABELS_DIR / "labels_full.csv"
    if not labels_full_path.exists():
        raise FileNotFoundError(
            f"Missing: {labels_full_path}\n"
            f"Run 03_label_construction.py first."
        )
    labels_full = pd.read_csv(labels_full_path)
    print(f"  Full labels: {len(labels_full)} rows")

    # 100-company held-out set
    labels_100 = pd.read_csv(config.LABELS_DIR / "labels_100.csv")
    eval_tickers = set(labels_100["symbol"].tolist())
    features_100 = pd.read_parquet(config.FEATURES_DIR / "features_100.parquet")
    print(f"  Eval set: {len(labels_100)} companies")

    # -- Build feature column list ----------------------------------------
    # Use only TF-IDF SVD + BoW SVD columns (available for full corpus)
    tfidf_cols = [c for c in full_feats.columns if c.startswith("tfidf_")]
    bow_cols   = [c for c in full_feats.columns if c.startswith("bow_")]
    feat_cols  = tfidf_cols + bow_cols
    print(f"  Sparse feature columns: {len(feat_cols)} "
          f"({len(tfidf_cols)} TF-IDF SVD + {len(bow_cols)} BoW SVD)")

    # -- Build training corpus (exclude eval tickers) ---------------------
    print("\n[2/5] Building training corpus...")

    # IMPORTANT: features_full_sparse and labels_full were both created from
    # the same transcript corpus in the same order (script 04 saves them
    # row-by-row from the same source). The correct join is POSITIONAL
    # (by row index), NOT by symbol.
    #
    # Merging on 'symbol' alone would cause a cartesian product explosion
    # because each company has ~60 transcripts across different quarters,
    # so symbol-based merge multiplies rows: 33k x 33k / N_companies - 2M rows.

    # Verify row counts match before positional join
    if len(full_feats) != len(labels_full):
        raise ValueError(
            f"Row count mismatch: features={len(full_feats)}, "
            f"labels={len(labels_full)}. "
            f"Cannot do positional join. Re-run 03 and 04 together."
        )

    # Positional join: reset both to integer index and concatenate columns
    full_feats_reset = full_feats.reset_index(drop=True)
    labels_full_reset = labels_full.reset_index(drop=True)

    # Ensure symbol column is in features (from 04, it is saved as a column)
    if "symbol" not in full_feats_reset.columns:
        # Try getting symbol from labels (same row order)
        full_feats_reset.insert(0, "symbol", labels_full_reset["symbol"])

    # Attach label columns to feature rows positionally
    merged_full = full_feats_reset.copy()
    merged_full["label_binary"]  = labels_full_reset["label_binary"].values
    merged_full["label_ternary"] = labels_full_reset["label_ternary"].values
    # Carry date through for the temporal leakage gate in the next step
    if "date" in labels_full_reset.columns:
        merged_full["date"] = labels_full_reset["date"].values
    # Also ensure symbol column matches
    if "symbol" in labels_full_reset.columns:
        merged_full["symbol"] = labels_full_reset["symbol"].values

    print(f"  After positional join: {len(merged_full)} rows (should be ~33k)")

    # Strict exclusion of eval tickers from training
    train_df = merged_full[~merged_full["symbol"].isin(eval_tickers)].copy()
    print(f"  After eval exclusion: {len(train_df)} training transcripts")
    print(f"  Eval tickers excluded: {len(eval_tickers)}")

    # ── Temporal leakage gate ─────────────────────────────────────────────
    # Prevent future-data leakage: the 100-company eval set uses each
    # company's MOST RECENT transcript. Any full-corpus transcript dated
    # on or after the earliest eval-set date could contain information
    # that was unavailable when those eval calls were made.
    # Gate: keep only training transcripts strictly BEFORE earliest eval date.
    if getattr(config, "XGB_FULLCORPUS_TEMPORAL_GATE", True):
        eval_dates = pd.to_datetime(labels_100["date"], errors="coerce").dropna()
        if len(eval_dates) > 0:
            cutoff_date = eval_dates.min()
            print(f"\n  Temporal gate active: cutoff = {cutoff_date.date()} "
                  f"(earliest date in eval set)")
            if "date" in train_df.columns:
                train_dates = pd.to_datetime(train_df["date"], errors="coerce")
                before_mask = train_dates < cutoff_date
                n_before = before_mask.sum()
                n_dropped = (~before_mask).sum()
                train_df = train_df[before_mask].copy()
                print(f"  After temporal gate: {n_before} kept, "
                      f"{n_dropped} dropped (on or after cutoff)")
            else:
                print("  WARNING: 'date' column not found in train_df — "
                      "temporal gate skipped. Merge labels_full dates in.")
        else:
            print("  WARNING: could not parse eval dates — temporal gate skipped.")
    else:
        print("  Temporal gate disabled (XGB_FULLCORPUS_TEMPORAL_GATE=False).")

    # -- Build eval feature matrix ----------------------------------------
    # features_100.parquet is indexed by symbol — reset for consistency
    if features_100.index.name == "symbol":
        eval_feats = features_100.reset_index()
    else:
        eval_feats = features_100.copy()
        if "symbol" not in eval_feats.columns:
            eval_feats = eval_feats.reset_index()

    # Merge eval features with labels_100 on symbol (safe: 1 row per company)
    eval_df = eval_feats.merge(
        labels_100[["symbol", "label_binary", "label_ternary"]],
        on="symbol", how="inner"
    )
    print(f"  Eval samples after merge: {len(eval_df)} (should be 100)")

    # Only keep feat_cols that exist in both train and eval
    feat_cols = [c for c in feat_cols
                 if c in train_df.columns and c in eval_df.columns]
    print(f"  Common feature columns: {len(feat_cols)}")

    # -- Run experiments --------------------------------------------------
    print("\n[3/5] Training and evaluating...")
    all_results = {}
    all_prediction_dfs = {}

    for task in config.TASKS:
        label_col = f"label_{task}"
        print(f"\n  === Task: {task} ===")

        # Training data (drop rows with missing labels)
        train_valid = train_df[train_df[label_col].notna()].copy()
        print(f"  Training samples: {len(train_valid)}")
        print(f"  Training label distribution: "
              f"{train_valid[label_col].value_counts().to_dict()}")

        # Eval data
        eval_valid = eval_df[eval_df[label_col].notna()].copy()
        print(f"  Eval samples: {len(eval_valid)}")
        print(f"  Eval label distribution: "
              f"{eval_valid[label_col].value_counts().to_dict()}")

        # Encode labels — fit on training set
        le = LabelEncoder()
        le.fit(train_valid[label_col].values)
        # Ensure eval labels are in the same space
        eval_labels_known = eval_valid[
            eval_valid[label_col].isin(le.classes_)
        ].copy()
        if len(eval_labels_known) < len(eval_valid):
            print(f"  WARNING: {len(eval_valid) - len(eval_labels_known)} eval "
                  f"samples have unseen labels — dropped")
            eval_valid = eval_labels_known

        y_train_full = le.transform(train_valid[label_col].values)
        y_eval = le.transform(eval_valid[label_col].values)
        n_classes = len(le.classes_)
        print(f"  Classes: {list(le.classes_)}")

        X_train_full = train_valid[feat_cols].values.astype(np.float32)
        X_eval = eval_valid[feat_cols].values.astype(np.float32)

        # Split training into train/val for Optuna tuning (80/20)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.20,
            stratify=y_train_full,
            random_state=config.RANDOM_SEED,
        )
        print(f"  Tuning split: {len(X_tr)} train / {len(X_val)} val")

        # Tune and train
        model, best_params = tune_and_train(
            X_tr, y_tr, X_val, y_val, n_classes,
            n_trials=config.OPTUNA_N_TRIALS,
        )

        # Save model
        joblib.dump(model, config.MODELS_DIR / f"xgb_fullcorpus_{task}.joblib")
        joblib.dump(le,    config.MODELS_DIR / f"xgb_fullcorpus_le_{task}.joblib")

        # Evaluate on held-out 100-company set
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)
        label_names = list(le.classes_)

        macro_f1  = f1_score(y_eval, y_pred, average="macro", zero_division=0)
        acc       = accuracy_score(y_eval, y_pred)
        kappa     = cohen_kappa_score(y_eval, y_pred)
        cm        = confusion_matrix(y_eval, y_pred, labels=range(n_classes))
        report    = classification_report(
            y_eval, y_pred, target_names=label_names,
            output_dict=True, zero_division=0,
        )

        # AUC-ROC using true probability scores
        try:
            from sklearn.preprocessing import label_binarize
            if n_classes == 2:
                auc = float(roc_auc_score(y_eval, y_prob[:, 1]))
            else:
                yt_bin = label_binarize(y_eval, classes=range(n_classes))
                present = [i for i in range(n_classes) if i in y_eval]
                auc = float(roc_auc_score(
                    yt_bin[:, present], y_prob[:, present],
                    average="macro", multi_class="ovr",
                ))
        except Exception as e:
            print(f"  AUC warning: {e}")
            auc = float("nan")

        print(f"\n  -- Results on 100-company held-out set --")
        print(f"  Macro F1 : {macro_f1:.4f}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}" if auc == auc else "  AUC-ROC  : N/A")
        print(f"  Kappa    : {kappa:.4f}")
        print(classification_report(y_eval, y_pred, target_names=label_names,
                                    zero_division=0))

        all_results[f"{task}_XGB-fullcorpus"] = {
            "task": task,
            "feature_set": "tfidf_bow_svd_80",
            "n_train": len(X_train_full),
            "n_eval": len(X_eval),
            "f1_macro": macro_f1,
            "accuracy": acc,
            "auc_roc": auc,
            "cohen_kappa": kappa,
            "best_params": best_params,
            "report": report,
            "confusion_matrix": cm.tolist(),
        }

        # Save predictions for 08_evaluation.py compatibility
        pred_labels = le.inverse_transform(y_pred)
        confidences = y_prob.max(axis=1)
        eval_symbols = eval_valid["symbol"].tolist()

        pred_df = pd.DataFrame({
            "symbol": eval_symbols,
            f"xgb_pred_{task}": pred_labels,
            f"xgb_conf_{task}": confidences,
        })
        pred_path = config.RESULTS_DIR / f"xgb_fullcorpus_predictions_{task}.csv"
        pred_df.to_csv(pred_path, index=False)
        all_prediction_dfs[task] = pred_df

        # Save confusion matrix figure
        plot_confusion(
            cm, label_names,
            title=f"XGB-fullcorpus — {task.title()} (N_train={len(X_train_full):,})",
            path=str(config.FIGURES_DIR / f"xgb_fullcorpus_confusion_{task}.png"),
        )

    # -- Summary comparison -----------------------------------------------
    print("\n[4/5] Comparison summary (XGB-fullcorpus vs XGB-100)...")

    xgb_100_path = config.RESULTS_DIR / "xgb_results.json"
    if xgb_100_path.exists():
        with open(xgb_100_path) as f:
            xgb_100_results = json.load(f)

        print(f"\n  {'Task':<10} {'Config':<25} {'Macro F1':>9} {'Accuracy':>9} {'Kappa':>8}")
        print(f"  {'-'*65}")
        for task in config.TASKS:
            # 100-company nested CV
            for cfg in ["XGB-full", "XGB-sentiment-only"]:
                key = f"{task}_{cfg}"
                if key in xgb_100_results:
                    r = xgb_100_results[key]
                    print(f"  {task:<10} {cfg:<25} "
                          f"{r['f1_macro']:>9.4f} "
                          f"{r['accuracy']:>9.4f} "
                          f"{r.get('cohen_kappa', float('nan')):>8.4f}  (N=100, nested CV)")
            # Full corpus
            key = f"{task}_XGB-fullcorpus"
            if key in all_results:
                r = all_results[key]
                n = r['n_train']
                print(f"  {task:<10} {'XGB-fullcorpus':<25} "
                      f"{r['f1_macro']:>9.4f} "
                      f"{r['accuracy']:>9.4f} "
                      f"{r['cohen_kappa']:>8.4f}  (N_train={n:,}, held-out eval)")

    # -- Save results -----------------------------------------------------
    print("\n[5/5] Saving results...")

    out_path = config.RESULTS_DIR / "xgb_fullcorpus_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results: {out_path}")

    for task, pred_df in all_prediction_dfs.items():
        p = config.RESULTS_DIR / f"xgb_fullcorpus_predictions_{task}.csv"
        print(f"  Predictions ({task}): {p}")

    for task in config.TASKS:
        p = config.FIGURES_DIR / f"xgb_fullcorpus_confusion_{task}.png"
        print(f"  Confusion matrix ({task}): {p}")

    print(f"\n{'=' * 70}")
    print("STEP 5b COMPLETE")
    print(f"{'=' * 70}")
    print("\nNOTE: To include XGB-fullcorpus in 08_evaluation.py results,")
    print("add the following to the XGB section of 08_evaluation.py:")
    print("  Load xgb_fullcorpus_results.json and xgb_fullcorpus_predictions_*.csv")
    print("  Add 'XGB-fullcorpus' rows to all_metrics before plotting.")


if __name__ == "__main__":
    main()