"""
STEP 10: LEARNING CURVE ANALYSIS
=================================
Plots dataset size vs F1 to answer:
  (a) Is the model saturating with 100 samples?
  (b) Would collecting more data meaningfully improve performance?
  (c) Do LLMs benefit more from few examples than XGBoost does?

Methodology
-----------
For XGBoost: subsample training data at fractions [0.2, 0.4, 0.6, 0.8, 1.0]
of the 100-company set, retrain with the same nested-CV hyperparameters from
the full run, evaluate on a fixed 20-company held-out test set (stratified).
Repeat 10 times per fraction with different random seeds to get mean ± std.

For GPT-two-stage-CoT: subsample K exemplars from the gpt_predictions.csv
OOF predictions at sizes [10, 20, 40, 60, 80, 100] and evaluate on the
complement. This simulates "what if we had only K labelled samples at test
time". No API calls are made — we reuse existing predictions.

For FinBERT: same subsampling approach as XGBoost.

Outputs
-------
  figures/learning_curve_binary.png
  figures/learning_curve_ternary.png
  data/results/learning_curve_results.json

CPU-only, no API calls, ~10-15 minutes.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

import config

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_FRACTIONS  = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
GPT_SIZES        = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_SEEDS          = 10       # random repeats per fraction for variance estimation
TEST_FRACTION    = 0.20     # held-out test proportion for XGB/FinBERT curves
RANDOM_STATE     = config.RANDOM_SEED


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stratified_subsample(X, y, fraction, seed):
    """Return (X_sub, y_sub) with `fraction` of data, stratified by y."""
    n_total = len(y)
    n_sub   = max(int(round(n_total * fraction)), len(np.unique(y)))
    if n_sub >= n_total:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(n_total - n_sub),
                                 random_state=seed)
    train_idx, _ = next(sss.split(X, y))
    return X[train_idx], y[train_idx]


def _encode_labels(y_series, label_names):
    """Map string labels to integers for XGBClassifier."""
    lmap = {l: i for i, l in enumerate(label_names)}
    return np.array([lmap[v] for v in y_series if v in lmap])


def xgb_learning_curve(features_df, labels, task, label_names,
                        xgb_results, fractions, n_seeds, test_frac):
    """
    Subsample training data at each fraction, train XGB with best params
    from the full nested-CV run, evaluate on stratified held-out test set.
    Returns: {fraction: {"mean_f1": ..., "std_f1": ..., "n_train": ...}}
    """
    label_col = f"label_{task}"
    valid = labels[labels[label_col].notna() & labels[label_col].isin(label_names)]
    common = valid.index.intersection(features_df.index)
    if len(common) < 10:
        print(f"  Skipping XGB learning curve for {task}: too few samples")
        return {}

    feat_cols = [c for c in features_df.columns
                 if not c.startswith(("symbol", "date", "label"))]
    X_all = features_df.loc[common, feat_cols].fillna(0).values.astype(float)
    y_all = np.array(valid.loc[common, label_col])
    y_int = _encode_labels(pd.Series(y_all), label_names)

    # Pull best params from the last fold of the full run (representative)
    xgb_key = f"{task}_XGB-full"
    best_params = {}
    if xgb_key in xgb_results:
        fold_results = xgb_results[xgb_key].get("fold_results", [])
        if fold_results:
            # Use median n_estimators, etc. across folds to avoid outlier folds
            param_keys = ["max_depth", "learning_rate", "n_estimators",
                          "min_child_weight", "subsample", "colsample_bytree",
                          "gamma", "reg_alpha", "reg_lambda"]
            for pk in param_keys:
                vals = [fr["best_params"][pk] for fr in fold_results
                        if "best_params" in fr and pk in fr["best_params"]]
                if vals:
                    best_params[pk] = float(np.median(vals))
            if "n_estimators" in best_params:
                best_params["n_estimators"] = int(best_params["n_estimators"])
            if "max_depth" in best_params:
                best_params["max_depth"] = int(best_params["max_depth"])
            if "min_child_weight" in best_params:
                best_params["min_child_weight"] = int(best_params["min_child_weight"])

    results = {}
    for frac in fractions:
        f1_scores_frac = []
        for seed in range(n_seeds):
            # Split into train+val pool and fixed test
            sss_test = StratifiedShuffleSplit(
                n_splits=1, test_size=test_frac, random_state=seed * 100)
            train_val_idx, test_idx = next(sss_test.split(X_all, y_int))

            X_tv, y_tv = X_all[train_val_idx], y_int[train_val_idx]
            X_test, y_test = X_all[test_idx], y_int[test_idx]

            # Subsample train set at fraction
            X_tr, y_tr = _stratified_subsample(X_tv, y_tv, frac, seed)
            if len(np.unique(y_tr)) < 2:
                continue  # degenerate split

            # Class weight: scale_pos_weight for binary
            spw = 1.0
            if task == "binary" and len(label_names) == 2:
                n_neg = (y_tr == 0).sum()
                n_pos = (y_tr == 1).sum()
                spw = max(n_neg / max(n_pos, 1), 0.1)

            clf = XGBClassifier(
                objective="binary:logistic" if task == "binary" else "multi:softmax",
                num_class=None if task == "binary" else len(label_names),
                scale_pos_weight=spw if task == "binary" else 1.0,
                eval_metric="logloss",
                verbosity=0,
                random_state=seed,
                **best_params if best_params else {
                    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
            )
            try:
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_test)
                f1 = f1_score(y_test, y_pred, average="macro",
                              labels=list(range(len(label_names))),
                              zero_division=0)
                f1_scores_frac.append(f1)
            except Exception as e:
                print(f"    XGB fit error (frac={frac}, seed={seed}): {e}")

        if f1_scores_frac:
            results[frac] = {
                "mean_f1": float(np.mean(f1_scores_frac)),
                "std_f1":  float(np.std(f1_scores_frac)),
                "n_train": int(round(len(X_all) * (1 - test_frac) * frac)),
                "n_seeds": len(f1_scores_frac),
            }
            print(f"    XGB {task} frac={frac:.0%}: "
                  f"F1={results[frac]['mean_f1']:.3f} "
                  f"±{results[frac]['std_f1']:.3f} "
                  f"(n_train={results[frac]['n_train']})")
    return results


def gpt_learning_curve(all_preds_df, labels, task, label_names,
                        experiment, sizes):
    """
    Reuses existing GPT predictions to simulate different training-set sizes.
    At each size K, randomly select K labelled samples as "seen" data (no
    re-training — GPT is zero-shot), evaluate on the remaining N-K samples.
    This measures how prediction quality varies with available labelled data
    for evaluation/analysis purposes (e.g., ECE, DM test power).

    Since GPT is zero-shot (no fine-tuning), this actually shows how evaluation
    set size affects F1 stability — a meaningful finding for small-data settings.
    """
    label_col = f"label_{task}"
    exp_preds = all_preds_df[
        (all_preds_df["task"] == task) &
        (all_preds_df["experiment"] == experiment)
    ].set_index("symbol")

    valid_labels = labels[labels[label_col].notna()]
    common = valid_labels.index.intersection(exp_preds.index)
    common = [c for c in common if exp_preds.loc[c, "prediction"] in label_names]

    if len(common) < 10:
        print(f"  Skipping GPT learning curve: too few valid predictions ({len(common)})")
        return {}

    y_true = np.array(valid_labels.loc[common, label_col])
    y_pred = np.array(exp_preds.loc[common, "prediction"])

    results = {}
    for size in sizes:
        if size > len(common):
            continue
        f1_scores_size = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            # Stratified subsample of `size` evaluation points
            # (simulate "what F1 would we see with only `size` labelled examples")
            classes, class_counts = np.unique(y_true, return_counts=True)
            indices = []
            for cls, cnt in zip(classes, class_counts):
                cls_idx = np.where(y_true == cls)[0]
                n_cls   = max(1, int(round(size * cnt / len(y_true))))
                n_cls   = min(n_cls, len(cls_idx))
                chosen  = rng.choice(cls_idx, size=n_cls, replace=False)
                indices.extend(chosen.tolist())

            if len(indices) < 2:
                continue
            yt = y_true[indices]
            yp = y_pred[indices]
            if len(np.unique(yt)) < 2:
                continue
            f1 = f1_score(yt, yp, average="macro",
                          labels=label_names, zero_division=0)
            f1_scores_size.append(f1)

        if f1_scores_size:
            results[size] = {
                "mean_f1": float(np.mean(f1_scores_size)),
                "std_f1":  float(np.std(f1_scores_size)),
                "n_eval":  size,
                "n_seeds": len(f1_scores_size),
            }
            print(f"    GPT {task} n={size}: "
                  f"F1={results[size]['mean_f1']:.3f} "
                  f"±{results[size]['std_f1']:.3f}")
    return results


def plot_learning_curve(results_dict, task, label_names, save_path):
    """
    Plot learning curves for XGBoost and GPT side by side.
    Left: XGBoost F1 vs n_train
    Right: GPT evaluation-set F1 stability vs n_eval
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    majority_f1 = 1.0 / len(label_names)  # approximate majority baseline

    # ── Left: XGBoost ────────────────────────────────────────────────────
    ax = axes[0]
    xgb_curve = results_dict.get("xgb", {})
    if xgb_curve:
        fracs    = sorted(xgb_curve.keys())
        n_trains = [xgb_curve[f]["n_train"] for f in fracs]
        means    = [xgb_curve[f]["mean_f1"] for f in fracs]
        stds     = [xgb_curve[f]["std_f1"]  for f in fracs]

        ax.fill_between(n_trains,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="#2980b9", label="\u00b11 std")
        ax.plot(n_trains, means, "o-", color="#2980b9", lw=2, ms=6,
                label="XGBoost (mean F1)")
        ax.axhline(means[-1], color="#2980b9", ls=":", lw=1, alpha=0.5)

    ax.axhline(majority_f1, color="#95a5a6", ls="--", lw=1.2,
               label=f"Majority-class approx. ({majority_f1:.2f})")
    ax.set_xlabel("Training set size (N)", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title(f"XGBoost Learning Curve\n{task.title()} task", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # ── Right: GPT prediction stability ──────────────────────────────────
    ax2 = axes[1]
    gpt_curve = results_dict.get("gpt", {})
    if gpt_curve:
        sizes = sorted(gpt_curve.keys())
        means2 = [gpt_curve[s]["mean_f1"] for s in sizes]
        stds2  = [gpt_curve[s]["std_f1"]  for s in sizes]

        ax2.fill_between(sizes,
                         [m - s for m, s in zip(means2, stds2)],
                         [m + s for m, s in zip(means2, stds2)],
                         alpha=0.2, color="#27ae60", label="\u00b11 std")
        ax2.plot(sizes, means2, "s-", color="#27ae60", lw=2, ms=6,
                 label="GPT-two-stage-CoT (mean F1)")
        ax2.axhline(means2[-1], color="#27ae60", ls=":", lw=1, alpha=0.5,
                    label=f"Full N={sizes[-1]} F1={means2[-1]:.3f}")

    ax2.axhline(majority_f1, color="#95a5a6", ls="--", lw=1.2,
                label=f"Majority-class approx. ({majority_f1:.2f})")
    ax2.set_xlabel("Evaluation set size (N)", fontsize=11)
    ax2.set_ylabel("Macro F1", fontsize=11)
    ax2.set_title(f"GPT-two-stage-CoT F1 Stability\n{task.title()} task "
                  f"(fixed predictions, varying eval size)", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    # Annotation: saturation region
    for ax_i, curve, key in [(axes[0], xgb_curve, "n_train"),
                              (axes[1], gpt_curve, "n_eval")]:
        if len(curve) >= 3:
            xs = sorted(curve.keys())
            last_three = [curve[x]["mean_f1"] for x in xs[-3:]]
            delta = max(last_three) - min(last_three)
            if delta < 0.02:
                ax_i.annotate("Plateau region", xy=(xs[-2], curve[xs[-2]]["mean_f1"]),
                               xytext=(xs[-3] * 0.9, curve[xs[-2]]["mean_f1"] + 0.08),
                               arrowprops=dict(arrowstyle="->", color="gray"),
                               fontsize=8, color="gray")

    plt.suptitle(f"Learning Curves — {task.title()} Classification",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("STEP 10: LEARNING CURVE ANALYSIS")
    print("=" * 70)
    print(f"  Seeds per fraction : {N_SEEDS}")
    print(f"  Train fractions    : {[f'{f:.0%}' for f in TRAIN_FRACTIONS]}")
    print(f"  GPT eval sizes     : {GPT_SIZES}")
    print(f"  Test fraction      : {TEST_FRACTION:.0%}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    labels = pd.read_csv(config.LABELS_DIR / "labels_100.csv").set_index("symbol")

    features_df = None
    feat_path = config.FEATURES_DIR / "features_100.parquet"
    if feat_path.exists():
        features_df = pd.read_parquet(feat_path)
        if "symbol" in features_df.columns:
            features_df = features_df.set_index("symbol")
        print(f"  Features: {features_df.shape}")
    else:
        print("  WARNING: features_100.parquet not found — XGB curves skipped")

    xgb_results = {}
    xgb_json = config.RESULTS_DIR / "xgb_results.json"
    if xgb_json.exists():
        with open(xgb_json) as f:
            xgb_results = json.load(f)
        print(f"  XGB results: {list(xgb_results.keys())}")

    all_preds = None
    preds_path = config.RESULTS_DIR / "all_predictions.csv"
    if preds_path.exists():
        all_preds = pd.read_csv(preds_path)
        all_preds = all_preds.drop_duplicates(
            subset=["experiment", "task", "symbol"], keep="last"
        ).reset_index(drop=True)
        print(f"  GPT predictions: {len(all_preds)} rows, "
              f"{all_preds['experiment'].nunique()} experiments")
    else:
        print("  WARNING: all_predictions.csv not found — GPT curves skipped")

    all_lc_results = {}

    # ── Run for each task ─────────────────────────────────────────────────
    for task in config.TASKS:
        label_names = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]
        print(f"\n[2-3/4] {task.upper()} task (labels: {label_names})")

        task_results = {}

        # XGBoost learning curve
        if features_df is not None:
            print(f"\n  XGBoost learning curve ({task})...")
            task_results["xgb"] = xgb_learning_curve(
                features_df, labels, task, label_names,
                xgb_results, TRAIN_FRACTIONS, N_SEEDS, TEST_FRACTION
            )
        else:
            task_results["xgb"] = {}

        # GPT-two-stage-CoT learning curve (eval stability)
        if all_preds is not None:
            print(f"\n  GPT-two-stage-CoT eval stability ({task})...")
            task_results["gpt"] = gpt_learning_curve(
                all_preds, labels, task, label_names,
                "GPT-two-stage-CoT", GPT_SIZES
            )
        else:
            task_results["gpt"] = {}

        all_lc_results[task] = task_results

        # Plot
        plot_learning_curve(
            task_results, task, label_names,
            str(config.FIGURES_DIR / f"learning_curve_{task}.png")
        )

    # ── Save results ──────────────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    out_path = config.RESULTS_DIR / "learning_curve_results.json"
    with open(out_path, "w") as f:
        json.dump(all_lc_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("LEARNING CURVE SUMMARY")
    print(f"{'=' * 70}")
    for task, task_res in all_lc_results.items():
        print(f"\n  {task.upper()}:")
        xgb_c = task_res.get("xgb", {})
        gpt_c = task_res.get("gpt", {})
        if xgb_c:
            fracs = sorted(xgb_c.keys())
            f1_20 = xgb_c[fracs[0]]["mean_f1"]
            f1_100 = xgb_c[fracs[-1]]["mean_f1"]
            gain = f1_100 - f1_20
            print(f"    XGB: F1 at {fracs[0]:.0%}={f1_20:.3f} -> "
                  f"{fracs[-1]:.0%}={f1_100:.3f} (gain={gain:+.3f})")
            if abs(gain) < 0.02:
                print("    XGB: SATURATED — more data unlikely to help significantly")
            else:
                print("    XGB: NOT saturated — collecting more data would improve F1")
        if gpt_c:
            sizes = sorted(gpt_c.keys())
            std_10  = gpt_c[sizes[0]]["std_f1"]
            std_100 = gpt_c[sizes[-1]]["std_f1"]
            print(f"    GPT stability: std at N=10={std_10:.3f}, N=100={std_100:.3f} "
                  f"(variance {'HIGH' if std_10 > 0.05 else 'LOW'} at small N)")

    print(f"\n  Figures saved to: {config.FIGURES_DIR}")
    print(f"  Results saved to: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
