"""
Step 9: Comprehensive Error Analysis
Per-sector accuracy, return magnitude vs accuracy, transcript length vs accuracy,
call timing analysis, most-confused companies, and error pattern identification.

Run AFTER 08_evaluation.py.
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

import config


def load_predictions_and_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all predictions and ground-truth labels."""
    frames = []
    for path_name in ["gpt_predictions.csv", "hybrid_predictions.csv",
                      "ensemble_prompt_predictions.csv"]:
        path = config.RESULTS_DIR / path_name
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("No prediction files found in results/")

    preds = pd.concat(frames, ignore_index=True)
    labels = pd.read_csv(config.LABELS_DIR / "labels_100.csv")
    return preds, labels


def load_metadata() -> pd.DataFrame:
    """Load company metadata (sector, transcript length, call timing)."""
    companies = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    labels = pd.read_csv(config.LABELS_DIR / "labels_100.csv")

    meta = companies.merge(labels, left_on="ticker", right_on="symbol", how="inner")

    # Load transcript lengths
    import json as _json
    lengths = {}
    for _, row in companies.iterrows():
        ticker = row["ticker"]
        path = config.RAW_TRANSCRIPTS_DIR / f"{ticker}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            content = data.get("content", "")
            lengths[ticker] = len(content.split()) if content else 0
    meta["transcript_word_count"] = meta["ticker"].map(lengths)

    return meta


def per_sector_accuracy(preds: pd.DataFrame, labels: pd.DataFrame,
                        meta: pd.DataFrame, task: str,
                        experiments: list[str]) -> pd.DataFrame:
    """Compute accuracy per GICS sector for selected experiments."""
    label_col = f"label_{task}"
    labels_map = labels.set_index("symbol")[label_col].to_dict()
    sector_map = meta.set_index("ticker")["sector"].to_dict()

    task_preds = preds[preds["task"] == task].copy()
    task_preds["true_label"] = task_preds["symbol"].map(labels_map)
    task_preds["sector"] = task_preds["symbol"].map(sector_map)
    task_preds = task_preds.dropna(subset=["true_label", "sector"])

    results = []
    for exp in experiments:
        exp_df = task_preds[task_preds["experiment"] == exp].copy()
        if exp_df.empty:
            continue
        exp_df["correct"] = exp_df["prediction"] == exp_df["true_label"]

        for sector, grp in exp_df.groupby("sector"):
            n_total = len(grp)
            n_correct = grp["correct"].sum()
            n_valid = grp["prediction"].notna().sum()
            results.append({
                "experiment": exp,
                "sector": sector,
                "n_samples": n_total,
                "n_valid": n_valid,
                "accuracy": n_correct / max(n_valid, 1),
            })

    return pd.DataFrame(results)



def sector_fisher_exact_tests(preds, labels, meta, task, experiments):
    """
    Fisher's exact test for sector-level accuracy differences.

    With ~9 companies per sector, chi-squared is unreliable (expected cell
    counts < 5). Fisher's exact test is the correct non-parametric choice.

    For each sector x experiment, tests H0: sector accuracy is not
    significantly different from all other sectors combined (one-vs-rest
    2x2 contingency table). Applies Holm-Bonferroni correction within each
    experiment to control family-wise error rate.

    Returns DataFrame with columns: experiment, sector, n_sector, n_correct,
    accuracy, n_others, acc_others, fisher_p, significant,
    holm_alpha, significant_corrected.
    """
    from scipy.stats import fisher_exact
    import pandas as pd

    label_col  = f"label_{task}"
    labels_map = labels.set_index("symbol")[label_col].to_dict()
    sector_map = meta.set_index("ticker")["sector"].to_dict()
    task_preds = preds[preds["task"] == task].copy()
    task_preds["true_label"] = task_preds["symbol"].map(labels_map)
    task_preds["sector"]     = task_preds["symbol"].map(sector_map)
    task_preds = task_preds.dropna(subset=["true_label", "sector", "prediction"])

    alpha = 0.05
    rows = []

    for exp in experiments:
        exp_df = task_preds[task_preds["experiment"] == exp].copy()
        if exp_df.empty:
            continue
        exp_df["correct"] = (exp_df["prediction"] == exp_df["true_label"]).astype(int)

        for sector in exp_df["sector"].unique():
            in_s  = exp_df[exp_df["sector"] == sector]
            out_s = exp_df[exp_df["sector"] != sector]
            if len(in_s) < 2 or len(out_s) < 2:
                continue
            c_in  = int(in_s["correct"].sum())
            w_in  = len(in_s)  - c_in
            c_out = int(out_s["correct"].sum())
            w_out = len(out_s) - c_out
            _, p = fisher_exact([[c_in, w_in], [c_out, w_out]],
                                 alternative="two-sided")
            rows.append({
                "experiment": exp, "sector": sector,
                "n_sector":   len(in_s),  "n_correct":  c_in,
                "accuracy":   round(c_in / max(len(in_s), 1), 4),
                "n_others":   len(out_s),
                "acc_others": round(c_out / max(len(out_s), 1), 4),
                "fisher_p":   round(float(p), 6),
                "significant": bool(p < alpha),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Holm-Bonferroni correction within each experiment
    corrected = []
    for exp, grp in df.groupby("experiment"):
        grp_sorted = grp.sort_values("fisher_p").reset_index(drop=True)
        m    = len(grp_sorted)
        stop = False
        for rank_zero, row in grp_sorted.iterrows():
            rank = rank_zero + 1
            holm_alpha = alpha / max(m - rank + 1, 1)
            rejected   = (not stop) and (row["fisher_p"] <= holm_alpha)
            if not rejected:
                stop = True
            d = row.to_dict()
            d["holm_alpha"]            = round(holm_alpha, 6)
            d["significant_corrected"] = rejected
            corrected.append(d)

    return pd.DataFrame(corrected)


def return_magnitude_analysis(preds: pd.DataFrame, labels: pd.DataFrame,
                              task: str, experiment: str) -> pd.DataFrame:
    """Analyse accuracy as a function of return magnitude."""
    label_col = f"label_{task}"
    labels_full = labels.copy()

    task_preds = preds[(preds["task"] == task) &
                       (preds["experiment"] == experiment)].copy()
    merged = task_preds.merge(labels_full, left_on="symbol", right_on="symbol")

    if "return" not in merged.columns:
        return pd.DataFrame()

    merged["abs_return"] = merged["return"].abs()
    merged["correct"] = merged["prediction"] == merged[label_col]

    # Bin by return magnitude
    bins = [0, 0.005, 0.01, 0.02, 0.05, 1.0]
    bin_labels = ["0-0.5%", "0.5-1%", "1-2%", "2-5%", "5%+"]
    merged["return_bin"] = pd.cut(merged["abs_return"], bins=bins,
                                  labels=bin_labels, right=True)

    results = []
    for bin_label, grp in merged.groupby("return_bin", observed=True):
        n = len(grp)
        n_correct = grp["correct"].sum()
        results.append({
            "return_bin": str(bin_label),
            "n_samples": n,
            "accuracy": n_correct / max(n, 1),
            "mean_abs_return": grp["abs_return"].mean(),
        })

    return pd.DataFrame(results)


def transcript_length_analysis(preds: pd.DataFrame, labels: pd.DataFrame,
                               meta: pd.DataFrame, task: str,
                               experiment: str) -> pd.DataFrame:
    """Accuracy vs transcript length."""
    label_col = f"label_{task}"
    labels_map = labels.set_index("symbol")[label_col].to_dict()
    length_map = meta.set_index("ticker")["transcript_word_count"].to_dict()

    task_preds = preds[(preds["task"] == task) &
                       (preds["experiment"] == experiment)].copy()
    task_preds["true_label"] = task_preds["symbol"].map(labels_map)
    task_preds["word_count"] = task_preds["symbol"].map(length_map)
    task_preds = task_preds.dropna(subset=["true_label", "word_count"])
    task_preds["correct"] = task_preds["prediction"] == task_preds["true_label"]

    # Tertiles
    task_preds["length_group"] = pd.qcut(task_preds["word_count"], q=3,
                                         labels=["Short", "Medium", "Long"])

    results = []
    for group, grp in task_preds.groupby("length_group", observed=True):
        results.append({
            "length_group": str(group),
            "n_samples": len(grp),
            "accuracy": grp["correct"].mean(),
            "mean_word_count": grp["word_count"].mean(),
        })

    return pd.DataFrame(results)


def call_timing_analysis(preds: pd.DataFrame, labels: pd.DataFrame,
                         task: str, experiment: str) -> pd.DataFrame:
    """Accuracy split by pre-market vs post-market call timing."""
    label_col = f"label_{task}"

    task_preds = preds[(preds["task"] == task) &
                       (preds["experiment"] == experiment)].copy()
    merged = task_preds.merge(labels, left_on="symbol", right_on="symbol")

    if "call_time" not in merged.columns:
        return pd.DataFrame()

    merged["correct"] = merged["prediction"] == merged[label_col]

    results = []
    for timing, grp in merged.groupby("call_time"):
        results.append({
            "call_timing": str(timing),
            "n_samples": len(grp),
            "accuracy": grp["correct"].mean(),
        })

    return pd.DataFrame(results)


def most_confused_companies(preds: pd.DataFrame, labels: pd.DataFrame,
                            task: str, top_n: int = 10) -> pd.DataFrame:
    """Find companies most frequently misclassified across all experiments."""
    label_col = f"label_{task}"
    labels_map = labels.set_index("symbol")[label_col].to_dict()

    task_preds = preds[preds["task"] == task].copy()
    task_preds["true_label"] = task_preds["symbol"].map(labels_map)
    task_preds = task_preds.dropna(subset=["true_label"])
    task_preds["correct"] = task_preds["prediction"] == task_preds["true_label"]

    # Error rate per company across all experiments
    company_errors = task_preds.groupby("symbol").agg(
        n_experiments=("correct", "count"),
        n_correct=("correct", "sum"),
    )
    company_errors["error_rate"] = 1 - company_errors["n_correct"] / company_errors["n_experiments"]
    company_errors = company_errors.sort_values("error_rate", ascending=False)

    return company_errors.head(top_n).reset_index()


def plot_sector_heatmap(sector_df: pd.DataFrame, task: str):
    """Plot sector × experiment accuracy heatmap."""
    if sector_df.empty:
        return

    pivot = sector_df.pivot(index="sector", columns="experiment", values="accuracy")
    pivot = pivot.fillna(0.5)

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.2),
                                    max(6, len(pivot.index) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    ax.set_title(f"Per-Sector Accuracy — {task.title()} Classification", fontsize=12)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"error_sector_heatmap_{task}.png", dpi=200)
    plt.close()


def plot_return_vs_accuracy(return_df: pd.DataFrame, task: str, experiment: str):
    """Bar chart of accuracy by return magnitude bin."""
    if return_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(return_df["return_bin"], return_df["accuracy"],
                  color="#4a90d9", edgecolor="black", linewidth=0.5)

    # Annotate N on each bar
    for bar, n in zip(bars, return_df["n_samples"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Absolute Return Magnitude")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Return Magnitude — {experiment} / {task.title()}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"error_return_vs_accuracy_{task}.png", dpi=200)
    plt.close()


def plot_confused_companies(confused_df: pd.DataFrame, task: str):
    """Horizontal bar chart of most-confused companies."""
    if confused_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(confused_df))
    colors = plt.cm.Reds(confused_df["error_rate"].values)
    ax.barh(y_pos, confused_df["error_rate"], color=colors, edgecolor="black",
            linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(confused_df["symbol"], fontsize=10)
    ax.set_xlabel("Error Rate (across all experiments)")
    ax.set_title(f"Most-Confused Companies — {task.title()} Classification")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"error_most_confused_{task}.png", dpi=200)
    plt.close()


def main():
    print("=" * 70)
    print("STEP 9: COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/6] Loading predictions and metadata...")
    preds, labels = load_predictions_and_labels()
    meta = load_metadata()

    available_exps = preds["experiment"].unique().tolist()
    print(f"  Available experiments: {len(available_exps)}")
    print(f"  Companies with metadata: {len(meta)}")

    # Key experiments to focus error analysis on
    key_experiments = [e for e in [
        "GPT-zero", "GPT-zero-CoT", "GPT-few-CoT", "GPT-feat-inject",
        "GPT-xgb-inject", "GPT-contrastive", "GPT-speaker-seg",
        "GPT-disagreement-ensemble",
    ] if e in available_exps]
    print(f"  Key experiments for analysis: {key_experiments}")

    all_analysis = {}

    for task in config.TASKS:
        print(f"\n{'='*50}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*50}")

        task_analysis = {}

        # ── Per-sector accuracy ──────────────────────────────────────
        print("\n[2/6] Per-sector accuracy...")
        sector_df = per_sector_accuracy(preds, labels, meta, task, key_experiments)
        if not sector_df.empty:
            task_analysis["per_sector"] = sector_df.to_dict("records")
            plot_sector_heatmap(sector_df, task)
            print(f"  Saved sector heatmap")

            # Identify weakest sectors
            sector_avg = sector_df.groupby("sector")["accuracy"].mean()
            worst_sectors = sector_avg.nsmallest(3)
            print(f"  Weakest sectors:")
            for s, acc in worst_sectors.items():
                print(f"    {s}: {acc:.3f}")

            # ── Fisher's exact test: which sector differences are significant?
            print(f"  Running Fisher's exact tests per sector...")
            fisher_df = sector_fisher_exact_tests(
                preds, labels, meta, task, key_experiments
            )
            if not fisher_df.empty:
                task_analysis["sector_fisher"] = fisher_df.to_dict("records")
                n_sig_raw  = int(fisher_df["significant"].sum())
                n_sig_corr = int(fisher_df["significant_corrected"].sum())
                print(f"  Fisher results: {n_sig_raw} significant (uncorrected), "
                      f"{n_sig_corr} after Holm-Bonferroni correction")
                if n_sig_corr > 0:
                    sig_rows = fisher_df[fisher_df["significant_corrected"]]
                    for _, row in sig_rows.iterrows():
                        print(f"    ** {row['experiment']} / {row['sector']}: "
                              f"acc={row['accuracy']:.3f} vs others={row['acc_others']:.3f} "
                              f"p={row['fisher_p']:.4f}")
                # Save Fisher results to CSV for paper
                fisher_path = config.RESULTS_DIR / f"sector_fisher_{task}.csv"
                fisher_df.to_csv(fisher_path, index=False)
                print(f"  Saved: {fisher_path}")

        # ── Return magnitude analysis ────────────────────────────────
        print("\n[3/6] Return magnitude vs accuracy...")
        best_exp = "GPT-zero" if "GPT-zero" in available_exps else key_experiments[0]
        return_df = return_magnitude_analysis(preds, labels, task, best_exp)
        if not return_df.empty:
            task_analysis["return_magnitude"] = return_df.to_dict("records")
            plot_return_vs_accuracy(return_df, task, best_exp)
            print(f"  Saved return vs accuracy plot")

        # ── Transcript length analysis ───────────────────────────────
        print("\n[4/6] Transcript length vs accuracy...")
        length_df = transcript_length_analysis(preds, labels, meta, task, best_exp)
        if not length_df.empty:
            task_analysis["transcript_length"] = length_df.to_dict("records")
            for _, row in length_df.iterrows():
                print(f"    {row['length_group']}: acc={row['accuracy']:.3f} "
                      f"(n={row['n_samples']}, ~{row['mean_word_count']:.0f} words)")

        # ── Call timing analysis ─────────────────────────────────────
        print("\n[5/6] Call timing analysis...")
        timing_df = call_timing_analysis(preds, labels, task, best_exp)
        if not timing_df.empty:
            task_analysis["call_timing"] = timing_df.to_dict("records")
            for _, row in timing_df.iterrows():
                print(f"    {row['call_timing']}: acc={row['accuracy']:.3f} "
                      f"(n={row['n_samples']})")

        # ── Most-confused companies ──────────────────────────────────
        print("\n[6/6] Most-confused companies...")
        confused_df = most_confused_companies(preds, labels, task)
        if not confused_df.empty:
            task_analysis["most_confused"] = confused_df.to_dict("records")
            plot_confused_companies(confused_df, task)
            print(f"  Top confused:")
            for _, row in confused_df.head(5).iterrows():
                print(f"    {row['symbol']}: error_rate={row['error_rate']:.3f} "
                      f"({row['n_correct']}/{row['n_experiments']} correct)")

        all_analysis[task] = task_analysis

    # ── Save all analysis ────────────────────────────────────────────────
    out_path = config.RESULTS_DIR / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_analysis, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("ERROR ANALYSIS COMPLETE")
    print(f"  Results: {out_path}")
    print(f"  Figures: {config.FIGURES_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
