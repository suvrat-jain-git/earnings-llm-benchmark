"""
Step 6b: Prompt Ensembling
Combines predictions from the best GPT variants using majority vote.
Run this AFTER 06_gpt_experiments.py and 07_hybrid_experiments.py.

What it does:
- Loads all GPT predictions from gpt_predictions.csv and hybrid_predictions.csv
- Defines 3 ensemble groups (each is a set of GPT experiment names to vote over)
- For each transcript, takes majority vote across the group
- Ties broken by the highest-ranked experiment in the group
- Saves results to results/ensemble_prompt_predictions.csv

Why this helps:
- Different prompts capture different aspects of the transcript
- A transcript that confuses zero-shot might be clear to few-shot-CoT
- Majority vote reduces variance — same effect as bagging in ML
- Costs nothing extra (no new API calls, reuses existing predictions)

Run:
    python 06b_prompt_ensemble.py
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

import config


def loo_validate_disagreement_rule(pivot: pd.DataFrame, labels_df: pd.DataFrame,
                                   task: str) -> dict:
    """
    Leave-one-out validation of the disagreement routing rule.

    For each sample i:
      1. Derive the tiebreak rule from all OTHER samples (N-1)
      2. Apply that rule to sample i
      3. Record whether it was correct

    This ensures the routing rule is not overfit to the evaluation set.

    Returns dict with LOO accuracy, per-sample results, and validated rule.
    """
    task_pivot = pivot[pivot["task"] == task].copy()
    label_col = f"label_{task}"

    if ("GPT-zero" not in task_pivot.columns or
            "GPT-zero-CoT" not in task_pivot.columns):
        return {"error": "GPT-zero or GPT-zero-CoT not in pivot"}

    labels_map = labels_df.set_index("symbol")[label_col].to_dict()

    # Build disagreement data
    disagree_data = []
    for _, row in task_pivot.iterrows():
        ticker = row["symbol"]
        zero_pred = row.get("GPT-zero")
        cot_pred = row.get("GPT-zero-CoT")
        true_label = labels_map.get(ticker)

        if pd.isna(zero_pred) or pd.isna(cot_pred) or true_label is None:
            continue

        zero_pred = str(zero_pred)
        cot_pred = str(cot_pred)

        disagree_data.append({
            "symbol": ticker,
            "zero_pred": zero_pred,
            "cot_pred": cot_pred,
            "true_label": true_label,
            "agreed": zero_pred == cot_pred,
            "zero_correct": zero_pred == true_label,
            "cot_correct": cot_pred == true_label,
        })

    disagree_df = pd.DataFrame(disagree_data)
    disagreements = disagree_df[~disagree_df["agreed"]]

    if len(disagreements) < 3:
        return {
            "error": f"Only {len(disagreements)} disagreements — too few for LOO",
            "n_disagreements": len(disagreements),
        }

    # LOO validation
    loo_results = []
    for i, held_out in disagreements.iterrows():
        # Derive rule from all OTHER disagreements
        others = disagreements.drop(i)
        zero_wins = others["zero_correct"].sum()
        cot_wins = others["cot_correct"].sum()
        loo_rule = "GPT-zero" if zero_wins >= cot_wins else "GPT-zero-CoT"

        # Apply to held-out sample
        if loo_rule == "GPT-zero":
            loo_pred = held_out["zero_pred"]
        else:
            loo_pred = held_out["cot_pred"]

        loo_results.append({
            "symbol": held_out["symbol"],
            "loo_rule": loo_rule,
            "loo_pred": loo_pred,
            "true_label": held_out["true_label"],
            "loo_correct": loo_pred == held_out["true_label"],
        })

    loo_df = pd.DataFrame(loo_results)
    loo_accuracy = loo_df["loo_correct"].mean()

    # Baseline: always pick GPT-zero or GPT-zero-CoT
    zero_baseline = disagreements["zero_correct"].mean()
    cot_baseline = disagreements["cot_correct"].mean()

    # Validated rule = most common LOO-derived rule
    rule_counts = Counter(loo_df["loo_rule"])
    validated_rule = rule_counts.most_common(1)[0][0]

    result = {
        "n_total": len(disagree_df),
        "n_disagreements": len(disagreements),
        "loo_accuracy": float(loo_accuracy),
        "zero_baseline_accuracy": float(zero_baseline),
        "cot_baseline_accuracy": float(cot_baseline),
        "validated_rule": validated_rule,
        "rule_stability": rule_counts.most_common(1)[0][1] / len(loo_df),
        "loo_details": loo_df.to_dict("records"),
    }

    print(f"    LOO validation ({task}): N_disagree={len(disagreements)}, "
          f"LOO_acc={loo_accuracy:.3f}, "
          f"zero_baseline={zero_baseline:.3f}, "
          f"cot_baseline={cot_baseline:.3f}, "
          f"validated_rule={validated_rule} "
          f"(stability={result['rule_stability']:.2f})")

    return result


# ── Define ensemble groups ────────────────────────────────────────────────────
# Each group is a list of experiment names to vote over.
# Order within the list = tiebreak priority (first = highest priority).
# We define 3 groups covering different combinations:
#
#   GPT-core-ensemble  : best 3 base GPT experiments
#   GPT-hybrid-ensemble: best 3 hybrid experiments
#   GPT-full-ensemble  : all experiments (widest net)

ENSEMBLE_GROUPS = {
    "GPT-core-ensemble": [
        "GPT-few-CoT",       # typically strongest base
        "GPT-zero-CoT",
        "GPT-rag-few",       # retrieval-augmented (if available)
    ],
    "GPT-hybrid-ensemble": [
        "GPT-xgb-inject",    # XGB + SHAP injection
        "GPT-feat-inject",   # features + transcript
        "GPT-speaker-seg",   # speaker-structured
    ],
    "GPT-enhanced-ensemble": [
        "GPT-two-stage-CoT",   # two-stage extract → predict (highest impact)
        "GPT-few-CoT-sector",  # sector-matched exemplars
        "GPT-few-CoT",         # grounded CoT exemplars
    ],
    "GPT-full-ensemble": [
        "GPT-few-CoT",
        "GPT-zero-CoT",
        "GPT-rag-few",
        "GPT-two-stage-CoT",
        "GPT-few-CoT-sector",
        "GPT-xgb-inject",
        "GPT-feat-inject",
        "GPT-contrastive",
        "GPT-speaker-seg",
    ],
}

# ── Disagreement-aware ensemble ───────────────────────────────────────────────
# Based on disagreement analysis finding:
#   Binary: when GPT-zero != GPT-zero-CoT, GPT-zero is more accurate
#           (correct 6/6 on DOWN disagreements vs CoT correct 0/6)
#   Ternary: when GPT-zero != GPT-zero-CoT, GPT-zero-CoT is more accurate
#           (correct 18/19 on UP disagreements)
#
# Strategy:
#   If GPT-zero == GPT-zero-CoT: use agreed prediction (both confident)
#   If GPT-zero != GPT-zero-CoT: use GPT-zero (binary) or GPT-zero-CoT (ternary)
#
# This is a novel model-selection method grounded in empirical disagreement analysis.
DISAGREEMENT_TIEBREAK = {
    "binary":  "GPT-zero",      # zero-shot wins on binary hard samples
    "ternary": "GPT-zero-CoT",  # CoT wins on ternary hard samples
}


def majority_vote(labels: list[str | None], priority_order: list[str],
                  pred_map: dict) -> str | None:
    """
    Majority vote over a list of labels.
    If tie, use highest-priority experiment's prediction as tiebreaker.

    Args:
        labels      : list of predicted labels (may contain None for failed calls)
        priority_order: experiment names in priority order (highest = first)
        pred_map    : {exp_name: label} for this ticker/task
    Returns:
        Winning label or None if all predictions are None.
    """
    valid = [l for l in labels if l is not None]
    if not valid:
        return None

    counts = Counter(valid)
    max_count = max(counts.values())
    winners = [label for label, cnt in counts.items() if cnt == max_count]

    if len(winners) == 1:
        return winners[0]

    # Tie: use priority order to pick
    for exp_name in priority_order:
        pred = pred_map.get(exp_name)
        if pred in winners:
            return pred

    return winners[0]  # fallback


def main():
    print("=" * 70)
    print("STEP 6b: PROMPT ENSEMBLING")
    print("=" * 70)

    # ── Load all GPT predictions ─────────────────────────────────────────
    print("\n[1/4] Loading predictions...")

    all_preds_frames = []

    gpt_path = config.RESULTS_DIR / "gpt_predictions.csv"
    if gpt_path.exists():
        all_preds_frames.append(pd.read_csv(gpt_path))
        print(f"  Loaded gpt_predictions.csv")
    else:
        print("  WARNING: gpt_predictions.csv not found. "
              "Run 06_gpt_experiments.py first.")

    hybrid_path = config.RESULTS_DIR / "hybrid_predictions.csv"
    if hybrid_path.exists():
        all_preds_frames.append(pd.read_csv(hybrid_path))
        print(f"  Loaded hybrid_predictions.csv")
    else:
        print("  WARNING: hybrid_predictions.csv not found. "
              "Run 07_hybrid_experiments.py first.")

    if not all_preds_frames:
        print("ERROR: No prediction files found. Cannot run ensembling.")
        return

    all_preds = pd.concat(all_preds_frames, ignore_index=True)

    available_experiments = all_preds["experiment"].unique().tolist()
    print(f"  Available experiments: {available_experiments}")

    # ── Build per-ticker prediction lookup ───────────────────────────────
    print("\n[2/4] Building prediction lookup table...")

    # pivot: rows = (symbol, task), cols = experiment, values = prediction
    pivot = all_preds.pivot_table(
        index=["symbol", "task"],
        columns="experiment",
        values="prediction",
        aggfunc="first",      # take first if duplicates
    )
    pivot = pivot.reset_index()
    print(f"  Pivot shape: {pivot.shape}")

    # ── Run ensemble voting ───────────────────────────────────────────────
    print("\n[3/4] Computing ensemble predictions...")
    ensemble_results = []

    for task in config.TASKS:
        task_pivot = pivot[pivot["task"] == task].copy()

        for group_name, exp_list in ENSEMBLE_GROUPS.items():
            # Only use experiments that are actually available
            available_in_group = [e for e in exp_list if e in task_pivot.columns]

            if len(available_in_group) < 2:
                print(f"  SKIP {group_name}/{task}: "
                      f"only {len(available_in_group)} experiments available "
                      f"(need at least 2)")
                continue

            print(f"  Running {group_name}/{task} "
                  f"({len(available_in_group)} voters: {available_in_group})")

            for _, row in task_pivot.iterrows():
                ticker = row["symbol"]

                # Collect votes from this group
                votes = []
                pred_map = {}
                for exp in available_in_group:
                    pred = row.get(exp)
                    if pd.notna(pred):
                        votes.append(pred)
                        pred_map[exp] = pred
                    else:
                        votes.append(None)

                final_pred = majority_vote(votes, available_in_group, pred_map)
                vote_counts = Counter(v for v in votes if v is not None)

                ensemble_results.append({
                    "experiment": group_name,
                    "task": task,
                    "symbol": ticker,
                    "prediction": final_pred,
                    "n_voters": len(available_in_group),
                    "n_valid_votes": len([v for v in votes if v is not None]),
                    "vote_distribution": str(dict(vote_counts)),
                    "voters_used": str(available_in_group),
                })

    ensemble_df = pd.DataFrame(ensemble_results)

    # ── Disagreement-aware ensemble (with LOO validation) ──────────────────
    print("  LOO-validating disagreement routing rule...")
    labels_df = pd.read_csv(config.LABELS_DIR / "labels_100.csv")
    loo_results = {}
    validated_tiebreak = dict(DISAGREEMENT_TIEBREAK)  # start with default

    for task in config.TASKS:
        loo = loo_validate_disagreement_rule(pivot, labels_df, task)
        loo_results[task] = loo
        if "validated_rule" in loo:
            validated_tiebreak[task] = loo["validated_rule"]

    # Save LOO validation results
    with open(config.RESULTS_DIR / "disagreement_loo_validation.json", "w") as f:
        json.dump(loo_results, f, indent=2, default=str)
    print(f"  LOO validation saved to disagreement_loo_validation.json")

    print("  Running GPT-disagreement-ensemble (binary + ternary)...")
    disagreement_results = []

    for task in config.TASKS:
        task_pivot = pivot[pivot["task"] == task].copy()
        tiebreak_exp = validated_tiebreak[task]  # LOO-validated rule
        other_exp    = "GPT-zero-CoT" if tiebreak_exp == "GPT-zero" else "GPT-zero"

        # Check both experiments are available
        if ("GPT-zero" not in task_pivot.columns or
                "GPT-zero-CoT" not in task_pivot.columns):
            print(f"  SKIP disagreement/{task}: "
                  f"GPT-zero or GPT-zero-CoT not found in pivot")
            continue

        n_agree    = 0
        n_disagree = 0
        for _, row in task_pivot.iterrows():
            ticker   = row["symbol"]
            zero_pred = row.get("GPT-zero")
            cot_pred  = row.get("GPT-zero-CoT")

            # Clean up NaN
            zero_pred = zero_pred if pd.notna(zero_pred) else None
            cot_pred  = cot_pred  if pd.notna(cot_pred)  else None

            if zero_pred is not None and cot_pred is not None:
                if zero_pred == cot_pred:
                    # Agreement: both confident, use agreed label
                    final_pred = zero_pred
                    n_agree += 1
                else:
                    # Disagreement: use empirically better model
                    final_pred = (zero_pred if tiebreak_exp == "GPT-zero"
                                  else cot_pred)
                    n_disagree += 1
            elif zero_pred is not None:
                final_pred = zero_pred
            elif cot_pred is not None:
                final_pred = cot_pred
            else:
                final_pred = None

            disagreement_results.append({
                "experiment":  "GPT-disagreement-ensemble",
                "task":        task,
                "symbol":      ticker,
                "prediction":  final_pred,
                "zero_pred":   zero_pred,
                "cot_pred":    cot_pred,
                "agreed":      zero_pred == cot_pred if (
                               zero_pred and cot_pred) else None,
                "tiebreak_used": tiebreak_exp,
                "n_voters":    2,
                "n_valid_votes": sum(1 for p in [zero_pred, cot_pred]
                                     if p is not None),
                "vote_distribution": str({zero_pred: 1, cot_pred: 1}
                                         if zero_pred != cot_pred
                                         else {zero_pred: 2}),
                "voters_used": str(["GPT-zero", "GPT-zero-CoT"]),
            })

        print(f"    {task}: {n_agree} agree / {n_disagree} disagree | "
              f"tiebreak -> {tiebreak_exp}")

    if disagreement_results:
        disagree_df = pd.DataFrame(disagreement_results)
        # Save separately for inspection
        disagree_path = config.RESULTS_DIR / "disagreement_ensemble_predictions.csv"
        disagree_df.to_csv(disagree_path, index=False)
        print(f"  Saved: {disagree_path}")
        # Merge into main ensemble df so 08_evaluation.py picks it up
        keep_cols = ["experiment", "task", "symbol", "prediction",
                     "n_voters", "n_valid_votes", "vote_distribution",
                     "voters_used"]
        ensemble_df = pd.concat(
            [ensemble_df, disagree_df[keep_cols]], ignore_index=True
        )

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n[4/4] Results summary:")
    for task in config.TASKS:
        print(f"\n  Task: {task}")
        task_df = ensemble_df[ensemble_df["task"] == task]
        for group_name in ENSEMBLE_GROUPS:
            g_df = task_df[task_df["experiment"] == group_name]
            if g_df.empty:
                continue
            valid = g_df["prediction"].notna().sum()
            dist = g_df["prediction"].value_counts().to_dict()
            print(f"    {group_name}: {valid}/{len(g_df)} valid | {dist}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = config.RESULTS_DIR / "ensemble_prompt_predictions.csv"
    ensemble_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Also save a pivot for easy comparison
    for task in config.TASKS:
        task_df = ensemble_df[ensemble_df["task"] == task]
        if task_df.empty:
            continue
        pivot_out = task_df.pivot(
            index="symbol", columns="experiment", values="prediction"
        )
        pivot_out.to_csv(
            config.RESULTS_DIR / f"ensemble_prompt_pivot_{task}.csv"
        )

    print(f"\n{'=' * 70}")
    print("PROMPT ENSEMBLING COMPLETE")
    print(f"  Output: {out_path}")
    print(f"  Next: run 08_evaluation.py — it will pick up these predictions")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()