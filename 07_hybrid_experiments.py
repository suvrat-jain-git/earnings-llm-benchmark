"""
Step 7: Hybrid Feature-Injection Experiments
5 hybrid strategies × 100 transcripts × 2 tasks (binary/ternary).
Plus ensemble (majority vote of XGBoost + best GPT).
"""
import json
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

import config
from utils.azure_openai_client import AzureGPTClient
from utils.transcript_parser import (
    parse_structured_content, parse_raw_content, truncate_transcript,
)
from utils.prompt_templates import (
    feat_inject, feat_only, speaker_seg, contrastive, xgb_inject,
    format_features_summary,
)
from utils.price_utils import assign_label_ternary, assign_label_binary
from utils.prediction_utils import parse_prediction  # shared; do not redefine locally


def main():
    print("=" * 70)
    print("STEP 7: HYBRID FEATURE-INJECTION EXPERIMENTS")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    client = AzureGPTClient()

    companies = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    features_df = pd.read_parquet(config.FEATURES_DIR / "features_100.parquet")
    sector_avgs = pd.read_parquet(config.FEATURES_DIR / "sector_averages.parquet")

    # Prefer LOO sector averages (excludes each company from its own sector average)
    loo_path = config.FEATURES_DIR / "sector_averages_loo.parquet"
    if loo_path.exists():
        sector_avgs_loo = pd.read_parquet(loo_path)
        print(f"  Using LOO sector averages (no self-influence)")
    else:
        sector_avgs_loo = None
        print(f"  WARNING: LOO sector averages not found, using standard averages")
    shap_data = joblib.load(config.RESULTS_DIR / "shap_data.joblib")

    company_sectors = companies.set_index("ticker")["sector"].to_dict()

    # Load transcripts
    transcripts = {}
    for _, row in companies.iterrows():
        ticker = row["ticker"]
        path = config.RAW_TRANSCRIPTS_DIR / f"{ticker}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            parsed = parse_structured_content(data.get("structured_content"))
            if parsed is None:
                parsed = parse_raw_content(data.get("content", ""))
            if parsed:
                transcripts[ticker] = {
                    "parsed": parsed,
                    "meta": {
                        "symbol": ticker,
                        "company_name": data.get("company_name", row.get("company_name", "")),
                        "date": data.get("date", ""),
                        "quarter": f"Q{data.get('quarter', '?')} {data.get('year', '?')}",
                    }
                }

    # Load XGBoost predictions
    xgb_preds = {}
    for task in config.TASKS:
        xgb_path = config.RESULTS_DIR / f"xgb_predictions_{task}.csv"
        if xgb_path.exists():
            df = pd.read_csv(xgb_path)
            xgb_preds[task] = df.set_index("symbol")

    print(f"  Transcripts: {len(transcripts)}")
    print(f"  Features: {features_df.shape}")

    # ── Run hybrid experiments ───────────────────────────────────────────
    print("\n[2/4] Running hybrid experiments...")
    all_predictions = []
    tickers = sorted(transcripts.keys())

    for task in config.TASKS:
        print(f"\n  === Task: {task} ===")

        for ticker in tqdm(tickers, desc=f"  Hybrid/{task}"):
            t_data = transcripts[ticker]
            parsed = t_data["parsed"]
            meta = t_data["meta"]

            # Get features for this ticker
            if ticker in features_df.index:
                feats = features_df.loc[ticker].to_dict()
            else:
                continue

            features_summary = format_features_summary(feats)
            transcript_text = truncate_transcript(parsed, config.MAX_TRANSCRIPT_TOKENS)
            sector = company_sectors.get(ticker, "Unknown")

            # ── 7a: Feature injection (features + transcript) ────────
            try:
                sys_p, usr_p = feat_inject(
                    transcript_text, features_summary,
                    meta["symbol"], meta["company_name"], meta["date"],
                    meta["quarter"], task
                )
                result = client.call(sys_p, usr_p, experiment_name="GPT-feat-inject",
                                     ticker=ticker)
                pred = parse_prediction(result["content"], task)
            except Exception as e:
                pred = None
                result = {"content": str(e), "input_tokens": 0, "output_tokens": 0}

            all_predictions.append({
                "experiment": "GPT-feat-inject", "task": task,
                "symbol": ticker, "prediction": pred,
                "raw_response": result["content"][:500],
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            })

            # ── 7b: Feature only (no transcript) ─────────────────────
            try:
                sys_p, usr_p = feat_only(
                    features_summary,
                    meta["symbol"], meta["company_name"], meta["date"],
                    meta["quarter"], task
                )
                result = client.call(sys_p, usr_p, experiment_name="GPT-feat-only",
                                     ticker=ticker)
                pred = parse_prediction(result["content"], task)
            except Exception as e:
                pred = None
                result = {"content": str(e), "input_tokens": 0, "output_tokens": 0}

            all_predictions.append({
                "experiment": "GPT-feat-only", "task": task,
                "symbol": ticker, "prediction": pred,
                "raw_response": result["content"][:500],
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            })

            # ── 7c: Speaker-segmented ────────────────────────────────
            try:
                mgmt_text = parsed.get("management_text", "")[:20000]
                qa_pairs = parsed.get("qa_pairs", [])[:5]
                sys_p, usr_p = speaker_seg(
                    mgmt_text, qa_pairs, features_summary,
                    meta["symbol"], meta["company_name"], meta["date"],
                    meta["quarter"], task
                )
                result = client.call(sys_p, usr_p, experiment_name="GPT-speaker-seg",
                                     ticker=ticker)
                pred = parse_prediction(result["content"], task)
            except Exception as e:
                pred = None
                result = {"content": str(e), "input_tokens": 0, "output_tokens": 0}

            all_predictions.append({
                "experiment": "GPT-speaker-seg", "task": task,
                "symbol": ticker, "prediction": pred,
                "raw_response": result["content"][:500],
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            })

            # ── 7d: Contrastive injection ────────────────────────────
            try:
                # Get contrastive features (L-M and statistical)
                contrastive_keys = [k for k in feats
                                    if any(k.startswith(p) for p in
                                           ["full_lm_", "mgmt_lm_", "analyst_lm_"])]
                company_feats = {k: feats[k] for k in contrastive_keys}

                # Use LOO sector averages if available (avoids self-influence)
                if sector_avgs_loo is not None and ticker in sector_avgs_loo.index:
                    sect_feats = sector_avgs_loo.loc[ticker].to_dict()
                    sect_feats = {k: sect_feats.get(k, 0.0) for k in contrastive_keys}
                elif sector in sector_avgs.index:
                    sect_feats = sector_avgs.loc[sector].to_dict()
                    sect_feats = {k: sect_feats.get(k, 0.0) for k in contrastive_keys}
                else:
                    sect_feats = {k: 0.0 for k in contrastive_keys}

                sys_p, usr_p = contrastive(
                    transcript_text, company_feats, sect_feats,
                    meta["symbol"], meta["company_name"], meta["date"],
                    meta["quarter"], sector, task
                )
                result = client.call(sys_p, usr_p, experiment_name="GPT-contrastive",
                                     ticker=ticker)
                pred = parse_prediction(result["content"], task)
            except Exception as e:
                pred = None
                result = {"content": str(e), "input_tokens": 0, "output_tokens": 0}

            all_predictions.append({
                "experiment": "GPT-contrastive", "task": task,
                "symbol": ticker, "prediction": pred,
                "raw_response": result["content"][:500],
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            })

            # ── 7e: XGBoost injection ────────────────────────────────
            try:
                if task in xgb_preds and ticker in xgb_preds[task].index:
                    xgb_pred_label = str(xgb_preds[task].loc[ticker, f"xgb_pred_{task}"])
                    xgb_conf = float(xgb_preds[task].loc[ticker, f"xgb_conf_{task}"])
                else:
                    xgb_pred_label = "UNKNOWN"
                    xgb_conf = 0.0

                shap_top = shap_data.get(task, {}).get("sample_shap", {}).get(ticker, [])

                sys_p, usr_p = xgb_inject(
                    transcript_text, xgb_pred_label, xgb_conf, shap_top,
                    meta["symbol"], meta["company_name"], meta["date"],
                    meta["quarter"], task
                )
                result = client.call(sys_p, usr_p, experiment_name="GPT-xgb-inject",
                                     ticker=ticker)
                pred = parse_prediction(result["content"], task)
            except Exception as e:
                pred = None
                result = {"content": str(e), "input_tokens": 0, "output_tokens": 0}

            all_predictions.append({
                "experiment": "GPT-xgb-inject", "task": task,
                "symbol": ticker, "prediction": pred,
                "raw_response": result["content"][:500],
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            })

    # ── Save hybrid results ──────────────────────────────────────────────
    print("\n[3/4] Saving hybrid results...")
    hybrid_df = pd.DataFrame(all_predictions)
    hybrid_df.to_csv(config.RESULTS_DIR / "hybrid_predictions.csv", index=False)

    # ── Ensemble (majority vote) ─────────────────────────────────────────
    print("\n[4/4] Computing ensemble predictions...")

    # Load GPT base predictions
    gpt_base = pd.read_csv(config.RESULTS_DIR / "gpt_predictions.csv")
    all_preds = pd.concat([gpt_base, hybrid_df], ignore_index=True)
    all_preds.to_csv(config.RESULTS_DIR / "all_predictions.csv", index=False)

    # Ensemble: XGBoost + each GPT variant, with confidence-weighted tiebreak
    # and optional FinBERT as a third voter (binary task only).
    #
    # Old logic: XGB always won on disagreement → ensemble could never
    # outperform XGB alone.  New logic uses XGB confidence to decide:
    #   - Agreement: use agreed label (both models confident)
    #   - Disagreement + XGB confident (≥0.60): trust XGB
    #   - Disagreement + XGB uncertain (<0.60): trust GPT
    # For binary task, FinBERT adds a genuine third vote when available.
    XGB_CONFIDENCE_THRESHOLD = 0.60

    # Load FinBERT predictions if available (binary only, adds 3rd voter)
    finbert_preds = {}
    finbert_path = config.RESULTS_DIR / "finbert_predictions.csv"
    if finbert_path.exists():
        fb_df = pd.read_csv(finbert_path).set_index("symbol")
        finbert_preds["binary"] = fb_df
        print(f"  FinBERT predictions loaded: {len(fb_df)} samples (binary 3rd voter)")
    else:
        print("  FinBERT predictions not found — using 2-voter ensemble only")

    ensemble_results = []
    for task in config.TASKS:
        if task not in xgb_preds:
            continue

        gpt_experiments = all_preds[all_preds["task"] == task]["experiment"].unique()

        for exp in gpt_experiments:
            exp_preds = all_preds[
                (all_preds["task"] == task) & (all_preds["experiment"] == exp)
            ].set_index("symbol")["prediction"]

            for ticker in tickers:
                xgb_label = None
                xgb_conf = 0.0
                if ticker in xgb_preds[task].index:
                    xgb_label = str(xgb_preds[task].loc[ticker, f"xgb_pred_{task}"])
                    xgb_conf = float(xgb_preds[task].loc[ticker, f"xgb_conf_{task}"])

                gpt_label = exp_preds.get(ticker)

                # Get FinBERT vote if available (binary task only)
                finbert_label = None
                if task in finbert_preds and ticker in finbert_preds[task].index:
                    finbert_label = str(
                        finbert_preds[task].loc[ticker, f"finbert_pred_{task}"]
                    )

                # ── Ensemble voting logic ─────────────────────────────
                votes = [v for v in [xgb_label, gpt_label, finbert_label]
                         if v is not None and v != "nan"]

                if len(votes) == 0:
                    ensemble_label = None
                elif len(votes) == 1:
                    ensemble_label = votes[0]
                else:
                    # Count votes
                    from collections import Counter
                    vote_counts = Counter(votes)
                    top_label, top_count = vote_counts.most_common(1)[0]

                    if top_count > len(votes) / 2:
                        # Clear majority — use it
                        ensemble_label = top_label
                    elif xgb_label and gpt_label and xgb_label != gpt_label:
                        # 2-way tie between XGB and GPT — use confidence
                        if xgb_conf >= XGB_CONFIDENCE_THRESHOLD:
                            ensemble_label = xgb_label
                        else:
                            ensemble_label = gpt_label
                    else:
                        # Fallback: use the most common vote
                        ensemble_label = top_label

                ensemble_results.append({
                    "experiment": f"Ensemble-XGB+{exp}",
                    "task": task,
                    "symbol": ticker,
                    "prediction": ensemble_label,
                    "xgb_pred": xgb_label,
                    "xgb_conf": xgb_conf,
                    "gpt_pred": gpt_label,
                    "finbert_pred": finbert_label,
                    "n_voters": len(votes),
                    "tiebreak_method": (
                        "majority" if len(votes) >= 3 and
                        Counter([v for v in votes if v]).most_common(1)[0][1] > 1
                        else f"xgb_conf_{xgb_conf:.2f}"
                        if xgb_label and gpt_label and xgb_label != gpt_label
                        else "agreement"
                    ),
                })

    ensemble_df = pd.DataFrame(ensemble_results)
    ensemble_df.to_csv(config.RESULTS_DIR / "ensemble_predictions.csv", index=False)

    # Cost summary
    cost = client.get_cost_summary()
    print(f"\n  Hybrid cost summary:")
    for k, v in cost.items():
        print(f"    {k}: {v}")

    with open(config.RESULTS_DIR / "hybrid_cost_summary.json", "w") as f:
        json.dump(cost, f, indent=2)

    print(f"\n{'=' * 70}")
    print("HYBRID EXPERIMENTS COMPLETE")
    print(f"  Hybrid predictions: {config.RESULTS_DIR / 'hybrid_predictions.csv'}")
    print(f"  Ensemble predictions: {config.RESULTS_DIR / 'ensemble_predictions.csv'}")
    print(f"  All predictions: {config.RESULTS_DIR / 'all_predictions.csv'}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 07c: RE-RUN HYBRID GPT-XGB-INJECT (clean OOF predictions)
# ═══════════════════════════════════════════════════════════════════════════════
# Previously in 07c_rerun_xgb_inject.py — now merged here to share
# parse_prediction and avoid code duplication.
#
# Run this AFTER 06_gpt_experiments.py rerun-xgb has finished.
# It re-runs ONLY the GPT-xgb-inject hybrid experiment (200 calls),
# patches hybrid_predictions.csv, rebuilds all_predictions.csv and
# ensemble_predictions.csv with the confidence-weighted voting logic.
# ═══════════════════════════════════════════════════════════════════════════════

def rerun_hybrid_xgb_inject():
    """
    Targeted re-run of GPT-xgb-inject in hybrid experiments only.
    Cost: ~200 API calls x ~5s = ~17 minutes, ~$2-3
    """
    print("=" * 70)
    print("STEP 07c: RE-RUN HYBRID GPT-XGB-INJECT (clean OOF predictions)")
    print("=" * 70)

    # ── Load corrected XGB predictions ────────────────────────────────────
    print("\n[1/5] Loading corrected XGB predictions...")
    xgb_preds = {}
    for task in config.TASKS:
        path = config.RESULTS_DIR / f"xgb_predictions_{task}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Please re-run 05_xgboost_baseline.py first."
            )
        df = pd.read_csv(path)
        xgb_preds[task] = df.set_index("symbol")
        avg_conf = df[f"xgb_conf_{task}"].mean()
        print(f"  {task}: {len(df)} predictions, avg confidence: {avg_conf:.3f}")
        if avg_conf > 0.90:
            print(f"  WARNING: avg confidence still very high ({avg_conf:.3f})")
            print(f"  Expected OOF range: 0.55 - 0.75. May still be leaky.")
            ans = input("  Continue anyway? (y/n): ").strip().lower()
            if ans != "y":
                print("  Aborted.")
                return

    # ── Load SHAP data ────────────────────────────────────────────────────
    print("\n[2/5] Loading SHAP data...")
    shap_data = joblib.load(config.RESULTS_DIR / "shap_data.joblib")
    print("  SHAP data loaded.")

    # ── Load transcripts ──────────────────────────────────────────────────
    print("\n[3/5] Loading transcripts...")
    companies = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    transcripts = {}
    for _, row in companies.iterrows():
        ticker = row["ticker"]
        path = config.RAW_TRANSCRIPTS_DIR / f"{ticker}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            parsed = parse_structured_content(data.get("structured_content"))
            if parsed is None:
                parsed = parse_raw_content(data.get("content", ""))
            if parsed:
                transcripts[ticker] = {
                    "parsed": parsed,
                    "meta": {
                        "symbol": ticker,
                        "company_name": data.get("company_name", row.get("company_name", "")),
                        "date": data.get("date", ""),
                        "quarter": f"Q{data.get('quarter', '?')} {data.get('year', '?')}",
                    }
                }

    tickers = sorted(transcripts.keys())
    client = AzureGPTClient()
    print(f"  Loaded {len(transcripts)} transcripts | Deployment: {client.deployment}")

    # ── Re-run GPT-xgb-inject hybrid ─────────────────────────────────────
    print("\n[4/5] Re-running hybrid GPT-xgb-inject with clean predictions...")
    new_predictions = []

    for task in config.TASKS:
        print(f"\n  --- Hybrid GPT-xgb-inject / {task} ---")

        for ticker in tqdm(tickers, desc=f"  hybrid-xgb/{task}"):
            t_data = transcripts[ticker]
            parsed = t_data["parsed"]
            meta = t_data["meta"]

            transcript_text = truncate_transcript(parsed, config.MAX_TRANSCRIPT_TOKENS)

            # Clean OOF XGB prediction
            if task in xgb_preds and ticker in xgb_preds[task].index:
                xgb_pred_label = str(xgb_preds[task].loc[ticker, f"xgb_pred_{task}"])
                xgb_conf = float(xgb_preds[task].loc[ticker, f"xgb_conf_{task}"])
            else:
                new_predictions.append({
                    "experiment": "GPT-xgb-inject",
                    "task": task,
                    "symbol": ticker,
                    "prediction": None,
                    "raw_response": "SKIPPED: no XGB prediction available",
                    "input_tokens": 0,
                    "output_tokens": 0,
                })
                continue

            shap_top = shap_data.get(task, {}).get("sample_shap", {}).get(ticker, [])

            try:
                sys_p, usr_p = xgb_inject(
                    transcript_text, xgb_pred_label, xgb_conf, shap_top,
                    meta["symbol"], meta["company_name"],
                    meta["date"], meta["quarter"], task
                )
                result = client.call(
                    sys_p, usr_p,
                    experiment_name="GPT-xgb-inject",
                    ticker=ticker,
                )
                response = result["content"]
                pred = parse_prediction(response, task)
            except Exception as e:
                print(f"    ERROR {ticker}: {e}")
                response = str(e)
                pred = None
                result = {"input_tokens": 0, "output_tokens": 0}

            new_predictions.append({
                "experiment": "GPT-xgb-inject",
                "task": task,
                "symbol": ticker,
                "prediction": pred,
                "raw_response": response[:500],
                "input_tokens": result.get("input_tokens", 0) if isinstance(result, dict) else 0,
                "output_tokens": result.get("output_tokens", 0) if isinstance(result, dict) else 0,
            })

        task_preds = [p for p in new_predictions if p["task"] == task]
        valid = [p for p in task_preds if p["prediction"] is not None]
        dist = pd.Series([p["prediction"] for p in valid]).value_counts().to_dict()
        print(f"    Valid: {len(valid)}/{len(task_preds)} | Distribution: {dist}")

    # ── Patch hybrid_predictions.csv ──────────────────────────────────────
    print("\n[5/5] Patching prediction files...")

    hybrid_path = config.RESULTS_DIR / "hybrid_predictions.csv"
    if hybrid_path.exists():
        existing_hybrid = pd.read_csv(hybrid_path)
        existing_hybrid_clean = existing_hybrid[
            existing_hybrid["experiment"] != "GPT-xgb-inject"
        ].copy()
        n_removed = len(existing_hybrid) - len(existing_hybrid_clean)
        print(f"  Removed {n_removed} old GPT-xgb-inject rows from hybrid_predictions.csv")
    else:
        existing_hybrid_clean = pd.DataFrame()

    new_df = pd.DataFrame(new_predictions)
    patched_hybrid = pd.concat([existing_hybrid_clean, new_df], ignore_index=True)
    patched_hybrid.to_csv(hybrid_path, index=False)
    print(f"  Saved patched hybrid_predictions.csv ({len(patched_hybrid)} rows)")

    # Rebuild all_predictions.csv from gpt + hybrid
    gpt_path = config.RESULTS_DIR / "gpt_predictions.csv"
    if gpt_path.exists():
        gpt_df = pd.read_csv(gpt_path)
        all_preds = pd.concat([gpt_df, patched_hybrid], ignore_index=True)
        all_preds.to_csv(config.RESULTS_DIR / "all_predictions.csv", index=False)
        print(f"  Rebuilt all_predictions.csv ({len(all_preds)} rows)")
    else:
        print("  WARNING: gpt_predictions.csv not found — all_predictions.csv not rebuilt")

    # Rebuild ensemble_predictions.csv with confidence-weighted voting
    all_preds_path = config.RESULTS_DIR / "all_predictions.csv"
    if all_preds_path.exists():
        all_preds = pd.read_csv(all_preds_path)
        ensemble_results = []

        XGB_CONFIDENCE_THRESHOLD = 0.60

        finbert_preds = {}
        finbert_path = config.RESULTS_DIR / "finbert_predictions.csv"
        if finbert_path.exists():
            fb_df = pd.read_csv(finbert_path).set_index("symbol")
            finbert_preds["binary"] = fb_df
            print(f"  FinBERT loaded for binary ensemble: {len(fb_df)} samples")

        for task in config.TASKS:
            if task not in xgb_preds:
                continue
            gpt_experiments = all_preds[all_preds["task"] == task]["experiment"].unique()

            for exp in gpt_experiments:
                exp_preds = all_preds[
                    (all_preds["task"] == task) & (all_preds["experiment"] == exp)
                ].set_index("symbol")["prediction"]

                for ticker in tickers:
                    xgb_label = None
                    xgb_conf_val = 0.0
                    if ticker in xgb_preds[task].index:
                        xgb_label = str(xgb_preds[task].loc[ticker, f"xgb_pred_{task}"])
                        xgb_conf_val = float(xgb_preds[task].loc[ticker, f"xgb_conf_{task}"])
                    gpt_label = exp_preds.get(ticker)

                    finbert_label = None
                    if task in finbert_preds and ticker in finbert_preds[task].index:
                        finbert_label = str(
                            finbert_preds[task].loc[ticker, f"finbert_pred_{task}"]
                        )

                    votes = [v for v in [xgb_label, gpt_label, finbert_label]
                             if v is not None and v != "nan"]

                    if len(votes) == 0:
                        ensemble_label = None
                    elif len(votes) == 1:
                        ensemble_label = votes[0]
                    else:
                        from collections import Counter
                        vote_counts = Counter(votes)
                        top_label, top_count = vote_counts.most_common(1)[0]

                        if top_count > len(votes) / 2:
                            ensemble_label = top_label
                        elif xgb_label and gpt_label and xgb_label != gpt_label:
                            if xgb_conf_val >= XGB_CONFIDENCE_THRESHOLD:
                                ensemble_label = xgb_label
                            else:
                                ensemble_label = gpt_label
                        else:
                            ensemble_label = top_label

                    ensemble_results.append({
                        "experiment": f"Ensemble-XGB+{exp}",
                        "task": task,
                        "symbol": ticker,
                        "prediction": ensemble_label,
                        "xgb_pred": xgb_label,
                        "xgb_conf": xgb_conf_val,
                        "gpt_pred": gpt_label,
                        "finbert_pred": finbert_label,
                    })

        ensemble_df = pd.DataFrame(ensemble_results)
        ensemble_df.to_csv(config.RESULTS_DIR / "ensemble_predictions.csv", index=False)
        print(f"  Rebuilt ensemble_predictions.csv ({len(ensemble_df)} rows)")

    # Cost summary
    cost = client.get_cost_summary()
    print(f"\n  Cost summary:")
    for k, v in cost.items():
        print(f"    {k}: {v}")

    print(f"\n{'=' * 70}")
    print("07c COMPLETE — Hybrid GPT-xgb-inject re-run with clean predictions")
    print("Next steps:")
    print("  1. Run 06b_prompt_ensemble.py  (re-compute prompt ensembles)")
    print("  2. Run 08_evaluation.py         (final clean evaluation)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rerun-xgb":
        # Usage: python 07_hybrid_experiments.py rerun-xgb
        rerun_hybrid_xgb_inject()
    else:
        # Default: run all hybrid experiments
        main()