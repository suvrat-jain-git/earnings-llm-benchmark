"""
Step 3: Label Construction
Assign UP/DOWN/FLAT labels based on earnings call timing and stock price movement.
Handles pre-market, during-market, and after-close call timing.
Includes market-adjusted (excess) returns, multi-day horizons, and sensitivity analysis.
"""
import sys
import io
# Ensure UTF-8 output on Windows terminals (fixes cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import config
from utils.price_utils import (
    classify_call_timing, get_pre_post_prices,
    compute_return, assign_label_ternary, assign_label_binary,
    get_price_window, fetch_spy_prices, compute_excess_return,
    assign_label_ternary_excess, assign_label_binary_excess,
    compute_multi_day_returns,
)


def process_transcript_label(ticker: str, date_str: str,
                             prices_dir: Path,
                             spy_prices: pd.DataFrame = None) -> dict:
    """Compute label for a single transcript, including excess returns and multi-day horizons."""
    # Load prices
    price_path = prices_dir / f"{ticker}.csv"
    base = {"pre_price": None, "post_price": None, "return": np.nan,
            "call_timing": "unknown", "label_ternary": None, "label_binary": None,
            "market_return": np.nan, "excess_return": np.nan,
            "label_ternary_excess": None, "label_binary_excess": None}

    # Add multi-day columns
    for h in config.RETURN_HORIZONS:
        base[f"return_{h}d"] = np.nan
        base[f"label_binary_{h}d"] = None
        base[f"label_ternary_{h}d"] = None

    if not price_path.exists():
        return base

    prices_df = pd.read_csv(price_path, index_col=0, parse_dates=True)

    # Classify call timing
    timing = classify_call_timing(date_str)

    # Get pre/post prices
    pre_price, post_price = get_pre_post_prices(prices_df, date_str, timing)

    # Compute raw return
    ret = compute_return(pre_price, post_price) if pre_price and post_price else np.nan

    result = {
        "call_timing": timing,
        "pre_price": pre_price,
        "post_price": post_price,
        "return": ret,
        "label_ternary": assign_label_ternary(ret, config.LABEL_THRESHOLD),
        "label_binary": assign_label_binary(ret, config.LABEL_THRESHOLD),
    }

    # Market-adjusted (excess) returns
    mkt_ret = np.nan
    excess_ret = np.nan
    if spy_prices is not None and not np.isnan(ret):
        spy_pre, spy_post = get_pre_post_prices(spy_prices, date_str, timing)
        mkt_ret = compute_return(spy_pre, spy_post) if spy_pre and spy_post else np.nan
        excess_ret = compute_excess_return(ret, mkt_ret)

    result["market_return"] = mkt_ret
    result["excess_return"] = excess_ret
    result["label_ternary_excess"] = assign_label_ternary_excess(excess_ret, config.LABEL_THRESHOLD)
    result["label_binary_excess"] = assign_label_binary_excess(excess_ret, config.LABEL_THRESHOLD)

    # Multi-day returns
    if pre_price is not None:
        multi_returns = compute_multi_day_returns(prices_df, date_str, timing, pre_price)
        for h, h_ret in multi_returns.items():
            result[f"return_{h}d"] = h_ret
            result[f"label_binary_{h}d"] = assign_label_binary(h_ret, config.LABEL_THRESHOLD)
            result[f"label_ternary_{h}d"] = assign_label_ternary(h_ret, config.LABEL_THRESHOLD)

    return result


def sensitivity_analysis(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute label distributions across multiple thresholds for robustness analysis.
    Saves results and prints distribution tables.
    """
    print("\n[SENSITIVITY] Label threshold sensitivity analysis...")
    rows = []
    for thresh in config.SENSITIVITY_THRESHOLDS:
        for _, row in labels_df.iterrows():
            ret = row["return"]
            rows.append({
                "threshold": thresh,
                "symbol": row["symbol"],
                "return": ret,
                "label_ternary": assign_label_ternary(ret, thresh),
                "label_binary": assign_label_binary(ret, thresh),
            })

    sens_df = pd.DataFrame(rows)
    out_path = config.LABELS_DIR / "labels_100_sensitivity.csv"
    sens_df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    # Print distribution per threshold
    for thresh in config.SENSITIVITY_THRESHOLDS:
        subset = sens_df[sens_df["threshold"] == thresh]
        n_up = (subset["label_ternary"] == "UP").sum()
        n_down = (subset["label_ternary"] == "DOWN").sum()
        n_flat = (subset["label_ternary"] == "FLAT").sum()
        n_binary = subset["label_binary"].notna().sum()
        print(f"  Threshold +/-{thresh:.4f}: UP={n_up} DOWN={n_down} FLAT={n_flat} | Binary N={n_binary}")

    return sens_df


def save_reproducibility_artifacts(labels_100: pd.DataFrame, index_100: pd.DataFrame):
    """Save frozen artifacts for exact reproducibility."""
    repro_dir = config.DATA_DIR / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    # Frozen company list
    index_100.to_csv(repro_dir / "selected_companies_frozen.csv", index=False)

    # Frozen labels
    labels_100.to_csv(repro_dir / "labels_100_frozen.csv", index=False)

    # Config snapshot
    config_snapshot = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(config).items()
        if not k.startswith("_") and k.isupper()
    }
    with open(repro_dir / "config_snapshot.json", "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    # Print SHA-256 hashes
    print("\n  Reproducibility artifact hashes (SHA-256):")
    for fname in ["selected_companies_frozen.csv", "labels_100_frozen.csv", "config_snapshot.json"]:
        fpath = repro_dir / fname
        h = hashlib.sha256(fpath.read_bytes()).hexdigest()[:16]
        print(f"    {fname}: {h}...")


def main():
    print("=" * 70)
    print("STEP 3: LABEL CONSTRUCTION")
    print("=" * 70)

    # ── 0. Fetch SPY prices for market adjustment ────────────────────────
    print("\n[0/5] Fetching SPY prices for market-adjusted returns...")
    # Derive date range dynamically from the 100-company transcripts so SPY
    # prices always cover every call, regardless of when the dataset was built.
    _index_for_dates = pd.read_csv(config.DATA_DIR / "transcripts_100_index.csv")
    _dates = pd.to_datetime(_index_for_dates["date"], errors="coerce").dropna()
    if len(_dates) > 0:
        _spy_start = (_dates.min() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        _spy_end   = (_dates.max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    else:
        _spy_start, _spy_end = "2024-01-01", "2026-01-01"
    print(f"  SPY fetch range: {_spy_start} -> {_spy_end}")
    spy_prices = fetch_spy_prices(_spy_start, _spy_end)
    if spy_prices.empty:
        print("  WARNING: Could not fetch SPY prices. Excess returns will be NaN.")
        spy_prices = None
    else:
        print(f"  SPY prices: {len(spy_prices)} trading days loaded.")

    # ── 1. Labels for 100-company set ────────────────────────────────────
    print("\n[1/5] Computing labels for 100-company set...")
    index_100 = pd.read_csv(config.DATA_DIR / "transcripts_100_index.csv")

    results = []
    for _, row in tqdm(index_100.iterrows(), total=len(index_100),
                       desc="Labeling 100-company"):
        ticker = row["symbol"]
        date_str = str(row["date"])

        label_info = process_transcript_label(ticker, date_str, config.PRICES_DIR,
                                              spy_prices=spy_prices)
        results.append({
            "symbol": ticker,
            "company_name": row.get("company_name", ""),
            "date": date_str,
            "year": row.get("year"),
            "quarter": row.get("quarter"),
            **label_info
        })

    labels_100 = pd.DataFrame(results)
    labels_100.to_csv(config.LABELS_DIR / "labels_100.csv", index=False)

    # Print distribution
    print(f"\n  100-company set label distribution:")
    print(f"\n  Ternary:")
    print(labels_100["label_ternary"].value_counts().to_string())
    print(f"  Null: {labels_100['label_ternary'].isna().sum()}")
    print(f"\n  Binary:")
    print(labels_100["label_binary"].value_counts().to_string())
    print(f"  Null/FLAT excluded: {labels_100['label_binary'].isna().sum()}")
    print(f"\n  Call timing distribution:")
    print(labels_100["call_timing"].value_counts().to_string())

    # Market-adjusted distribution
    if spy_prices is not None:
        print(f"\n  Market-adjusted (excess return) labels:")
        print(f"  Ternary (excess):")
        print(labels_100["label_ternary_excess"].value_counts().to_string())
        print(f"  Binary (excess):")
        print(labels_100["label_binary_excess"].value_counts().to_string())

    # Multi-day return distributions
    for h in config.RETURN_HORIZONS:
        col = f"label_binary_{h}d"
        if col in labels_100.columns:
            n_valid = labels_100[col].notna().sum()
            print(f"\n  {h}-day binary labels: N={n_valid}")
            print(labels_100[col].value_counts().to_string())

    # ── 2. Sensitivity analysis ──────────────────────────────────────────
    print("\n[2/5] Label threshold sensitivity analysis...")
    sensitivity_analysis(labels_100)

    # ── 3. Labels for expanded set (all transcripts) ─────────────────────
    print("\n[3/5] Computing labels for expanded XGBoost set...")
    print("  (This requires fetching prices for all transcripts - may take a while)")

    full_index_path = config.DATA_DIR / "transcripts_full_index.csv"
    if not full_index_path.exists():
        print("  WARNING: transcripts_full_index.csv not found. Skipping full set.")
        labels_full = pd.DataFrame()
    else:
        full_index = pd.read_csv(full_index_path)
        print(f"  Total transcripts in full set: {len(full_index)}")

        # Fetch prices and compute labels in batches
        full_results = []
        failed = 0
        for _, row in tqdm(full_index.iterrows(), total=len(full_index),
                           desc="Labeling full set"):
            ticker = str(row["symbol"])
            date_str = str(row["date"])

            # Check if we already have prices for this ticker (from 100-company set)
            price_path = config.PRICES_DIR / f"{ticker}.csv"

            # For full set, we may need to fetch prices for dates not covered
            # Try using existing price file first, then fetch if needed
            if price_path.exists():
                prices_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
                call_dt = pd.Timestamp(date_str.split()[0])

                # Check if call date is within our price data range
                if (prices_df.index.min() <= call_dt <= prices_df.index.max() or
                        abs((prices_df.index.min() - call_dt).days) < 15):
                    timing = classify_call_timing(date_str)
                    pre_price, post_price = get_pre_post_prices(prices_df, date_str, timing)
                    ret = compute_return(pre_price, post_price) if pre_price and post_price else np.nan
                else:
                    # Need fresh data for this date
                    try:
                        prices_df = get_price_window(ticker, date_str, config.PRICE_WINDOW_DAYS)
                        if not prices_df.empty:
                            timing = classify_call_timing(date_str)
                            pre_price, post_price = get_pre_post_prices(prices_df, date_str, timing)
                            ret = compute_return(pre_price, post_price) if pre_price and post_price else np.nan
                        else:
                            timing, pre_price, post_price, ret = "unknown", None, None, np.nan
                            failed += 1
                    except Exception:
                        timing, pre_price, post_price, ret = "unknown", None, None, np.nan
                        failed += 1
            else:
                try:
                    prices_df = get_price_window(ticker, date_str, config.PRICE_WINDOW_DAYS)
                    if not prices_df.empty:
                        # Cache it
                        prices_df.to_csv(price_path)
                        timing = classify_call_timing(date_str)
                        pre_price, post_price = get_pre_post_prices(prices_df, date_str, timing)
                        ret = compute_return(pre_price, post_price) if pre_price and post_price else np.nan
                    else:
                        timing, pre_price, post_price, ret = "unknown", None, None, np.nan
                        failed += 1
                except Exception:
                    timing, pre_price, post_price, ret = "unknown", None, None, np.nan
                    failed += 1

            full_results.append({
                "symbol": ticker,
                "date": date_str,
                "year": row.get("year"),
                "quarter": row.get("quarter"),
                "call_timing": timing,
                "pre_price": pre_price,
                "post_price": post_price,
                "return": ret,
                "label_ternary": assign_label_ternary(ret, config.LABEL_THRESHOLD),
                "label_binary": assign_label_binary(ret, config.LABEL_THRESHOLD),
            })

        labels_full = pd.DataFrame(full_results)
        labels_full.to_csv(config.LABELS_DIR / "labels_full.csv", index=False)

        print(f"\n  Full set label distribution:")
        print(f"  Total: {len(labels_full)}, Failed price fetch: {failed}")
        print(f"\n  Ternary:")
        print(labels_full["label_ternary"].value_counts().to_string())
        print(f"  Null: {labels_full['label_ternary'].isna().sum()}")
        print(f"\n  Binary:")
        print(labels_full["label_binary"].value_counts().to_string())
        print(f"  Null/FLAT excluded: {labels_full['label_binary'].isna().sum()}")

    # ── 4. Summary statistics ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("LABEL CONSTRUCTION COMPLETE")
    print(f"  100-company labels: {config.LABELS_DIR / 'labels_100.csv'}")
    if not labels_full.empty:
        print(f"  Full labels: {config.LABELS_DIR / 'labels_full.csv'}")

    # Return distribution
    valid_returns = labels_100["return"].dropna()
    if len(valid_returns) > 0:
        print(f"\n  100-company return statistics:")
        print(f"    Mean: {valid_returns.mean():.4f}")
        print(f"    Std:  {valid_returns.std():.4f}")
        print(f"    Min:  {valid_returns.min():.4f}")
        print(f"    Max:  {valid_returns.max():.4f}")

    # ── 5. Reproducibility artifacts ─────────────────────────────────────
    print("\n[5/5] Saving reproducibility artifacts...")
    save_reproducibility_artifacts(labels_100, index_100)

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

