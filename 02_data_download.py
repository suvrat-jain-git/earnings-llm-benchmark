"""
Step 2: Data Download
Download transcripts from HuggingFace (kurry dataset) and price data from yfinance.
- For the 100 selected companies: most recent transcript + price window
- For expanded XGBoost set: all available transcripts + prices
"""
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from datetime import timedelta

import config
from utils.price_utils import get_price_window


def main():
    print("=" * 70)
    print("STEP 2: DATA DOWNLOAD")
    print("=" * 70)

    # ── 1. Load selected companies ───────────────────────────────────────
    companies = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    selected_tickers = set(companies["ticker"].tolist())
    print(f"\nSelected companies: {len(selected_tickers)}")

    # ── 2. Load kurry dataset ────────────────────────────────────────────
    print("\n[1/4] Loading kurry/sp500_earnings_transcripts...")
    ds = load_dataset(config.HF_DATASET_TRANSCRIPTS, split="train")
    df = ds.to_pandas()
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # ── 3. Get most recent transcript per selected company ───────────────
    print("\n[2/4] Selecting most recent transcript per company...")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date_parsed", ascending=False)

    # Most recent per selected company (for LLM experiments)
    selected_transcripts = []
    for ticker in selected_tickers:
        ticker_df = df[df["symbol"] == ticker]
        if len(ticker_df) > 0:
            row = ticker_df.iloc[0]
            selected_transcripts.append(row)

    df_selected = pd.DataFrame(selected_transcripts)
    print(f"  Found transcripts for {len(df_selected)}/{len(selected_tickers)} companies")

    missing = selected_tickers - set(df_selected["symbol"].tolist())
    if missing:
        print(f"  Missing tickers: {missing}")

    # ── 4. Save transcripts (100-company set) ────────────────────────────
    print("\n[3/4] Saving transcripts and fetching prices for 100-company set...")
    failed_prices = []
    transcript_records = []

    for _, row in tqdm(df_selected.iterrows(), total=len(df_selected),
                       desc="Processing selected"):
        ticker = row["symbol"]
        date_str = str(row["date"])

        # Save transcript JSON
        transcript_data = {
            "symbol": ticker,
            "company_name": row.get("company_name", ""),
            "date": date_str,
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "quarter": int(row["quarter"]) if pd.notna(row.get("quarter")) else None,
            "content": row.get("content", ""),
            "structured_content": row.get("structured_content", None),
        }

        out_path = config.RAW_TRANSCRIPTS_DIR / f"{ticker}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, default=str)

        # Fetch prices
        try:
            prices = get_price_window(ticker, date_str, config.PRICE_WINDOW_DAYS)
            if not prices.empty:
                prices.to_csv(config.PRICES_DIR / f"{ticker}.csv")
            else:
                failed_prices.append(ticker)
        except Exception as e:
            failed_prices.append(ticker)
            print(f"  Price fetch failed for {ticker}: {e}")

        transcript_records.append({
            "symbol": ticker,
            "company_name": row.get("company_name", ""),
            "date": date_str,
            "year": row.get("year"),
            "quarter": row.get("quarter"),
            "has_structured": row.get("structured_content") is not None,
            "has_prices": ticker not in failed_prices,
        })

    # Save index
    pd.DataFrame(transcript_records).to_csv(
        config.DATA_DIR / "transcripts_100_index.csv", index=False
    )

    print(f"  Saved {len(transcript_records)} transcripts")
    if failed_prices:
        print(f"  Failed price downloads: {len(failed_prices)} — {failed_prices[:10]}")

    # ── 5. Expanded set for XGBoost ──────────────────────────────────────
    print("\n[4/4] Preparing expanded XGBoost dataset (all transcripts)...")

    # All transcripts with valid symbols
    df_full = df.dropna(subset=["symbol", "date_parsed"]).copy()
    df_full = df_full.drop_duplicates(subset=["symbol", "year", "quarter"])
    print(f"  Total unique transcripts: {len(df_full)}")

    # Save full index (we'll fetch prices during label construction)
    full_records = []
    for _, row in df_full.iterrows():
        full_records.append({
            "symbol": row["symbol"],
            "company_name": row.get("company_name", ""),
            "date": str(row["date"]),
            "year": row.get("year"),
            "quarter": row.get("quarter"),
        })

    pd.DataFrame(full_records).to_csv(
        config.DATA_DIR / "transcripts_full_index.csv", index=False
    )

    # Save full transcript texts as parquet for efficient loading
    df_full_save = df_full[["symbol", "date", "year", "quarter",
                            "content", "structured_content"]].copy()
    df_full_save.to_parquet(config.DATA_DIR / "transcripts_full.parquet", index=False)

    print(f"  Saved full index: {len(full_records)} records")
    print(f"  Saved full transcripts to parquet")

    print(f"\n{'=' * 70}")
    print(f"DATA DOWNLOAD COMPLETE")
    print(f"  100-company transcripts: {config.RAW_TRANSCRIPTS_DIR}")
    print(f"  Price data: {config.PRICES_DIR}")
    print(f"  Full index: {config.DATA_DIR / 'transcripts_full_index.csv'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
