"""
Step 1: Company Selection
Select 100 S&P 500 companies with sector diversity using stratified sampling.
Uses glopardo dataset for GICS sector labels, ranks by market cap.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datasets import load_dataset
from tqdm import tqdm

import config


def main():
    print("=" * 70)
    print("STEP 1: COMPANY SELECTION")
    print("=" * 70)

    # ── 1. Load glopardo dataset for sector labels ───────────────────────
    print("\n[1/4] Loading glopardo/sp500-earnings-transcripts for sector labels...")
    ds = load_dataset(config.HF_DATASET_SECTORS, split="train")
    df_sectors = ds.to_pandas()

    print(f"  Total records: {len(df_sectors)}")
    print(f"  Columns: {list(df_sectors.columns)}")

    # Get unique ticker-sector mapping (use most recent record per ticker)
    df_sectors = df_sectors.sort_values("year", ascending=False)
    ticker_sector = (
        df_sectors.groupby("ticker")
        .first()
        .reset_index()[["ticker", "sector"]]
        .dropna(subset=["sector"])
    )
    print(f"  Unique tickers with sector: {len(ticker_sector)}")
    print(f"  Sectors found: {ticker_sector['sector'].nunique()}")
    print(f"\n  Sector distribution:")
    print(ticker_sector["sector"].value_counts().to_string())

    # ── 2. Also check which tickers are in kurry dataset ─────────────────
    print("\n[2/4] Loading kurry/sp500_earnings_transcripts to find available tickers...")
    ds_kurry = load_dataset(config.HF_DATASET_TRANSCRIPTS, split="train")
    df_kurry = ds_kurry.to_pandas()

    kurry_tickers = set(df_kurry["symbol"].dropna().unique())
    print(f"  Tickers in kurry dataset: {len(kurry_tickers)}")

    # Intersection: tickers present in both datasets
    ticker_sector = ticker_sector[ticker_sector["ticker"].isin(kurry_tickers)].copy()
    print(f"  Tickers in BOTH datasets: {len(ticker_sector)}")

    _DUPES = {"GOOG", "BRK.A", "DISCK", "FOXA", "NWSA", "LBRDA", "LBRDK"}
    before_tickers = set(ticker_sector["ticker"].tolist())
    ticker_sector = ticker_sector[~ticker_sector["ticker"].isin(_DUPES)].copy()
    actually_removed = _DUPES & before_tickers
    if actually_removed:
        print(f"  Removed {len(actually_removed)} duplicate share-class tickers: {actually_removed}")

    # ── 3. Get market cap for ranking ────────────────────────────────────
    print("\n[3/4] Fetching market caps via yfinance (this may take a few minutes)...")
    market_caps = {}
    failed = []
    for ticker in tqdm(ticker_sector["ticker"].tolist(), desc="Fetching market caps"):
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap", None)
            if mc and mc > 0:
                market_caps[ticker] = mc
            else:
                failed.append(ticker)
        except Exception:
            failed.append(ticker)

    print(f"  Successfully fetched: {len(market_caps)}")
    if failed:
        print(f"  Failed/missing: {len(failed)} tickers")

    ticker_sector["market_cap"] = ticker_sector["ticker"].map(market_caps)
    ticker_sector = ticker_sector.dropna(subset=["market_cap"])
    ticker_sector = ticker_sector.sort_values("market_cap", ascending=False)

    print(f"  Tickers with valid market cap: {len(ticker_sector)}")

    # ── 4. Stratified sampling ───────────────────────────────────────────
    print(f"\n[4/4] Stratified sampling of {config.N_COMPANIES_LLM} companies...")

    # Calculate sector weights (proportion of tickers in each sector)
    sector_counts = ticker_sector["sector"].value_counts()
    total = sector_counts.sum()
    sector_weights = sector_counts / total

    # Allocate slots proportional to weight, minimum 1 per sector
    n_target = config.N_COMPANIES_LLM
    sector_slots = {}
    for sector in sector_weights.index:
        sector_slots[sector] = max(1, int(round(sector_weights[sector] * n_target)))

    # Adjust to hit exactly n_target
    while sum(sector_slots.values()) > n_target:
        # Remove from largest allocation
        largest = max(sector_slots, key=sector_slots.get)
        sector_slots[largest] -= 1
    while sum(sector_slots.values()) < n_target:
        # Add to smallest allocation
        smallest = min(sector_slots, key=sector_slots.get)
        sector_slots[smallest] += 1

    print(f"\n  Target allocation per sector:")
    for sector, n in sorted(sector_slots.items(), key=lambda x: -x[1]):
        available = len(ticker_sector[ticker_sector["sector"] == sector])
        print(f"    {sector}: {n} (available: {available})")

    # Select top N by market cap per sector
    selected = []
    for sector, n in sector_slots.items():
        sector_df = ticker_sector[ticker_sector["sector"] == sector].head(n)
        selected.append(sector_df)

    selected_df = pd.concat(selected, ignore_index=True)
    selected_df = selected_df.sort_values("market_cap", ascending=False)

    # ── Get company names from kurry dataset ─────────────────────────────
    name_map = df_kurry.groupby("symbol")["company_name"].first().to_dict()
    selected_df["company_name"] = selected_df["ticker"].map(name_map)

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = config.DATA_DIR / "selected_companies.csv"
    selected_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"RESULT: Selected {len(selected_df)} companies")
    print(f"Saved to: {output_path}")
    print(f"{'=' * 70}")
    print(f"\nSector distribution in selection:")
    print(selected_df["sector"].value_counts().to_string())
    print(f"\nTop 10 by market cap:")
    print(selected_df[["ticker", "company_name", "sector", "market_cap"]].head(10).to_string())


if __name__ == "__main__":
    main()
