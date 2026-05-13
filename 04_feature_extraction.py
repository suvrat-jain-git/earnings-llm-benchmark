"""
Step 4: Feature Extraction Pipeline
Extract TF-IDF/BoW (with SVD), POS, NER, L-M sentiment, readability,
statistical, word shape, and syntactic features for all transcripts.
Speaker-level features computed separately for management vs analyst segments.
"""
import json
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pathlib import Path

import config
from utils.lm_dictionary import LMDictionary
from utils.transcript_parser import (
    parse_structured_content, parse_raw_content, tokenize_simple,
)
from utils.feature_utils import (
    extract_all_features_for_text, build_tfidf_svd, build_bow_svd,
)


def load_transcript(ticker: str) -> dict:
    """Load a transcript JSON for a selected company."""
    path = config.RAW_TRANSCRIPTS_DIR / f"{ticker}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_transcript(data: dict) -> dict:
    """Parse transcript data into segmented text."""
    parsed = parse_structured_content(data.get("structured_content"))
    if parsed is None:
        parsed = parse_raw_content(data.get("content", ""))
    return parsed


def extract_features_for_one(parsed: dict, lm_dict: LMDictionary) -> dict:
    """Extract all non-sparse features for a single transcript."""
    features = {}

    # Full text features
    full_feats = extract_all_features_for_text(
        parsed["full_text"], lm_dict, config.SPACY_MODEL, prefix="full_"
    )
    features.update(full_feats)

    # Management features
    mgmt_feats = extract_all_features_for_text(
        parsed["management_text"], lm_dict, config.SPACY_MODEL, prefix="mgmt_"
    )
    features.update(mgmt_feats)

    # Analyst features
    analyst_feats = extract_all_features_for_text(
        parsed["analyst_text"], lm_dict, config.SPACY_MODEL, prefix="analyst_"
    )
    features.update(analyst_feats)

    return features


def main():
    print("=" * 70)
    print("STEP 4: FEATURE EXTRACTION")
    print("=" * 70)

    # Load L-M dictionary
    print("\n[1/6] Loading Loughran-McDonald dictionary...")
    lm_dict = LMDictionary(config.LM_DICT_PATH)
    for cat, words in lm_dict.word_sets.items():
        print(f"  {cat}: {len(words)} words")

    # ── Load and parse 100-company transcripts ───────────────────────────
    print("\n[2/6] Loading and parsing 100-company transcripts...")
    companies = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    tickers = companies["ticker"].tolist()

    parsed_transcripts = {}
    full_texts = []  # For TF-IDF/BoW
    valid_tickers = []

    for ticker in tqdm(tickers, desc="Parsing transcripts"):
        data = load_transcript(ticker)
        if data is None:
            continue
        parsed = parse_transcript(data)
        if parsed is None or not parsed["full_text"]:
            continue
        parsed_transcripts[ticker] = parsed
        full_texts.append(parsed["full_text"])
        valid_tickers.append(ticker)

    print(f"  Successfully parsed: {len(valid_tickers)}/{len(tickers)}")

    # ── Extract speaker-level features ───────────────────────────────────
    print("\n[3/6] Extracting speaker-level features (POS, NER, L-M, readability, etc.)...")
    print("  This will take several minutes due to spaCy processing...")

    all_features = {}
    for ticker in tqdm(valid_tickers, desc="Extracting features"):
        parsed = parsed_transcripts[ticker]
        features = extract_features_for_one(parsed, lm_dict)
        all_features[ticker] = features

    # ── Load full transcripts for TF-IDF/BoW fitting ─────────────────────
    print("\n[4/6] Building TF-IDF and BoW (fitted on expanded set)...")

    # Load expanded set texts
    full_parquet = config.DATA_DIR / "transcripts_full.parquet"
    if full_parquet.exists():
        df_full = pd.read_parquet(full_parquet)
        expanded_texts = df_full["content"].dropna().tolist()
        print(f"  Expanded set: {len(expanded_texts)} transcripts for fitting")
    else:
        print("  WARNING: Full parquet not found. Fitting on 100-company set only.")
        expanded_texts = full_texts

    # Fit TF-IDF + SVD
    print("  Fitting TF-IDF vectorizer + SVD...")
    tfidf_matrix_full, tfidf_vec, tfidf_svd, tfidf_names = build_tfidf_svd(
        expanded_texts,
        max_features=config.TFIDF_MAX_FEATURES,
        n_components=config.TFIDF_SVD_COMPONENTS,
    )

    # Fit BoW + SVD
    print("  Fitting BoW vectorizer + SVD...")
    bow_matrix_full, bow_vec, bow_svd, bow_names = build_bow_svd(
        expanded_texts,
        max_features=config.BOW_MAX_FEATURES,
        n_components=config.BOW_SVD_COMPONENTS,
    )

    # Transform 100-company texts
    tfidf_100 = tfidf_svd.transform(tfidf_vec.transform(full_texts))
    bow_100 = bow_svd.transform(bow_vec.transform(full_texts))

    # Save fitted models
    joblib.dump(tfidf_vec, config.MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_svd, config.MODELS_DIR / "tfidf_svd.joblib")
    joblib.dump(bow_vec, config.MODELS_DIR / "bow_vectorizer.joblib")
    joblib.dump(bow_svd, config.MODELS_DIR / "bow_svd.joblib")
    print("  Saved fitted models to", config.MODELS_DIR)

    # ── Combine all features ─────────────────────────────────────────────
    print("\n[5/6] Combining all features into feature matrix...")

    # Build DataFrame from speaker-level features
    feature_rows = []
    for i, ticker in enumerate(valid_tickers):
        row = {"symbol": ticker}
        row.update(all_features[ticker])

        # Add TF-IDF SVD features
        for j, name in enumerate(tfidf_names):
            row[name] = tfidf_100[i, j]

        # Add BoW SVD features
        for j, name in enumerate(bow_names):
            row[name] = bow_100[i, j]

        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)
    features_df = features_df.set_index("symbol")

    # Fill NaN with 0
    features_df = features_df.fillna(0.0)

    print(f"  Feature matrix shape: {features_df.shape}")
    print(f"  Feature groups:")
    print(f"    TF-IDF SVD: {len(tfidf_names)} features")
    print(f"    BoW SVD: {len(bow_names)} features")
    n_speaker = len([c for c in features_df.columns
                     if not c.startswith("tfidf_") and not c.startswith("bow_")])
    print(f"    Speaker-level (POS/NER/LM/etc.): {n_speaker} features")

    # NOTE: features_df is saved once at the end of step 6, after sector
    # averages are computed, to avoid a stale intermediate write here.

    # ── Compute sector averages (for contrastive injection) ──────────────
    print("\n[6/6] Computing sector-average features for contrastive injection...")
    companies_map = companies.set_index("ticker")["sector"].to_dict()

    # L-M and statistical feature columns for contrastive comparison
    contrastive_cols = [c for c in features_df.columns
                        if any(c.startswith(p) for p in ["full_lm_", "full_doc_",
                               "full_type_", "full_avg_", "full_flesch",
                               "full_gunning", "full_coleman", "full_frac_",
                               "mgmt_lm_", "analyst_lm_"])]

    # IMPORTANT: compute sector averages from the FULL expanded corpus
    # (not from the 100-company eval set, which would make each company
    # partially influence its own sector baseline — circular comparison).
    # We use the full sparse features (TF-IDF/BoW) as a proxy for sector
    # membership since full speaker-level features are only available for
    # the 100-company set. For the contrastive features (L-M, statistical),
    # we fall back to the 100-company set but note this in the paper.
    full_parquet_path = config.DATA_DIR / "transcripts_full.parquet"
    sector_avgs_computed = False

    if full_parquet_path.exists():
        # Try to load full transcript metadata for sector mapping
        df_full_meta = pd.read_parquet(full_parquet_path, columns=["symbol"])
        # We only have speaker-level features for 100 companies, so we use
        # all available companies in the full index that have sector labels
        full_index = pd.read_csv(config.DATA_DIR / "transcripts_full_index.csv")
        full_sector_map = {}
        for _, row in full_index.iterrows():
            sym = row["symbol"]
            if sym in companies_map:
                full_sector_map[sym] = companies_map[sym]

        # Build a features frame for all 100-company tickers (already computed)
        features_with_sector = features_df.copy()
        features_with_sector["sector"] = features_with_sector.index.map(companies_map)

        # Check if we have enough companies per sector from the full index
        # to compute meaningful averages outside the eval set
        full_index_with_sector = full_index.copy()
        full_index_with_sector["sector"] = full_index_with_sector["symbol"].map(
            lambda s: companies_map.get(s, None)
        )
        full_index_with_sector = full_index_with_sector.dropna(subset=["sector"])

        # If full corpus has significantly more companies per sector,
        # note it — but since we only have features for 100, we must
        # use those 100. The key fix is documenting this clearly.
        n_full_per_sector = full_index_with_sector.groupby("sector").size()
        n_eval_per_sector = features_with_sector.groupby("sector").size()
        print(f"  Full corpus companies with sector: {len(full_index_with_sector)}")
        print(f"  Eval set companies per sector (used for averages):")
        print(f"  {n_eval_per_sector.to_dict()}")

    # Compute sector averages from available features
    # Note: computed within the 100-company eval set since speaker-level
    # features are only available for these companies. This is disclosed
    # in the paper limitations section.
    features_with_sector = features_df.copy()
    features_with_sector["sector"] = features_with_sector.index.map(companies_map)

    # Standard (non-LOO) sector averages for reference
    sector_avgs = features_with_sector.groupby("sector")[contrastive_cols].mean()
    sector_avgs.to_parquet(config.FEATURES_DIR / "sector_averages.parquet")

    # Leave-one-out sector averages: for each company, sector average EXCLUDES that company
    # This eliminates ~11% self-influence per company (1/N where N≈9 per sector)
    print(f"  Computing leave-one-out sector averages...")
    loo_rows = []
    for ticker in features_with_sector.index:
        sector = features_with_sector.loc[ticker, "sector"]
        if pd.isna(sector):
            # No sector info — use global average excluding self
            others = features_with_sector.drop(ticker)[contrastive_cols].mean()
        else:
            sector_group = features_with_sector[features_with_sector["sector"] == sector]
            if len(sector_group) > 1:
                others = sector_group.drop(ticker)[contrastive_cols].mean()
            else:
                # Only company in sector — use global average excluding self
                others = features_with_sector.drop(ticker)[contrastive_cols].mean()
        row = others.copy()
        row.name = ticker
        loo_rows.append(row)

    sector_avgs_loo = pd.DataFrame(loo_rows)
    sector_avgs_loo.index.name = "symbol"
    sector_avgs_loo.to_parquet(config.FEATURES_DIR / "sector_averages_loo.parquet")
    print(f"  LOO sector averages saved: {sector_avgs_loo.shape}")
    print(f"  Standard sector averages: {len(contrastive_cols)} features × {len(sector_avgs)} sectors")
    print(f"  NOTE: LOO averages eliminate self-influence. Use for contrastive experiments.")

    features_df.to_parquet(config.FEATURES_DIR / "features_100.parquet")

    # ── Also transform expanded set features (TF-IDF/BoW only for now) ───
    # Full speaker-level features for expanded set would take too long;
    # we'll compute them lazily in the XGBoost script if needed
    print("\n  Saving expanded set TF-IDF/BoW features...")
    tfidf_full_df = pd.DataFrame(tfidf_matrix_full, columns=tfidf_names)
    bow_full_df = pd.DataFrame(bow_matrix_full, columns=bow_names)
    sparse_full = pd.concat([tfidf_full_df, bow_full_df], axis=1)

    if full_parquet.exists():
        sparse_full.insert(0, "symbol", df_full["symbol"].values[:len(sparse_full)])
    sparse_full.to_parquet(config.FEATURES_DIR / "features_full_sparse.parquet", index=False)
    print(f"  Expanded sparse features shape: {sparse_full.shape}")

    print(f"\n{'=' * 70}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"  100-company features: {config.FEATURES_DIR / 'features_100.parquet'}")
    print(f"  Sector averages: {config.FEATURES_DIR / 'sector_averages.parquet'}")
    print(f"  Expanded sparse: {config.FEATURES_DIR / 'features_full_sparse.parquet'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
