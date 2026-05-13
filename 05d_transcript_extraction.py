"""
Step 5d: Two-Stage Transcript Extraction (Stage 1)

Sends each of the 100 transcripts to GPT with an extraction-only prompt.
GPT extracts structured signals (revenue, EPS, guidance, margins, tone,
analyst pushback, surprise factor) WITHOUT making a prediction.

The extractions are cached to data/features/extractions_100.json and
reused by all Stage 2 prediction experiments in 06_gpt_experiments.py.

This separates "comprehend the transcript" from "predict direction":
  - Stage 1 (this script): ~12K tokens in → ~400 tokens structured output
  - Stage 2 (in 06_gpt_experiments.py): ~400 tokens in → prediction

Cost: ~100 API calls × ~12K input tokens ≈ $3.00
Time: ~10 minutes

Run AFTER: 02_data_download.py (needs raw transcripts)
Run BEFORE: 06_gpt_experiments.py (which loads the extractions)
"""
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import config
from utils.azure_openai_client import AzureGPTClient
from utils.transcript_parser import (
    parse_structured_content, parse_raw_content, truncate_transcript,
)
from utils.prompt_templates import extraction_prompt


def main():
    print("=" * 70)
    print("STEP 5d: TWO-STAGE TRANSCRIPT EXTRACTION (Stage 1)")
    print("=" * 70)

    # ── Check for existing extractions (resume support) ──────────────────
    existing = {}
    if config.EXTRACTIONS_PATH.exists():
        with open(config.EXTRACTIONS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"\n  Found existing extractions: {len(existing)} transcripts")
        print(f"  Will skip already-extracted tickers (delete file to re-run all)")

    # ── Load transcripts ─────────────────────────────────────────────────
    print("\n[1/3] Loading transcripts...")
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
                        "company_name": data.get("company_name",
                                                  row.get("company_name", "")),
                        "date": data.get("date", ""),
                        "quarter": f"Q{data.get('quarter', '?')} "
                                   f"{data.get('year', '?')}",
                    }
                }

    print(f"  Loaded {len(transcripts)} transcripts")

    # ── Determine which tickers need extraction ──────────────────────────
    tickers_to_extract = [t for t in sorted(transcripts.keys())
                          if t not in existing]
    print(f"  Already extracted: {len(existing)}")
    print(f"  Need extraction:  {len(tickers_to_extract)}")

    if not tickers_to_extract:
        print("\n  All transcripts already extracted. Nothing to do.")
        print(f"  Extractions file: {config.EXTRACTIONS_PATH}")
        return

    # ── Initialize GPT client ────────────────────────────────────────────
    print("\n[2/3] Running Stage 1 extraction...")
    client = AzureGPTClient()
    print(f"  Deployment: {client.deployment}")

    extractions = dict(existing)  # start from existing
    failed = []

    for ticker in tqdm(tickers_to_extract, desc="  Extracting"):
        t_data = transcripts[ticker]
        parsed = t_data["parsed"]
        meta = t_data["meta"]

        # Use the full token budget for extraction — we want GPT to see
        # as much of the transcript as possible in Stage 1
        transcript_text = truncate_transcript(parsed, config.MAX_TRANSCRIPT_TOKENS)

        sys_p, usr_p = extraction_prompt(
            transcript_text,
            meta["symbol"], meta["company_name"],
            meta["date"], meta["quarter"],
        )

        try:
            result = client.call(
                sys_p, usr_p,
                temperature=0.0,
                max_tokens=config.EXTRACTION_MAX_TOKENS,
                experiment_name="two-stage-extraction",
                ticker=ticker,
            )
            extraction_text = result["content"].strip()

            # Validate: should contain at least some of the expected fields
            has_fields = sum(1 for field in ["REVENUE:", "EPS:", "GUIDANCE:",
                                              "MARGINS:", "MGMT_TONE:"]
                            if field in extraction_text.upper())
            if has_fields < 3:
                print(f"\n    WARNING: {ticker} extraction has only "
                      f"{has_fields}/5 expected fields")

            extractions[ticker] = {
                "extraction": extraction_text,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            }

        except Exception as e:
            print(f"\n    ERROR {ticker}: {e}")
            failed.append(ticker)

        # Save incrementally every 10 tickers (resume support)
        if len(extractions) % 10 == 0:
            with open(config.EXTRACTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(extractions, f, ensure_ascii=False, indent=2)

    # ── Save final results ───────────────────────────────────────────────
    print("\n[3/3] Saving extractions...")
    with open(config.EXTRACTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False, indent=2)

    # Print summary
    total_input = sum(e.get("input_tokens", 0) for e in extractions.values())
    total_output = sum(e.get("output_tokens", 0) for e in extractions.values())

    print(f"\n  Extractions saved: {config.EXTRACTIONS_PATH}")
    print(f"  Total extracted: {len(extractions)}")
    if failed:
        print(f"  Failed: {len(failed)} — {failed}")
    print(f"  Total input tokens:  {total_input:,}")
    print(f"  Total output tokens: {total_output:,}")

    # Cost estimate
    cost = client.get_cost_summary()
    print(f"\n  Cost summary:")
    for k, v in cost.items():
        print(f"    {k}: {v}")

    # Print a sample extraction
    sample_ticker = list(extractions.keys())[0]
    sample = extractions[sample_ticker]["extraction"]
    print(f"\n  Sample extraction ({sample_ticker}):")
    for line in sample.split("\n")[:10]:
        print(f"    {line}")

    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE (Stage 1 of two-stage pipeline)")
    print("Next: run 06_gpt_experiments.py — it will load these extractions")
    print("      and run GPT-two-stage / GPT-two-stage-CoT experiments")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
