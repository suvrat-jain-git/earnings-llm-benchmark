"""
Step 6: GPT Experiments (Zero-shot, Few-shot, with/without CoT)
Runs 4 base experiments × 100 transcripts × 2 tasks (binary/ternary).
"""
import json
import re
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

import config
from utils.azure_openai_client import AzureGPTClient
from utils.transcript_parser import (
    parse_structured_content, parse_raw_content, truncate_transcript,
    strip_boilerplate,
)
from utils.prompt_templates import (
    zero_shot, zero_shot_cot, few_shot, few_shot_cot, rag_few_shot,
    xgb_inject, two_stage_predict, two_stage_predict_cot,
)
from utils.prediction_utils import parse_prediction  # shared; do not redefine locally


def load_transcripts() -> dict:
    """Load all 100-company transcripts."""
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
    return transcripts


# ═══════════════════════════════════════════════════════════════════════════════
# GROUNDED CoT REASONING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

_REASONING_GENERATION_PROMPT = """\
You are a financial analyst reviewing an earnings call transcript.
The stock moved {label} after this call.

Analyse the transcript and explain WHY the market reacted this way.
You must cite specific evidence from the transcript for each step.

Follow this exact structure:

Step 1 — EARNINGS SURPRISE
Did reported figures beat, meet, or miss expectations? Quote any specific
numbers (revenue, EPS, margins) mentioned in the transcript.

Step 2 — GUIDANCE
Was forward guidance raised, maintained, or lowered? Quote the exact
language management used about the outlook.

Step 3 — TONE & LANGUAGE
Describe management sentiment (confident/cautious/defensive) with
specific examples. Describe analyst Q&A tone (constructive/sceptical/
probing) with examples of what they pushed back on.

Step 4 — OVERALL SIGNAL
Weigh the above signals. State the dominant signal and explain why
the {label} reaction makes sense (or note if it was surprising).

Be concise but specific. Every claim must reference something actually
said in the transcript."""


def generate_grounded_reasoning(
    client: "AzureGPTClient",
    transcript_excerpt: str,
    label: str,
    symbol: str,
) -> str:
    """
    Use GPT to generate transcript-grounded 4-step reasoning for a
    few-shot exemplar.  This replaces the hardcoded templated reasoning
    with analysis that actually references content in the transcript.

    Called once per exemplar (3 per task × 2 tasks = 6 total API calls).

    Args:
        client: initialised AzureGPTClient
        transcript_excerpt: the exemplar transcript text (~2000 tokens)
        label: ground truth label (UP / DOWN / FLAT)
        symbol: ticker symbol (for logging)

    Returns:
        4-step reasoning string grounded in transcript evidence.
        Falls back to the old templated reasoning if the API call fails.
    """
    system = (
        "You are a financial analyst. Respond ONLY with the structured "
        "4-step analysis requested. No preamble, no conclusion beyond Step 4."
    )
    user = (
        _REASONING_GENERATION_PROMPT.format(label=label)
        + f"\n\nCompany: {symbol}\n\n"
        + f"=== Transcript ===\n{transcript_excerpt}"
    )

    try:
        result = client.call(
            system, user,
            temperature=0.0,
            max_tokens=500,
            experiment_name="exemplar-reasoning-gen",
            ticker=symbol,
        )
        reasoning = result["content"].strip()

        # Validate that the response contains the expected structure
        if "Step 1" in reasoning and "Step 2" in reasoning:
            return reasoning

        # If structure is missing, wrap it
        print(f"    WARNING: reasoning for {symbol}/{label} missing step markers, using as-is")
        return reasoning

    except Exception as e:
        print(f"    WARNING: reasoning generation failed for {symbol}/{label}: {e}")
        print(f"    Falling back to templated reasoning.")
        return _fallback_templated_reasoning(label)


def _fallback_templated_reasoning(label: str) -> str:
    """Original templated reasoning as a fallback if GPT call fails."""
    label_lower = label.lower()
    if label == "UP":
        step1 = "Reported figures beat or met analyst expectations, with positive EPS or revenue performance noted in the prepared remarks."
        step2 = "Management raised or maintained forward guidance, using confident forward-looking language."
        step3 = "Management tone was confident and optimistic. Analyst Q&A was constructive with no major probing on weaknesses."
        step4 = f"Dominant signal is positive — beat + affirmed guidance points to {label_lower} reaction."
    elif label == "DOWN":
        step1 = "Reported figures missed or met expectations with notable weakness in margins or key segment performance."
        step2 = "Forward guidance was cut or management used heavily hedged language around the outlook."
        step3 = "Management tone was cautious and defensive. Analysts probed on cost pressures, competition, or demand softness."
        step4 = f"Dominant signal is negative — miss or guidance cut points to {label_lower} reaction."
    else:  # FLAT
        step1 = "Reported figures were broadly in line with expectations — no material beat or miss."
        step2 = "Guidance was maintained with no meaningful change to the full-year outlook."
        step3 = "Management tone was neutral and measured. Analyst Q&A was routine with no major surprises."
        step4 = f"Signals are balanced — in-line results and unchanged guidance points to a {label_lower} reaction."

    return (
        f"Step 1 — EARNINGS SURPRISE\n{step1}\n\n"
        f"Step 2 — GUIDANCE\n{step2}\n\n"
        f"Step 3 — TONE & LANGUAGE\n{step3}\n\n"
        f"Step 4 — OVERALL SIGNAL\n{step4}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR-AWARE EXEMPLAR SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_sector_exemplar_pool() -> dict:
    """
    Build a pool of candidate few-shot exemplars keyed by sector.

    Returns:
        {
            sector_name: {
                task: {
                    label_class: [
                        {"symbol", "date", "quarter", "transcript_excerpt",
                         "label", "content_for_reasoning"},
                        ...
                    ]
                }
            }
        }

    Uses the glopardo dataset (cached by HuggingFace from step 1) for
    sector labels covering the full S&P 500, not just the 100 eval set.
    Exemplars are drawn from OUTSIDE the eval set only.
    """
    labels_full = pd.read_csv(config.LABELS_DIR / "labels_full.csv")
    companies_100 = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    eval_tickers = set(companies_100["ticker"].tolist())

    # Build sector map: try glopardo dataset first (covers full S&P 500)
    sector_map = {}
    try:
        from datasets import load_dataset
        ds = load_dataset(config.HF_DATASET_SECTORS, split="train")
        df_sectors = ds.to_pandas()
        df_sectors = df_sectors.sort_values("year", ascending=False)
        ticker_sector = (
            df_sectors.groupby("ticker").first()
            .reset_index()[["ticker", "sector"]]
            .dropna(subset=["sector"])
        )
        sector_map = dict(zip(ticker_sector["ticker"], ticker_sector["sector"]))
        print(f"  Loaded sector map from glopardo: {len(sector_map)} tickers")
    except Exception as e:
        print(f"  WARNING: Could not load glopardo dataset: {e}")
        print(f"  Falling back to selected_companies.csv for sector map")

    # Fallback / supplement from selected_companies.csv
    for _, row in companies_100.iterrows():
        if row["ticker"] not in sector_map and pd.notna(row.get("sector")):
            sector_map[row["ticker"]] = row["sector"]

    if not sector_map:
        print("  WARNING: No sector data available. Sector-aware exemplars disabled.")
        return {}

    # Load full transcripts
    full_parquet = config.DATA_DIR / "transcripts_full.parquet"
    if not full_parquet.exists():
        print("  WARNING: transcripts_full.parquet not found.")
        return {}
    df_full = pd.read_parquet(full_parquet)

    # Build pool
    pool = {}  # sector -> task -> label -> [candidates]

    for task in config.TASKS:
        label_col = f"label_{task}"
        available = labels_full[
            (~labels_full["symbol"].isin(eval_tickers)) &
            (labels_full[label_col].notna())
        ].copy()

        classes = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]

        for _, row in available.iterrows():
            symbol = row["symbol"]
            sector = sector_map.get(symbol)
            if not sector:
                continue

            label = row[label_col]
            if label not in classes:
                continue

            # Find transcript text
            match = df_full[
                (df_full["symbol"] == symbol) &
                (df_full["year"] == row.get("year")) &
                (df_full["quarter"] == row.get("quarter"))
            ]
            if len(match) == 0 or not pd.notna(match.iloc[0].get("content")):
                continue

            content = str(match.iloc[0]["content"])
            if len(content) < 200:
                continue

            excerpt = content[:8000]
            if len(content) > 8000:
                excerpt += "\n[... truncated ...]"

            if sector not in pool:
                pool[sector] = {}
            if task not in pool[sector]:
                pool[sector][task] = {}
            if label not in pool[sector][task]:
                pool[sector][task][label] = []

            pool[sector][task][label].append({
                "symbol": symbol,
                "date": str(row["date"]),
                "quarter": f"Q{row.get('quarter', '?')} {row.get('year', '?')}",
                "transcript_excerpt": excerpt,
                "label": label,
            })

    # Print summary
    total_sectors = len(pool)
    total_candidates = sum(
        len(cands)
        for sector_data in pool.values()
        for task_data in sector_data.values()
        for cands in task_data.values()
    )
    print(f"  Sector exemplar pool: {total_sectors} sectors, "
          f"{total_candidates} total candidates")

    return pool


def select_sector_exemplars(pool: dict, sector: str, task: str) -> list[dict]:
    """
    Select 1 exemplar per class from the same sector as the query company.
    Falls back to cross-sector selection if same-sector doesn't cover all classes.

    Args:
        pool: output of build_sector_exemplar_pool()
        sector: GICS sector of the query company
        task: "binary" or "ternary"

    Returns:
        list of exemplar dicts with keys: symbol, date, quarter,
        transcript_excerpt, label, reasoning (templated)
    """
    classes = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]
    exemplars = []
    used_symbols = set()

    # First pass: same sector
    sector_data = pool.get(sector, {}).get(task, {})
    for cls in classes:
        candidates = sector_data.get(cls, [])
        for cand in candidates:
            if cand["symbol"] not in used_symbols:
                exemplar = dict(cand)
                exemplar["company_name"] = ""
                exemplar["reasoning"] = _fallback_templated_reasoning(cls)
                exemplars.append(exemplar)
                used_symbols.add(cand["symbol"])
                break

    # Second pass: fill missing classes from any sector
    found_classes = {e["label"] for e in exemplars}
    missing = [cls for cls in classes if cls not in found_classes]

    if missing:
        for cls in missing:
            found = False
            for other_sector, s_data in pool.items():
                for cand in s_data.get(task, {}).get(cls, []):
                    if cand["symbol"] not in used_symbols:
                        exemplar = dict(cand)
                        exemplar["company_name"] = ""
                        exemplar["reasoning"] = _fallback_templated_reasoning(cls)
                        exemplars.append(exemplar)
                        used_symbols.add(cand["symbol"])
                        found = True
                        break
                if found:
                    break

    return exemplars


def select_few_shot_exemplars(task: str,
                              client: "AzureGPTClient | None" = None) -> list[dict]:
    """
    Select 3 few-shot exemplars (1 per class) from transcripts
    NOT in the 100-company eval set.

    If a GPT client is provided, generates transcript-grounded reasoning
    for each exemplar (6 API calls total across both tasks).
    Otherwise falls back to templated reasoning.
    """
    # Load full labels
    labels_full = pd.read_csv(config.LABELS_DIR / "labels_full.csv")
    companies_100 = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    eval_tickers = set(companies_100["ticker"].tolist())

    label_col = f"label_{task}"
    available = labels_full[
        (~labels_full["symbol"].isin(eval_tickers)) &
        (labels_full[label_col].notna())
    ].copy()

    # Load full transcripts for exemplar text
    full_parquet = config.DATA_DIR / "transcripts_full.parquet"
    if full_parquet.exists():
        df_full = pd.read_parquet(full_parquet)
    else:
        return []

    classes = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]
    exemplars = []

    for cls in classes:
        candidates = available[available[label_col] == cls]
        if len(candidates) == 0:
            continue

        # Pick the first candidate that has transcript text
        for _, cand in candidates.iterrows():
            match = df_full[
                (df_full["symbol"] == cand["symbol"]) &
                (df_full["year"] == cand.get("year")) &
                (df_full["quarter"] == cand.get("quarter"))
            ]
            if len(match) > 0 and pd.notna(match.iloc[0].get("content")):
                content = str(match.iloc[0]["content"])
                # Truncate excerpt to ~2000 tokens
                excerpt = content[:8000]  # ~2000 tokens
                if len(content) > 8000:
                    excerpt += "\n[... truncated ...]"

                # Generate transcript-grounded reasoning via GPT if client
                # is available; otherwise fall back to templated reasoning.
                if client is not None:
                    print(f"    Generating grounded reasoning for "
                          f"{cand['symbol']}/{cls} exemplar...")
                    reasoning = generate_grounded_reasoning(
                        client, excerpt, cls, cand["symbol"],
                    )
                else:
                    reasoning = _fallback_templated_reasoning(cls)

                exemplars.append({
                    "symbol": cand["symbol"],
                    "company_name": "",
                    "date": str(cand["date"]),
                    "quarter": f"Q{cand.get('quarter', '?')} {cand.get('year', '?')}",
                    "transcript_excerpt": excerpt,
                    "label": cls,
                    "reasoning": reasoning,
                })
                break

    return exemplars


def build_rag_index(task: str) -> tuple:
    """
    Build a RAG retrieval index from transcripts OUTSIDE the 100-company
    eval set. Uses sentence-transformers to embed each transcript excerpt,
    then retrieves the top-K most similar ones at query time.

    Returns:
        (embeddings np.array, metadata list, model)
        where metadata[i] = {symbol, date, quarter, excerpt, label, reasoning}
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  WARNING: sentence-transformers not installed. "
              "Run: pip install sentence-transformers")
        return None, None, None

    print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load full transcript pool and labels
    labels_full = pd.read_csv(config.LABELS_DIR / "labels_full.csv")
    companies_100 = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    eval_tickers = set(companies_100["ticker"].tolist())

    label_col = f"label_{task}"
    available = labels_full[
        (~labels_full["symbol"].isin(eval_tickers)) &
        (labels_full[label_col].notna())
    ].copy()

    full_parquet = config.DATA_DIR / "transcripts_full.parquet"
    if not full_parquet.exists():
        print("  WARNING: transcripts_full.parquet not found. RAG unavailable.")
        return None, None, None

    df_full = pd.read_parquet(full_parquet)
    print(f"  Pool size for RAG: {len(available)} transcripts")

    # Build corpus: one entry per available labelled transcript
    corpus_meta = []
    corpus_texts = []

    for _, row in available.iterrows():
        match = df_full[
            (df_full["symbol"] == row["symbol"]) &
            (df_full["year"] == row.get("year")) &
            (df_full["quarter"] == row.get("quarter"))
        ]
        if len(match) == 0:
            continue
        content = str(match.iloc[0].get("content", ""))
        if not content or len(content) < 100:
            continue

        # Strip boilerplate (safe-harbor disclaimers) before embedding
        # so the embedding captures actual content, not legal boilerplate
        content = strip_boilerplate(content)

        # Use first 2000 chars as the embedding text (efficient + representative)
        excerpt = content[:8000]
        embed_text = content[:2000]

        label = row[label_col]
        label_lower = label.lower()

        # RAG corpus reasoning stays templated (not GPT-generated) because
        # the corpus contains hundreds/thousands of exemplars and generating
        # grounded reasoning for each would cost hundreds of API calls.
        # Only the static few-shot exemplars (3 per task) get GPT-grounded
        # reasoning — those are shown to GPT for EVERY prediction so the
        # investment pays off.  RAG exemplars are shown at most once and
        # are already similarity-matched, so the transcript excerpt itself
        # provides the grounding.
        reasoning = _fallback_templated_reasoning(label)

        corpus_meta.append({
            "symbol": row["symbol"],
            "company_name": "",
            "date": str(row["date"]),
            "quarter": f"Q{row.get('quarter', '?')} {row.get('year', '?')}",
            "transcript_excerpt": excerpt,
            "label": label,
            "reasoning": reasoning,
        })
        corpus_texts.append(embed_text)

    if not corpus_texts:
        print("  WARNING: RAG corpus is empty.")
        return None, None, None

    print(f"  Embedding {len(corpus_texts)} corpus transcripts...")
    embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # so dot product = cosine similarity
    )

    print(f"  RAG index built: {embeddings.shape}")
    return embeddings, corpus_meta, model


def retrieve_similar_exemplars(
    query_text: str,
    corpus_embeddings: np.ndarray,
    corpus_meta: list,
    model,
    task: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve top-K most similar exemplars for a query transcript.
    Ensures at least one UP and one DOWN exemplar if possible
    (label-balanced retrieval).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Embed query (first 2000 chars, stripped of boilerplate, same as corpus)
    query_clean = strip_boilerplate(query_text)
    query_embed = model.encode(
        [query_clean[:2000]],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Cosine similarities
    sims = cosine_similarity(query_embed, corpus_embeddings)[0]

    # Sort by similarity descending
    ranked_idx = np.argsort(sims)[::-1]

    # Label-balanced selection: try to include at least 1 per class
    valid_labels = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]
    selected = []
    seen_labels = set()

    # First pass: one per class (highest similarity per class)
    for idx in ranked_idx:
        label = corpus_meta[idx]["label"]
        if label in valid_labels and label not in seen_labels:
            entry = dict(corpus_meta[idx])
            entry["similarity_score"] = float(sims[idx])
            selected.append(entry)
            seen_labels.add(label)
        if len(seen_labels) == len(valid_labels):
            break

    # Second pass: fill remaining slots with highest similarity overall
    for idx in ranked_idx:
        if len(selected) >= top_k:
            break
        sym = corpus_meta[idx]["symbol"]
        # Avoid duplicates
        if not any(e["symbol"] == sym and e["date"] == corpus_meta[idx]["date"]
                   for e in selected):
            entry = dict(corpus_meta[idx])
            entry["similarity_score"] = float(sims[idx])
            selected.append(entry)

    # Sort final selection by similarity
    selected.sort(key=lambda x: x["similarity_score"], reverse=True)
    return selected[:top_k]


def main():
    print("=" * 70)
    print("STEP 6: GPT EXPERIMENTS (ZERO-SHOT / FEW-SHOT)")
    print("=" * 70)

    # ── Initialize GPT client ────────────────────────────────────────────
    print("\n[1/6] Initializing Azure OpenAI client...")
    client = AzureGPTClient()
    print(f"  Deployment: {client.deployment}")

    # ── Load transcripts ─────────────────────────────────────────────────
    print("\n[2/6] Loading transcripts...")
    transcripts = load_transcripts()
    print(f"  Loaded {len(transcripts)} transcripts")

    # ── Select few-shot exemplars (static) ───────────────────────────────
    print("\n[3/6] Selecting static few-shot exemplars + generating grounded reasoning...")
    exemplars = {}
    for task in config.TASKS:
        exemplars[task] = select_few_shot_exemplars(task, client=client)
        print(f"  {task}: {len(exemplars[task])} exemplars "
              f"({[e['label'] for e in exemplars[task]]})")

    # ── Build RAG index ──────────────────────────────────────────────────
    print("\n[4/6] Building RAG retrieval index...")
    rag_index = {}
    for task in config.TASKS:
        print(f"  Building index for task: {task}")
        emb, meta, emb_model = build_rag_index(task)
        rag_index[task] = {"embeddings": emb, "meta": meta, "model": emb_model}
    rag_available = all(
        rag_index[t]["embeddings"] is not None for t in config.TASKS
    )
    if rag_available:
        print("  RAG index ready.")
    else:
        print("  RAG index unavailable — GPT-rag-few will be skipped.")

    # ── Build sector-aware exemplar pool ──────────────────────────────────
    print("\n  Building sector-aware exemplar pool...")
    sector_pool = build_sector_exemplar_pool()
    # Load sector map for eval tickers
    companies_100 = pd.read_csv(config.DATA_DIR / "selected_companies.csv")
    company_sectors = companies_100.set_index("ticker")["sector"].to_dict()
    sector_available = len(sector_pool) > 0

    # ── Define base experiments ──────────────────────────────────────────
    experiments = {
        "GPT-zero": lambda tr, meta, task: zero_shot(
            tr, meta["symbol"], meta["company_name"], meta["date"], meta["quarter"], task
        ),
        "GPT-zero-CoT": lambda tr, meta, task: zero_shot_cot(
            tr, meta["symbol"], meta["company_name"], meta["date"], meta["quarter"], task
        ),
        "GPT-few": lambda tr, meta, task: few_shot(
            tr, meta["symbol"], meta["company_name"], meta["date"], meta["quarter"],
            exemplars[task], task
        ),
        "GPT-few-CoT": lambda tr, meta, task: few_shot_cot(
            tr, meta["symbol"], meta["company_name"], meta["date"], meta["quarter"],
            exemplars[task], task
        ),
    }

    # ── Run base experiments ─────────────────────────────────────────────
    print("\n[5/6] Running experiments...")
    all_predictions = []
    tickers = sorted(transcripts.keys())

    for exp_name, prompt_fn in experiments.items():
        for task in config.TASKS:
            print(f"\n  --- {exp_name} / {task} ---")

            for ticker in tqdm(tickers, desc=f"  {exp_name}/{task}"):
                t_data = transcripts[ticker]
                parsed = t_data["parsed"]
                meta = t_data["meta"]

                # Truncate transcript for GPT
                transcript_text = truncate_transcript(parsed, config.MAX_TRANSCRIPT_TOKENS)

                # Build prompt
                system_prompt, user_prompt = prompt_fn(transcript_text, meta, task)

                # Call GPT
                result = {}  # ensure result is always bound even if call fails
                try:
                    result = client.call(
                        system_prompt, user_prompt,
                        experiment_name=exp_name, ticker=ticker,
                    )
                    response = result["content"]
                    pred = parse_prediction(response, task)
                except Exception as e:
                    print(f"    ERROR {ticker}: {e}")
                    response = str(e)
                    pred = None

                all_predictions.append({
                    "experiment": exp_name,
                    "task": task,
                    "symbol": ticker,
                    "prediction": pred,
                    "raw_response": response[:500],  # Truncate for storage
                    "input_tokens": result.get("input_tokens", 0) if isinstance(result, dict) else 0,
                    "output_tokens": result.get("output_tokens", 0) if isinstance(result, dict) else 0,
                })

            # Print interim stats
            task_preds = [p for p in all_predictions
                         if p["experiment"] == exp_name and p["task"] == task]
            valid = [p for p in task_preds if p["prediction"] is not None]
            print(f"    Valid predictions: {len(valid)}/{len(task_preds)}")
            if valid:
                pred_dist = pd.Series([p["prediction"] for p in valid]).value_counts()
                print(f"    Distribution: {pred_dist.to_dict()}")

    # ── Sector-aware few-shot experiments ─────────────────────────────────
    # Per-ticker exemplars matched by GICS sector. A tech company gets tech
    # exemplars; a utility gets utility exemplars. Same prompt templates as
    # GPT-few / GPT-few-CoT, just different exemplar selection.
    if sector_available:
        print("\n  --- GPT-few-sector / GPT-few-CoT-sector ---")
        sector_experiments = {
            "GPT-few-sector": few_shot,
            "GPT-few-CoT-sector": few_shot_cot,
        }

        for exp_name, prompt_fn in sector_experiments.items():
            for task in config.TASKS:
                print(f"\n  --- {exp_name} / {task} ---")

                for ticker in tqdm(tickers, desc=f"  {exp_name}/{task}"):
                    t_data = transcripts[ticker]
                    parsed = t_data["parsed"]
                    meta = t_data["meta"]

                    transcript_text = truncate_transcript(
                        parsed, config.MAX_TRANSCRIPT_TOKENS
                    )

                    # Select exemplars from the same sector
                    sector = company_sectors.get(ticker, "Unknown")
                    sector_exs = select_sector_exemplars(
                        sector_pool, sector, task
                    )
                    if not sector_exs:
                        # Fallback to static exemplars if sector pool empty
                        sector_exs = exemplars[task]

                    try:
                        system_prompt, user_prompt = prompt_fn(
                            transcript_text,
                            meta["symbol"], meta["company_name"],
                            meta["date"], meta["quarter"],
                            sector_exs, task,
                        )
                        result = client.call(
                            system_prompt, user_prompt,
                            experiment_name=exp_name, ticker=ticker,
                        )
                        response = result["content"]
                        pred = parse_prediction(response, task)
                    except Exception as e:
                        print(f"    ERROR {ticker}: {e}")
                        response = str(e)
                        pred = None
                        result = {"input_tokens": 0, "output_tokens": 0}

                    all_predictions.append({
                        "experiment": exp_name,
                        "task": task,
                        "symbol": ticker,
                        "prediction": pred,
                        "raw_response": response[:500],
                        "input_tokens": result.get("input_tokens", 0)
                            if isinstance(result, dict) else 0,
                        "output_tokens": result.get("output_tokens", 0)
                            if isinstance(result, dict) else 0,
                    })

                task_preds = [p for p in all_predictions
                             if p["experiment"] == exp_name and p["task"] == task]
                valid = [p for p in task_preds if p["prediction"] is not None]
                print(f"    Valid predictions: {len(valid)}/{len(task_preds)}")
                if valid:
                    pred_dist = pd.Series(
                        [p["prediction"] for p in valid]
                    ).value_counts()
                    print(f"    Distribution: {pred_dist.to_dict()}")
    else:
        print("\n  Sector pool unavailable — skipping GPT-few-sector experiments.")

    # ── RAG few-shot experiment ──────────────────────────────────────────
    if rag_available:
        print("\n  --- GPT-rag-few ---")
        for task in config.TASKS:
            print(f"\n  --- GPT-rag-few / {task} ---")
            idx_data = rag_index[task]

            for ticker in tqdm(tickers, desc=f"  GPT-rag-few/{task}"):
                t_data = transcripts[ticker]
                parsed = t_data["parsed"]
                meta = t_data["meta"]

                transcript_text = truncate_transcript(
                    parsed, config.MAX_TRANSCRIPT_TOKENS
                )

                # Retrieve similar exemplars for this specific transcript
                retrieved = retrieve_similar_exemplars(
                    query_text=parsed["full_text"],
                    corpus_embeddings=idx_data["embeddings"],
                    corpus_meta=idx_data["meta"],
                    model=idx_data["model"],
                    task=task,
                    top_k=config.GPT_FEW_SHOT_N,
                )

                result = {}  # ensure result is always bound even if call fails
                try:
                    system_prompt, user_prompt = rag_few_shot(
                        transcript_text,
                        meta["symbol"], meta["company_name"],
                        meta["date"], meta["quarter"],
                        retrieved, task,
                    )
                    result = client.call(
                        system_prompt, user_prompt,
                        experiment_name="GPT-rag-few", ticker=ticker,
                    )
                    response = result["content"]
                    pred = parse_prediction(response, task)
                except Exception as e:
                    print(f"    ERROR {ticker}: {e}")
                    response = str(e)
                    pred = None
                    result = {"input_tokens": 0, "output_tokens": 0}

                all_predictions.append({
                    "experiment": "GPT-rag-few",
                    "task": task,
                    "symbol": ticker,
                    "prediction": pred,
                    "raw_response": response[:500],
                    "input_tokens": result.get("input_tokens", 0) if isinstance(result, dict) else 0,
                    "output_tokens": result.get("output_tokens", 0) if isinstance(result, dict) else 0,
                })

            rag_preds = [p for p in all_predictions
                         if p["experiment"] == "GPT-rag-few" and p["task"] == task]
            valid = [p for p in rag_preds if p["prediction"] is not None]
            print(f"    Valid predictions: {len(valid)}/{len(rag_preds)}")
            if valid:
                pred_dist = pd.Series([p["prediction"] for p in valid]).value_counts()
                print(f"    Distribution: {pred_dist.to_dict()}")

    # ── Two-stage extract → predict experiments ──────────────────────────
    print("\n  --- GPT-two-stage / GPT-two-stage-CoT ---")
    if config.EXTRACTIONS_PATH.exists():
        with open(config.EXTRACTIONS_PATH, "r", encoding="utf-8") as f:
            extractions = json.load(f)
        print(f"  Loaded {len(extractions)} pre-extracted summaries")

        two_stage_experiments = {
            "GPT-two-stage": two_stage_predict,
            "GPT-two-stage-CoT": two_stage_predict_cot,
        }

        for exp_name, prompt_fn in two_stage_experiments.items():
            for task in config.TASKS:
                print(f"\n  --- {exp_name} / {task} ---")

                for ticker in tqdm(tickers, desc=f"  {exp_name}/{task}"):
                    if ticker not in extractions:
                        all_predictions.append({
                            "experiment": exp_name, "task": task,
                            "symbol": ticker, "prediction": None,
                            "raw_response": "SKIPPED: no extraction available",
                            "input_tokens": 0, "output_tokens": 0,
                        })
                        continue

                    meta = transcripts[ticker]["meta"]
                    extraction_text = extractions[ticker]["extraction"]

                    try:
                        system_prompt, user_prompt = prompt_fn(
                            extraction_text,
                            meta["symbol"], meta["company_name"],
                            meta["date"], meta["quarter"], task,
                        )
                        result = client.call(
                            system_prompt, user_prompt,
                            experiment_name=exp_name, ticker=ticker,
                        )
                        response = result["content"]
                        pred = parse_prediction(response, task)
                    except Exception as e:
                        print(f"    ERROR {ticker}: {e}")
                        response = str(e)
                        pred = None
                        result = {"input_tokens": 0, "output_tokens": 0}

                    all_predictions.append({
                        "experiment": exp_name, "task": task,
                        "symbol": ticker, "prediction": pred,
                        "raw_response": response[:500],
                        "input_tokens": result.get("input_tokens", 0)
                            if isinstance(result, dict) else 0,
                        "output_tokens": result.get("output_tokens", 0)
                            if isinstance(result, dict) else 0,
                    })

                task_preds = [p for p in all_predictions
                             if p["experiment"] == exp_name and p["task"] == task]
                valid = [p for p in task_preds if p["prediction"] is not None]
                print(f"    Valid predictions: {len(valid)}/{len(task_preds)}")
                if valid:
                    pred_dist = pd.Series(
                        [p["prediction"] for p in valid]
                    ).value_counts()
                    print(f"    Distribution: {pred_dist.to_dict()}")
    else:
        print(f"  WARNING: {config.EXTRACTIONS_PATH} not found.")
        print(f"  Run 05d_transcript_extraction.py first to enable two-stage experiments.")
        print(f"  Skipping GPT-two-stage and GPT-two-stage-CoT.")


    # ── Self-consistency on GPT-two-stage-CoT (best value model) ───────────
    # Runs the same prompt N times with temperature>0 then takes a majority vote.
    # Applied only to GPT-two-stage-CoT: Pareto-optimal F1/cost, compact input.
    SC_RUNS        = 5
    SC_TEMPERATURE = 0.7
    SC_EXP_NAME    = "GPT-two-stage-CoT-SC"

    print(f"\n  --- {SC_EXP_NAME} ({SC_RUNS} runs, temp={SC_TEMPERATURE}) ---")
    if config.EXTRACTIONS_PATH.exists():
        with open(config.EXTRACTIONS_PATH, "r", encoding="utf-8") as _f:
            extractions_sc = json.load(_f)

        for task in config.TASKS:
            print(f"\n  --- {SC_EXP_NAME} / {task} ---")
            for ticker in tqdm(tickers, desc=f"  {SC_EXP_NAME}/{task}"):
                if ticker not in extractions_sc:
                    all_predictions.append({
                        "experiment": SC_EXP_NAME, "task": task,
                        "symbol": ticker, "prediction": None,
                        "raw_response": "SKIPPED: no extraction",
                        "input_tokens": 0, "output_tokens": 0,
                        "confidence": None,
                    })
                    continue

                meta = transcripts[ticker]["meta"]
                extraction_text = extractions_sc[ticker]["extraction"]
                run_preds, run_in_tok, run_out_tok = [], 0, 0

                for run_i in range(SC_RUNS):
                    _result = {}
                    try:
                        _sys, _usr = two_stage_predict_cot(
                            extraction_text,
                            meta["symbol"], meta["company_name"],
                            meta["date"], meta["quarter"], task,
                        )
                        _result = client.call(
                            _sys, _usr,
                            temperature=SC_TEMPERATURE,
                            experiment_name=f"{SC_EXP_NAME}-r{run_i}",
                            ticker=ticker,
                        )
                        run_preds.append(parse_prediction(_result["content"], task))
                        run_in_tok  += _result.get("input_tokens",  0)
                        run_out_tok += _result.get("output_tokens", 0)
                    except Exception as _e:
                        print(f"    ERROR {ticker} run {run_i}: {_e}")
                        run_preds.append(None)

                from collections import Counter as _Counter
                valid_rp = [p for p in run_preds if p is not None]
                if valid_rp:
                    _vote = _Counter(valid_rp)
                    majority_pred = _vote.most_common(1)[0][0]
                    agreement     = _vote.most_common(1)[0][1] / len(valid_rp)
                else:
                    majority_pred, agreement = None, 0.0

                all_predictions.append({
                    "experiment":  SC_EXP_NAME,
                    "task":        task,
                    "symbol":      ticker,
                    "prediction":  majority_pred,
                    "raw_response": f"sc_votes={run_preds} agreement={agreement:.2f}",
                    "input_tokens":  run_in_tok,
                    "output_tokens": run_out_tok,
                    "confidence":    round(agreement, 3),
                })

            _task_preds = [p for p in all_predictions
                           if p["experiment"] == SC_EXP_NAME and p["task"] == task]
            _valid = [p for p in _task_preds if p["prediction"] is not None]
            _agr   = [p["confidence"] for p in _task_preds if p.get("confidence") is not None]
            _mean_agr = sum(_agr)/len(_agr) if _agr else 0.0
            print(f"    Valid: {len(_valid)}/{len(_task_preds)} | Mean agreement: {_mean_agr:.2f}")
    else:
        print(f"  WARNING: extractions not found — skipping {SC_EXP_NAME}.")


    # ── Save results ─────────────────────────────────────────────────────
    # ── GPT-zero-calibrated: 3 runs at temp=0.3, majority vote ───────────
    print("\n  --- GPT-zero-calibrated ---")
    n_runs = getattr(config, "GPT_CALIBRATION_RUNS", 3)
    cal_temp = getattr(config, "GPT_CALIBRATION_TEMPERATURE", 0.3)
    for task in config.TASKS:
        print(f"\n  --- GPT-zero-calibrated / {task} ---")
        for ticker in tqdm(tickers, desc=f"  GPT-zero-calibrated/{task}"):
            t_data = transcripts[ticker]
            parsed = t_data["parsed"]
            meta = t_data["meta"]
            transcript_text = truncate_transcript(parsed, config.MAX_TRANSCRIPT_TOKENS)
            system_prompt, user_prompt = zero_shot(
                transcript_text, meta["symbol"], meta["company_name"],
                meta["date"], meta["quarter"], task,
            )

            run_preds = []
            run_responses = []
            run_input_tokens = 0
            run_output_tokens = 0
            for run_i in range(n_runs):
                try:
                    result = client.call(
                        system_prompt, user_prompt,
                        temperature=cal_temp,
                        experiment_name=f"GPT-zero-calibrated-r{run_i}",
                        ticker=ticker,
                    )
                    response = result["content"]
                    pred = parse_prediction(response, task)
                    run_preds.append(pred)
                    run_responses.append(response[:200])
                    run_input_tokens += result.get("input_tokens", 0)
                    run_output_tokens += result.get("output_tokens", 0)
                except Exception as e:
                    run_preds.append(None)
                    run_responses.append(str(e)[:200])

            # Majority vote (exclude None)
            valid_preds = [p for p in run_preds if p is not None]
            if valid_preds:
                from collections import Counter
                vote_counts = Counter(valid_preds)
                majority_pred = vote_counts.most_common(1)[0][0]
                agreement = vote_counts.most_common(1)[0][1] / len(valid_preds)
            else:
                majority_pred = None
                agreement = 0.0

            all_predictions.append({
                "experiment": "GPT-zero-calibrated",
                "task": task,
                "symbol": ticker,
                "prediction": majority_pred,
                "raw_response": f"votes={run_preds} agreement={agreement:.2f}",
                "input_tokens": run_input_tokens,
                "output_tokens": run_output_tokens,
                "confidence": round(agreement, 3),
            })

        cal_preds = [p for p in all_predictions
                     if p["experiment"] == "GPT-zero-calibrated" and p["task"] == task]
        valid = [p for p in cal_preds if p["prediction"] is not None]
        print(f"    Valid predictions: {len(valid)}/{len(cal_preds)}")
        if valid:
            avg_conf = np.mean([p.get("confidence", 0) for p in valid])
            print(f"    Mean agreement (confidence): {avg_conf:.3f}")
            pred_dist = pd.Series([p["prediction"] for p in valid]).value_counts()
            print(f"    Distribution: {pred_dist.to_dict()}")

    # ── Save results ─────────────────────────────────────────────────────
    print("\n[6/6] Saving results...")
    preds_df = pd.DataFrame(all_predictions)
    preds_df.to_csv(config.RESULTS_DIR / "gpt_predictions.csv", index=False)

    # Pivot for easy comparison
    for task in config.TASKS:
        task_df = preds_df[preds_df["task"] == task].pivot(
            index="symbol", columns="experiment", values="prediction"
        )
        task_df.to_csv(config.RESULTS_DIR / f"gpt_predictions_pivot_{task}.csv")

    # Cost summary
    cost = client.get_cost_summary()
    print(f"\n  Cost summary:")
    for k, v in cost.items():
        print(f"    {k}: {v}")

    with open(config.RESULTS_DIR / "gpt_cost_summary.json", "w") as f:
        json.dump(cost, f, indent=2)

    print(f"\n{'=' * 70}")
    print("GPT EXPERIMENTS COMPLETE")
    print(f"  Predictions: {config.RESULTS_DIR / 'gpt_predictions.csv'}")
    print(f"  Cost log: {config.RESULTS_DIR / 'gpt_cost_summary.json'}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 06c: RE-RUN GPT-XGB-INJECT (clean OOF predictions)
# ═══════════════════════════════════════════════════════════════════════════════
# Previously in 06c_rerun_xgb_inject.py — now merged here to share
# parse_prediction, load_transcripts, and avoid code duplication.
#
# Run this AFTER 05_xgboost_baseline.py has been fixed and re-run.
# It re-runs ONLY the GPT-xgb-inject experiment (200 calls), patches
# gpt_predictions.csv, and leaves all other experiment rows untouched.
# ═══════════════════════════════════════════════════════════════════════════════

def rerun_xgb_inject():
    """
    Targeted re-run of GPT-xgb-inject experiment only.
    Cost: ~200 API calls x ~5s = ~17 minutes, ~$2-3
    """
    print("=" * 70)
    print("STEP 06c: RE-RUN GPT-XGB-INJECT (clean OOF predictions)")
    print("=" * 70)

    # ── Verify corrected XGB predictions exist ────────────────────────────
    print("\n[1/5] Checking corrected XGB predictions...")
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
            print(f"  WARNING: avg confidence {avg_conf:.3f} is very high.")
            print(f"  This may still be in-sample predictions.")
            print(f"  Expected OOF avg confidence: typically 0.55 - 0.75")
            ans = input("  Continue anyway? (y/n): ").strip().lower()
            if ans != "y":
                print("  Aborted. Please re-run 05_xgboost_baseline.py first.")
                return

    # ── Load SHAP data ────────────────────────────────────────────────────
    print("\n[2/5] Loading SHAP data...")
    shap_path = config.RESULTS_DIR / "shap_data.joblib"
    if not shap_path.exists():
        raise FileNotFoundError(
            f"Missing: {shap_path}\n"
            f"Please re-run 05_xgboost_baseline.py first."
        )
    shap_data = joblib.load(shap_path)
    print("  SHAP data loaded.")

    # ── Load transcripts ──────────────────────────────────────────────────
    print("\n[3/5] Loading transcripts...")
    transcripts = load_transcripts()
    tickers = sorted(transcripts.keys())
    print(f"  Loaded {len(transcripts)} transcripts")

    # ── Initialize client ─────────────────────────────────────────────────
    client = AzureGPTClient()
    print(f"  Deployment: {client.deployment}")

    # ── Run GPT-xgb-inject only ───────────────────────────────────────────
    print("\n[4/5] Re-running GPT-xgb-inject with clean predictions...")
    new_predictions = []

    for task in config.TASKS:
        print(f"\n  --- GPT-xgb-inject / {task} ---")

        for ticker in tqdm(tickers, desc=f"  xgb-inject/{task}"):
            t_data = transcripts[ticker]
            parsed = t_data["parsed"]
            meta = t_data["meta"]

            transcript_text = truncate_transcript(parsed, config.MAX_TRANSCRIPT_TOKENS)

            # Get CLEAN out-of-fold XGB prediction
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

    # ── Patch gpt_predictions.csv ─────────────────────────────────────────
    print("\n[5/5] Patching gpt_predictions.csv with clean results...")

    gpt_path = config.RESULTS_DIR / "gpt_predictions.csv"
    if gpt_path.exists():
        existing = pd.read_csv(gpt_path)
        existing_clean = existing[existing["experiment"] != "GPT-xgb-inject"].copy()
        n_removed = len(existing) - len(existing_clean)
        print(f"  Removed {n_removed} old GPT-xgb-inject rows from gpt_predictions.csv")
    else:
        existing_clean = pd.DataFrame()
        print("  gpt_predictions.csv not found — creating fresh")

    new_df = pd.DataFrame(new_predictions)
    patched = pd.concat([existing_clean, new_df], ignore_index=True)
    patched.to_csv(gpt_path, index=False)
    print(f"  Saved patched gpt_predictions.csv ({len(patched)} rows total)")

    # Cost summary
    cost = client.get_cost_summary()
    print(f"\n  Cost summary:")
    for k, v in cost.items():
        print(f"    {k}: {v}")

    print(f"\n{'=' * 70}")
    print("06c COMPLETE — GPT-xgb-inject re-run with clean OOF predictions")
    print("Next steps:")
    print("  1. Run 07c_rerun_xgb_inject.py  (re-run hybrid xgb-inject)")
    print("  2. Run 06b_prompt_ensemble.py    (re-compute ensembles)")
    print("  3. Run 08_evaluation.py          (final evaluation)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rerun-xgb":
        # Usage: python 06_gpt_experiments.py rerun-xgb
        rerun_xgb_inject()
    else:
        # Default: run all GPT experiments
        main()