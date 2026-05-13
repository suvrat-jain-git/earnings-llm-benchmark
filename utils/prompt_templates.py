"""
All GPT prompt templates for the experiments.

Improvements over original:
1. Richer, more grounding system prompts with explicit label definitions
   and decision criteria the model should anchor to.
2. Structured 4-step CoT that forces the model to reason through specific
   financial signals before committing to a label — reduces hallucination
   and improves calibration.
3. Strict output format with XML-style tags so parse_prediction is robust
   even when the model produces verbose CoT output. The old format used
   plain "PREDICTION:" which was sometimes buried or repeated in the text.
4. Better feature summary formatting: raw counts alongside ratios so the
   model can reason about magnitude, not just normalised fractions.
5. Contrastive prompt now computes and labels the direction of deviation
   (ABOVE/BELOW sector average) inline so GPT doesn't have to do the
   arithmetic itself.
6. XGBoost injection prompt explains what SHAP values mean before showing
   them, preventing the model from misinterpreting the sign convention.
7. Few-shot exemplar reasoning is now a structured 4-step template that
   matches the CoT format, making in-context learning more effective.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_BINARY = """\
You are a quantitative financial analyst specialising in post-earnings stock \
price reactions. Your task is to predict whether a company's stock will close \
HIGHER (UP) or LOWER (DOWN) on the first full trading day after its earnings call, \
relative to the closing price on the trading day immediately before the call.

Label definitions:
  UP   — stock price rises more than 0.5 % after the call
  DOWN — stock price falls more than 0.5 % after the call

Key signals to consider, in order of importance:
  1. Earnings surprise — did reported EPS / revenue beat or miss analyst consensus?
  2. Forward guidance — was outlook raised, maintained, or cut?
  3. Management tone — use of forward-looking positive language vs hedging / risk words
  4. Analyst Q&A — are analysts probing on weaknesses, or affirming confidence?
  5. Structural signals — gross margin trajectory, cost control commentary, \
capex guidance

Rules:
  - Respond ONLY with the structured format shown in the user message.
  - Do not add any text outside the specified tags.
  - If signals are genuinely mixed with no dominant direction, lean toward the \
stronger fundamental signal (guidance and earnings surprise outweigh tone alone).
"""

SYSTEM_TERNARY = """\
You are a quantitative financial analyst specialising in post-earnings stock \
price reactions. Your task is to predict whether a company's stock will close \
HIGHER (UP), LOWER (DOWN), or roughly UNCHANGED (FLAT) on the first full trading \
day after its earnings call, relative to the closing price on the trading day \
immediately before the call.

Label definitions:
  UP   — stock price rises more than 0.5 % after the call
  DOWN — stock price falls more than 0.5 % after the call
  FLAT — stock price moves within ±0.5 % (in-line quarter, no material surprise)

Key signals to consider, in order of importance:
  1. Earnings surprise — did reported EPS / revenue beat or miss analyst consensus?
  2. Forward guidance — was outlook raised, maintained, or cut?
  3. Management tone — use of forward-looking positive language vs hedging / risk words
  4. Analyst Q&A — are analysts probing on weaknesses, or affirming confidence?
  5. Structural signals — gross margin trajectory, cost control commentary, \
capex guidance

FLAT is appropriate when:
  - Results matched expectations with no guidance change
  - Positive and negative signals are roughly balanced
  - Management commentary is neutral / in-line

Rules:
  - Respond ONLY with the structured format shown in the user message.
  - Do not add any text outside the specified tags.
"""





def _label_options(task: str) -> str:
    if task == "binary":
        return "UP or DOWN"
    return "UP, DOWN, or FLAT"


def _label_pipe(task: str) -> str:
    if task == "binary":
        return "UP|DOWN"
    return "UP|DOWN|FLAT"


def _system(task: str) -> str:
    return SYSTEM_BINARY if task == "binary" else SYSTEM_TERNARY


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CoT INSTRUCTION BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

_COT_INSTRUCTION = """\
Reason through the following four steps before predicting. Be concise in each step.

Step 1 — EARNINGS SURPRISE
Did reported figures (revenue, EPS, margins) beat, meet, or miss expectations? \
Quote any specific numbers mentioned.

Step 2 — GUIDANCE
Was forward guidance raised, maintained, lowered, or absent? \
Note the exact language used (e.g. "raised full-year EPS guidance to X").

Step 3 — TONE & LANGUAGE
Summarise management sentiment (confident / cautious / defensive) and \
analyst Q&A tone (constructive / sceptical / probing on risk).

Step 4 — OVERALL SIGNAL
Weigh the above. State the dominant signal and any material counter-signals.

Then output your prediction using EXACTLY this format — no other text:

<reasoning>
[Your four-step analysis here]
</reasoning>
<prediction>[{label_pipe}]</prediction>"""

_DIRECT_INSTRUCTION = """\
Before predicting, briefly assess three key signals from the transcript:
(a) Earnings surprise — did reported revenue/EPS beat, meet, or miss expectations?
(b) Forward guidance — was the outlook raised, maintained, or lowered?
(c) Dominant signal — considering tone and analyst reaction, which direction dominates?

Output using EXACTLY this format:

<reasoning>
(a) Earnings surprise: [beat/miss/inline — cite any numbers mentioned]
(b) Guidance: [raised/maintained/lowered/not mentioned]
(c) Dominant signal: [state direction and why]
</reasoning>
<prediction>[{label_pipe}]</prediction>"""


# ═══════════════════════════════════════════════════════════════════════════════
# ZERO-SHOT
# ═══════════════════════════════════════════════════════════════════════════════

def zero_shot(transcript: str, symbol: str, company_name: str,
              date: str, quarter: str, task: str = "ternary") -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _DIRECT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# ZERO-SHOT + CHAIN-OF-THOUGHT
# ═══════════════════════════════════════════════════════════════════════════════

def zero_shot_cot(transcript: str, symbol: str, company_name: str,
                  date: str, quarter: str, task: str = "ternary") -> tuple[str, str]:
    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT
# ═══════════════════════════════════════════════════════════════════════════════

def few_shot(transcript: str, symbol: str, company_name: str,
             date: str, quarter: str, exemplars: list[dict],
             task: str = "ternary") -> tuple[str, str]:
    examples_text = ""
    for i, ex in enumerate(exemplars, 1):
        examples_text += (
            f"=== Example {i} ===\n"
            f"Company: {ex['symbol']} ({ex.get('company_name', '')})\n"
            f"Date: {ex['date']} | Period: {ex['quarter']}\n\n"
            f"Transcript excerpt:\n{ex['transcript_excerpt']}\n\n"
            f"<reasoning>\n[Analysis omitted — label provided directly]\n</reasoning>\n"
            f"<prediction>{ex['label']}</prediction>\n\n"
        )

    user = (
        f"Below are {len(exemplars)} labelled examples showing the expected output "
        f"format. Study them, then predict for the new transcript.\n\n"
        f"{examples_text}"
        f"=== Now predict ===\n"
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _DIRECT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT + CHAIN-OF-THOUGHT
# ═══════════════════════════════════════════════════════════════════════════════

def few_shot_cot(transcript: str, symbol: str, company_name: str,
                 date: str, quarter: str, exemplars: list[dict],
                 task: str = "ternary") -> tuple[str, str]:
    examples_text = ""
    for i, ex in enumerate(exemplars, 1):
        examples_text += (
            f"=== Example {i} ===\n"
            f"Company: {ex['symbol']} ({ex.get('company_name', '')})\n"
            f"Date: {ex['date']} | Period: {ex['quarter']}\n\n"
            f"Transcript excerpt:\n{ex['transcript_excerpt']}\n\n"
            f"<reasoning>\n{ex['reasoning']}\n</reasoning>\n"
            f"<prediction>{ex['label']}</prediction>\n\n"
        )

    user = (
        f"Below are {len(exemplars)} labelled examples with structured reasoning. "
        f"Follow the same four-step reasoning format.\n\n"
        f"{examples_text}"
        f"=== Now predict ===\n"
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE INJECTION (7a): structured features + transcript
# ═══════════════════════════════════════════════════════════════════════════════

def feat_inject(transcript: str, features_summary: str, symbol: str,
                company_name: str, date: str, quarter: str,
                task: str = "ternary") -> tuple[str, str]:
    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Pre-Computed NLP Features ===\n"
        f"The following signals were extracted automatically from the transcript "
        f"using financial NLP tools. Use them alongside your reading of the text.\n\n"
        f"{features_summary}\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ONLY (7b): features without transcript
# ═══════════════════════════════════════════════════════════════════════════════

def feat_only(features_summary: str, symbol: str, company_name: str,
              date: str, quarter: str, task: str = "ternary") -> tuple[str, str]:
    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"The raw transcript is NOT available. Reason solely from the quantitative "
        f"NLP features below, which were extracted from the earnings call transcript.\n\n"
        f"=== Pre-Computed NLP Features ===\n{features_summary}\n\n"
        f"Interpret these quantitative NLP features in the context of the earnings call "
        f"to predict the stock price reaction. Consider how sentiment, tone divergence, "
        f"and language patterns relate to likely market response.\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# SPEAKER-SEGMENTED (7c): management remarks + Q&A + features
# ═══════════════════════════════════════════════════════════════════════════════

def speaker_seg(mgmt_text: str, qa_pairs: list[dict], features_summary: str,
                symbol: str, company_name: str, date: str, quarter: str,
                task: str = "ternary") -> tuple[str, str]:
    qa_text = ""
    for i, qa in enumerate(qa_pairs[:5], 1):
        qa_text += (
            f"\n[Q{i}] Analyst ({qa['analyst_speaker']}):\n{qa['question']}\n\n"
            f"[A{i}] Management ({qa['mgmt_speaker']}):\n{qa['answer']}\n"
        )

    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"The transcript is structured below by speaker role. "
        f"Management prepared remarks typically contain guidance and spin; "
        f"analyst Q&A often reveals true concerns.\n\n"
        f"=== Management Prepared Remarks ===\n{mgmt_text}\n\n"
        f"=== Analyst Q&A Session ===\n{qa_text}\n\n"
        f"=== Pre-Computed NLP Signals ===\n{features_summary}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRASTIVE (7d): company features vs sector averages
# ═══════════════════════════════════════════════════════════════════════════════

def contrastive(transcript: str, company_features: dict, sector_avg_features: dict,
                symbol: str, company_name: str, date: str, quarter: str,
                sector: str, task: str = "ternary") -> tuple[str, str]:
    contrast_lines = []
    for key in sorted(company_features.keys()):
        comp_val = company_features[key]
        sect_val = sector_avg_features.get(key, 0.0)
        diff = comp_val - sect_val
        if abs(diff) < 1e-6:
            direction = "= SECTOR AVG"
        elif diff > 0:
            direction = f"ABOVE avg by {abs(diff):.4f}"
        else:
            direction = f"BELOW avg by {abs(diff):.4f}"
        clean_key = (key.replace("full_lm_", "")
                        .replace("mgmt_lm_", "mgmt.")
                        .replace("analyst_lm_", "analyst.")
                        .replace("_", " "))
        contrast_lines.append(
            f"  {clean_key:<35} {comp_val:>8.4f}   [{direction}]"
        )
    contrast_text = "\n".join(contrast_lines)

    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Sector: {sector}\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Company vs. {sector} Sector Norms ===\n"
        f"{'Feature':<35} {'Value':>8}   Relative to sector\n"
        f"{'-' * 65}\n"
        f"{contrast_text}\n\n"
        f"Interpretation guide:\n"
        f"  - ABOVE avg Negative score = more negative language than peers\n"
        f"  - ABOVE avg Uncertainty = more hedging than peers (bearish signal)\n"
        f"  - ABOVE avg Positive + BELOW avg Negative = unusually bullish tone\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost INJECTION (7e): XGB prediction + SHAP + transcript
# ═══════════════════════════════════════════════════════════════════════════════

def xgb_inject(transcript: str, xgb_prediction: str, xgb_confidence: float,
               shap_top_features: list[tuple[str, float]],
               symbol: str, company_name: str, date: str, quarter: str,
               task: str = "ternary") -> tuple[str, str]:
    shap_lines = []
    for name, value in shap_top_features:
        direction = "pushes toward UP" if value > 0 else "pushes toward DOWN"
        clean_name = name.replace("_", " ").replace("full ", "").replace("mgmt ", "mgmt: ")
        shap_lines.append(f"  {clean_name:<40} {value:>+.4f}  ({direction})")
    shap_text = "\n".join(shap_lines)

    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Quantitative Model (XGBoost) Analysis ===\n"
        f"An XGBoost classifier trained on financial NLP features from earnings calls "
        f"has analysed this transcript and produced the following:\n\n"
        f"  Model prediction : {xgb_prediction}\n"
        f"  Model confidence : {xgb_confidence:.1%}\n\n"
        f"Top 5 features driving this prediction (SHAP values):\n"
        f"  Positive SHAP = feature pushes prediction toward UP\n"
        f"  Negative SHAP = feature pushes prediction toward DOWN\n\n"
        f"{shap_text}\n\n"
        f"=== Your Task ===\n"
        f"Read the transcript below and form your own view. You may agree with or "
        f"disagree with the model's prediction — provide your reasoning either way. "
        f"The model captures quantitative patterns; you provide contextual judgment.\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: format feature dict into readable summary text
# ═══════════════════════════════════════════════════════════════════════════════

def format_features_summary(features: dict) -> str:
    """
    Curated feature summary for LLM prompt injection.

    Design principles:
      - Include ONLY features an LLM can reason about directionally.
      - Add interpretive labels (bullish/bearish/neutral) so the model
        doesn't have to calibrate raw ratios it has never seen before.
      - Highlight management-vs-analyst divergence — a key signal that
        raw numbers alone surface but raw transcript text can obscure.
      - EXCLUDE features that are meaningful to XGBoost but opaque to
        an LLM: NER density per token, readability indices, type-token
        ratio, fraction uppercase, vocabulary richness.  These add
        token cost without aiding LLM reasoning.
    """
    lines = []

    # ── 1. Overall Transcript Tone (L-M net signal) ──────────────────────
    full_pos = features.get("full_lm_positive", 0)
    full_neg = features.get("full_lm_negative", 0)
    full_unc = features.get("full_lm_uncertainty", 0)
    full_lit = features.get("full_lm_litigious", 0)
    full_con = features.get("full_lm_constraining", 0)
    full_strong = features.get("full_lm_strong_modal", 0)
    full_weak = features.get("full_lm_weak_modal", 0)

    net_sentiment = full_pos - full_neg

    if net_sentiment > 0.005:
        tone_label = "NET POSITIVE — more positive than negative language"
    elif net_sentiment < -0.005:
        tone_label = "NET NEGATIVE — more negative than positive language"
    else:
        tone_label = "NEUTRAL — positive and negative language roughly balanced"

    lines.append("=== Overall Transcript Tone ===")
    lines.append(f"  Assessment: {tone_label}")
    lines.append(f"  Positive word ratio:     {full_pos:.4f}")
    lines.append(f"  Negative word ratio:     {full_neg:.4f}")
    lines.append(f"  Net sentiment (pos-neg): {net_sentiment:+.4f}")
    lines.append(f"  Uncertainty ratio:       {full_unc:.4f}"
                 + ("  (elevated — hedging language detected)" if full_unc > 0.015 else ""))
    lines.append(f"  Litigious ratio:         {full_lit:.4f}"
                 + ("  (elevated — legal risk language detected)" if full_lit > 0.010 else ""))

    if full_con > 0.005:
        lines.append(f"  Constraining ratio:      {full_con:.4f}"
                     + "  (notable — language about constraints/limitations)")

    # Strong vs weak modals — confidence proxy
    if full_strong > 0 or full_weak > 0:
        if full_strong > full_weak * 1.5:
            modal_note = "confident language dominates ('will', 'shall')"
        elif full_weak > full_strong * 1.5:
            modal_note = "hedged language dominates ('may', 'could', 'might')"
        else:
            modal_note = "mixed confidence language"
        lines.append(f"  Modal language:          {modal_note}")

    # ── 2. Management vs Analyst Tone Divergence ─────────────────────────
    mgmt_pos = features.get("mgmt_lm_positive", 0)
    mgmt_neg = features.get("mgmt_lm_negative", 0)
    mgmt_unc = features.get("mgmt_lm_uncertainty", 0)
    anlst_pos = features.get("analyst_lm_positive", 0)
    anlst_neg = features.get("analyst_lm_negative", 0)
    anlst_unc = features.get("analyst_lm_uncertainty", 0)

    mgmt_net = mgmt_pos - mgmt_neg
    anlst_net = anlst_pos - anlst_neg

    lines.append("\n=== Management vs Analyst Tone ===")
    lines.append(f"  Management net sentiment: {mgmt_net:+.4f}"
                 + (f"  (positive)" if mgmt_net > 0.005
                    else f"  (negative)" if mgmt_net < -0.005
                    else f"  (neutral)"))
    lines.append(f"  Analyst net sentiment:    {anlst_net:+.4f}"
                 + (f"  (positive)" if anlst_net > 0.005
                    else f"  (negative)" if anlst_net < -0.005
                    else f"  (neutral)"))

    # Divergence interpretation
    divergence = mgmt_net - anlst_net
    if divergence > 0.010:
        lines.append(f"  DIVERGENCE: Management significantly more positive than analysts"
                     f" (gap: {divergence:+.4f})")
    elif divergence < -0.010:
        lines.append(f"  DIVERGENCE: Analysts more positive than management"
                     f" (gap: {divergence:+.4f})")
    else:
        lines.append(f"  Alignment: management and analyst tone are consistent"
                     f" (gap: {divergence:+.4f})")

    # Uncertainty divergence
    unc_div = mgmt_unc - anlst_unc
    if abs(unc_div) > 0.005:
        who_hedges = "Management" if unc_div > 0 else "Analysts"
        lines.append(f"  {who_hedges} use more hedging/uncertainty language"
                     f" (uncertainty gap: {unc_div:+.4f})")

    # ── 3. Data Density (proxy for quantitative vs qualitative call) ─────
    frac_numeric = features.get("full_frac_numeric", 0)
    frac_dollar = features.get("full_frac_dollar", 0)
    frac_percent = features.get("full_frac_percent", 0)

    if frac_numeric > 0 or frac_dollar > 0 or frac_percent > 0:
        lines.append("\n=== Data Density ===")
        if frac_numeric > 0.04:
            lines.append(f"  Numeric token density: {frac_numeric:.4f}"
                         f"  (high — data-rich call with many specific figures)")
        elif frac_numeric > 0.02:
            lines.append(f"  Numeric token density: {frac_numeric:.4f}"
                         f"  (moderate — some quantitative detail)")
        elif frac_numeric > 0:
            lines.append(f"  Numeric token density: {frac_numeric:.4f}"
                         f"  (low — qualitative discussion dominates)")
        if frac_dollar > 0.005:
            lines.append(f"  Dollar-amount mentions: {frac_dollar:.4f}")
        if frac_percent > 0.005:
            lines.append(f"  Percentage mentions:    {frac_percent:.4f}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE EXTRACT → PREDICT
# Stage 1: Extract structured signals from transcript (run once, cache)
# Stage 2: Predict from the extraction (fast, ~500 token input)
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_SYSTEM = """\
You are a financial analyst. Your job is to extract key signals from an \
earnings call transcript. Extract ONLY factual information present in the \
transcript — do not speculate, infer, or predict. If a signal is not \
mentioned, write "Not mentioned"."""

EXTRACTION_USER_TEMPLATE = """\
Company: {symbol} ({company_name})
Earnings call date: {date} | Period: {quarter}

=== Transcript ===
{transcript}

=== Extract the following signals ===

Respond in EXACTLY this format (one line per field):

REVENUE: [reported figure and whether it beat/missed/met consensus — quote numbers]
EPS: [reported figure and whether it beat/missed/met consensus — quote numbers]
GUIDANCE: [was full-year or next-quarter outlook raised/maintained/lowered — quote exact language]
MARGINS: [gross/operating margin direction — expanding/contracting/stable — cite figures if given]
MGMT_TONE: [3 most bullish phrases from management, verbatim]
MGMT_CONCERNS: [3 most cautious/hedging phrases from management, verbatim]
ANALYST_PUSHBACK: [top 3 topics analysts challenged or probed on]
SURPRISE: [single most unexpected disclosure or data point in the call]"""


def extraction_prompt(transcript: str, symbol: str, company_name: str,
                      date: str, quarter: str) -> tuple[str, str]:
    """Stage 1 prompt: extract structured signals from transcript."""
    user = EXTRACTION_USER_TEMPLATE.format(
        symbol=symbol, company_name=company_name,
        date=date, quarter=quarter,
        transcript=transcript,
    )
    return EXTRACTION_SYSTEM, user


def two_stage_predict(extraction: str, symbol: str, company_name: str,
                      date: str, quarter: str,
                      task: str = "ternary") -> tuple[str, str]:
    """
    Stage 2 prompt (direct): predict from pre-extracted signals.
    No raw transcript — only the structured extraction.
    """
    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"The following signals were extracted from the earnings call transcript "
        f"by a financial analyst. Use them to predict the stock price reaction.\n\n"
        f"=== Extracted Signals ===\n{extraction}\n\n"
        + _DIRECT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


def two_stage_predict_cot(extraction: str, symbol: str, company_name: str,
                          date: str, quarter: str,
                          task: str = "ternary") -> tuple[str, str]:
    """
    Stage 2 prompt (CoT): predict from pre-extracted signals with
    structured 4-step reasoning.
    """
    user = (
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"The following signals were extracted from the earnings call transcript "
        f"by a financial analyst. Use them to predict the stock price reaction.\n\n"
        f"=== Extracted Signals ===\n{extraction}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user


# ═══════════════════════════════════════════════════════════════════════════════
# RAG FEW-SHOT (new): retrieved similar exemplars + transcript
# Same structure as few_shot_cot but exemplars are similarity-retrieved,
# not randomly selected. Called from 06_gpt_experiments.py after embedding.
# ═══════════════════════════════════════════════════════════════════════════════

def rag_few_shot(transcript: str, symbol: str, company_name: str,
                 date: str, quarter: str, exemplars: list[dict],
                 task: str = "ternary") -> tuple[str, str]:
    """
    RAG few-shot prompt. Exemplars are the most semantically similar
    transcripts to the query, retrieved via cosine similarity on embeddings.

    exemplars: list of dicts with keys:
        symbol, company_name, date, quarter,
        transcript_excerpt, label, reasoning, similarity_score
    """
    examples_text = ""
    for i, ex in enumerate(exemplars, 1):
        sim = ex.get("similarity_score", 0.0)
        examples_text += (
            f"=== Retrieved Example {i} (similarity: {sim:.3f}) ===\n"
            f"Company: {ex['symbol']} ({ex.get('company_name', '')})\n"
            f"Date: {ex['date']} | Period: {ex['quarter']}\n\n"
            f"Transcript excerpt:\n{ex['transcript_excerpt']}\n\n"
            f"<reasoning>\n{ex['reasoning']}\n</reasoning>\n"
            f"<prediction>{ex['label']}</prediction>\n\n"
        )

    user = (
        f"The following examples were retrieved because their earnings call "
        f"transcripts are most similar to the one you must now predict. "
        f"Use their patterns as guidance.\n\n"
        f"{examples_text}"
        f"=== Now predict ===\n"
        f"Company: {symbol} ({company_name})\n"
        f"Earnings call date: {date} | Period: {quarter}\n\n"
        f"=== Transcript ===\n{transcript}\n\n"
        + _COT_INSTRUCTION.format(label_pipe=_label_pipe(task))
    )
    return _system(task), user