"""
Transcript parsing: speaker segmentation, role classification, preprocessing.
Works with the kurry/sp500_earnings_transcripts HuggingFace dataset format
where structured_content is a list of {speaker, text} dicts.
"""
import re
import spacy
from functools import lru_cache

# Management role keywords (case-insensitive match)
_MGMT_KEYWORDS = [
    "ceo", "chief executive", "president", "cfo", "chief financial",
    "coo", "chief operating", "cto", "chief technology",
    "chairman", "chairwoman", "chairperson", "vice president",
    "vp", "svp", "evp", "director", "treasurer", "controller",
    "general counsel", "secretary", "head of", "managing director",
    "founder", "co-founder",
]

# Analyst-side keywords
_ANALYST_KEYWORDS = [
    "analyst", "research", "securities", "capital", "morgan",
    "goldman", "barclays", "jpmorgan", "citigroup", "bofa",
    "wells fargo", "ubs", "credit suisse", "deutsche bank",
    "bernstein", "needham", "piper", "raymond james", "stifel",
    "jefferies", "cowen", "oppenheimer", "rbc", "td ",
]

# Operator / moderator keywords
_OPERATOR_KEYWORDS = ["operator", "moderator", "conference call"]


def classify_speaker(speaker_text: str) -> str:
    """Classify a speaker line into 'management', 'analyst', or 'operator'."""
    s = speaker_text.lower().strip()

    for kw in _OPERATOR_KEYWORDS:
        if kw in s:
            return "operator"

    for kw in _MGMT_KEYWORDS:
        if kw in s:
            return "management"

    for kw in _ANALYST_KEYWORDS:
        if kw in s:
            return "analyst"

    # Heuristic: if the speaker line contains a company/firm name pattern
    # or a question mark, likely analyst; otherwise default to management
    # (prepared remarks speakers are usually management)
    if "?" in s:
        return "analyst"

    return "management"  # conservative default


def parse_structured_content(structured_content) -> dict:
    """
    Parse structured_content (list of {speaker, text} dicts or JSON string)
    into segmented text by role.

    Returns:
        {
            "full_text": str,
            "management_text": str,
            "analyst_text": str,
            "operator_text": str,
            "speakers": [{"speaker": str, "role": str, "text": str}, ...],
            "qa_pairs": [{"analyst_speaker": str, "question": str,
                          "mgmt_speaker": str, "answer": str}, ...]
        }
    """
    import json

    if structured_content is None:
        return None

    # Handle JSON string
    if isinstance(structured_content, str):
        try:
            structured_content = json.loads(structured_content)
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(structured_content, list) or len(structured_content) == 0:
        return None

    full_parts = []
    mgmt_parts = []
    analyst_parts = []
    operator_parts = []
    speakers_list = []

    for entry in structured_content:
        if not isinstance(entry, dict):
            continue
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "")
        if not text or not text.strip():
            continue

        role = classify_speaker(speaker)
        full_parts.append(text)
        speakers_list.append({"speaker": speaker, "role": role, "text": text})

        if role == "management":
            mgmt_parts.append(text)
        elif role == "analyst":
            analyst_parts.append(text)
        else:
            operator_parts.append(text)

    # Extract Q&A pairs: find analyst question followed by management answer
    qa_pairs = []
    for i, sp in enumerate(speakers_list):
        if sp["role"] == "analyst" and i + 1 < len(speakers_list):
            next_sp = speakers_list[i + 1]
            if next_sp["role"] == "management":
                qa_pairs.append({
                    "analyst_speaker": sp["speaker"],
                    "question": sp["text"],
                    "mgmt_speaker": next_sp["speaker"],
                    "answer": next_sp["text"],
                })

    return {
        "full_text": "\n\n".join(full_parts),
        "management_text": "\n\n".join(mgmt_parts),
        "analyst_text": "\n\n".join(analyst_parts),
        "operator_text": "\n\n".join(operator_parts),
        "speakers": speakers_list,
        "qa_pairs": qa_pairs,
    }


def parse_raw_content(content: str) -> dict:
    """
    Fallback parser for plain text transcripts (no speaker segmentation).
    Returns same structure but with empty management/analyst splits.
    """
    if not content or not content.strip():
        return None

    return {
        "full_text": content.strip(),
        "management_text": "",
        "analyst_text": "",
        "operator_text": "",
        "speakers": [],
        "qa_pairs": [],
    }


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for L-M scoring."""
    return re.findall(r"[A-Za-z]+", text)


@lru_cache(maxsize=8)
def load_spacy_model(model_name: str = None):
    """Load and cache spaCy model. Defaults to config.SPACY_MODEL."""
    import config as _config
    model_name = model_name or _config.SPACY_MODEL
    return spacy.load(model_name, disable=["textcat"])


def preprocess_spacy(text: str, model_name: str = None,
                     max_length: int = 1_000_000):
    """
    Run spaCy pipeline on text. Returns the spaCy Doc.
    Handles large documents by increasing max_length.
    model_name defaults to config.SPACY_MODEL when None.
    """
    import config as _config
    model_name = model_name or _config.SPACY_MODEL
    nlp = load_spacy_model(model_name)
    nlp.max_length = max(nlp.max_length, max_length)
    return nlp(text[:max_length])


def truncate_transcript(parsed: dict, max_tokens: int = 12000) -> str:
    """
    Section-aware transcript truncation for LLM prompt injection.

    Improvements over previous version:
      1. Strips safe-harbor disclaimers and operator boilerplate (~200-500
         tokens reclaimed for actual signal).
      2. Allocates token budget proportionally: 60 % prepared remarks,
         40 % Q&A — guarantees the model always sees BOTH sections.
      3. Prioritises the LONGEST Q&A exchanges (most contentious and
         informative) rather than the first 3 chronologically.
      4. Donates unused budget from short prepared remarks to Q&A.
      5. Falls back gracefully when structured fields are missing.

    Args:
        parsed: dict with keys full_text, management_text, analyst_text,
                qa_pairs (from parse_structured_content / parse_raw_content)
        max_tokens: maximum token budget (whitespace-split words)

    Returns:
        Structured transcript string with [PREPARED REMARKS] and
        [KEY Q&A EXCHANGES] section headers.
    """
    mgmt_text = parsed.get("management_text", "")
    qa_pairs = parsed.get("qa_pairs", [])
    full_text = parsed.get("full_text", "")

    # ── Fallback: no structured content available ────────────────────────
    if not mgmt_text and not qa_pairs:
        cleaned = strip_boilerplate(full_text)
        words = cleaned.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens])
        return cleaned

    # ── Budget allocation ────────────────────────────────────────────────
    # Reserve a small overhead for section headers (~30 tokens)
    effective_budget = max_tokens - 30
    mgmt_budget = int(effective_budget * 0.60)
    qa_budget = int(effective_budget * 0.40)

    # ── Prepared remarks ─────────────────────────────────────────────────
    mgmt_cleaned = strip_boilerplate(mgmt_text)
    mgmt_words = mgmt_cleaned.split()
    if len(mgmt_words) > mgmt_budget:
        mgmt_final = " ".join(mgmt_words[:mgmt_budget])
    else:
        mgmt_final = mgmt_cleaned
        # Donate unused budget to Q&A
        qa_budget += (mgmt_budget - len(mgmt_words))

    # ── Q&A exchanges ────────────────────────────────────────────────────
    qa_section = ""
    if qa_pairs:
        # Sort by exchange length descending — longest exchanges are
        # typically the most contentious and informative.  An analyst
        # spending 3 follow-ups on margins tells you more than a
        # 1-sentence "congratulations on a great quarter."
        scored_pairs = []
        for qa in qa_pairs:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            length = len(q.split()) + len(a.split())
            scored_pairs.append((length, qa))

        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        qa_parts = []
        qa_tokens_used = 0
        for _, qa in scored_pairs:
            analyst = qa.get("analyst_speaker", "Analyst")
            mgmt_speaker = qa.get("mgmt_speaker", "Management")
            q = qa.get("question", "")
            a = qa.get("answer", "")

            exchange = (
                f"Analyst ({analyst}): {q}\n"
                f"Management ({mgmt_speaker}): {a}"
            )
            exchange_tokens = len(exchange.split())

            if qa_tokens_used + exchange_tokens > qa_budget:
                # Try to fit a truncated version of this exchange
                remaining = qa_budget - qa_tokens_used
                if remaining > 50:  # only include if meaningful
                    exchange_words = exchange.split()
                    qa_parts.append(" ".join(exchange_words[:remaining]))
                break

            qa_parts.append(exchange)
            qa_tokens_used += exchange_tokens

        qa_section = "\n\n".join(qa_parts)
    elif parsed.get("analyst_text", ""):
        # No structured Q&A pairs but analyst_text exists — use it directly
        analyst_words = parsed["analyst_text"].split()
        if len(analyst_words) > qa_budget:
            qa_section = " ".join(analyst_words[:qa_budget])
        else:
            qa_section = parsed["analyst_text"]

    # ── Assemble ─────────────────────────────────────────────────────────
    parts = ["[PREPARED REMARKS]", mgmt_final]

    if qa_section.strip():
        parts.append("\n[KEY Q&A EXCHANGES]")
        parts.append(qa_section)

    return "\n".join(parts)


# ── Boilerplate stripping helpers ────────────────────────────────────────────
# Safe-harbor disclaimers appear at the start of almost every earnings call
# and waste ~200-500 tokens of LLM context budget with zero predictive signal.

_BOILERPLATE_MARKERS = [
    "forward-looking statements",
    "forward looking statements",
    "safe harbor",
    "safe-harbor",
    "private securities litigation",
    "this call is being recorded",
    "this conference is being recorded",
    "actual results may differ materially",
]


def strip_boilerplate(text: str) -> str:
    """
    Remove safe-harbor disclaimers and operator intro from transcript start.

    Strategy: find where actual remarks begin by looking for common
    transition phrases (CEO/CFO starting to speak substantively).
    If no transition found, skip past the last boilerplate sentence.
    If no boilerplate detected at all, return the text unchanged.
    """
    if not text:
        return text

    lower = text[:3000].lower()  # only scan first ~750 tokens

    # Check if boilerplate is present at all
    has_boilerplate = any(marker in lower for marker in _BOILERPLATE_MARKERS)
    if not has_boilerplate:
        return text

    # Find the end of the last boilerplate sentence
    last_boilerplate_pos = 0
    for marker in _BOILERPLATE_MARKERS:
        pos = lower.find(marker)
        if pos >= 0:
            end_of_sentence = text.find(".", pos + len(marker))
            if end_of_sentence > 0:
                last_boilerplate_pos = max(last_boilerplate_pos,
                                           end_of_sentence + 1)

    if last_boilerplate_pos == 0:
        return text

    # Find the first transition phrase after the boilerplate
    transition_phrases = [
        "thank you", "good morning", "good afternoon", "good evening",
        "thanks, ", "thanks everyone", "let me start", "i'll begin",
        "i would like to", "let me begin", "i'd like to start",
        "let's get started", "turning to our results",
        "we are pleased", "we're pleased", "i'm pleased",
    ]

    search_region = text[last_boilerplate_pos:
                         last_boilerplate_pos + 2000].lower()
    best_pos = len(search_region)  # default: keep everything after boilerplate

    for phrase in transition_phrases:
        pos = search_region.find(phrase)
        if 0 <= pos < best_pos:
            best_pos = pos

    cut_point = last_boilerplate_pos + best_pos
    if 0 < cut_point < len(text):
        return text[cut_point:].lstrip()

    return text