"""
Shared prediction-parsing utility used by 06_gpt_experiments.py and
07_hybrid_experiments.py.

Keeping parse_prediction in one place ensures any fix to the parsing
logic propagates to all experiment scripts automatically.
"""
import re


def parse_prediction(response: str, task: str) -> str | None:
    """
    Extract predicted label from GPT response.

    Handles:
      1. New XML-tag format  <prediction>UP</prediction>  (most reliable)
      2. Legacy plain-text   PREDICTION: UP
      3. Last non-empty line (fallback for zero-shot direct responses)
      4. Last occurrence anywhere in response (final fallback)

    Args:
        response: raw string returned by the model
        task: "binary" (UP/DOWN) or "ternary" (UP/DOWN/FLAT)

    Returns:
        Uppercase label string, or None if no valid label found.
    """
    text = response.upper().strip()
    valid_labels = ["UP", "DOWN", "FLAT"] if task == "ternary" else ["UP", "DOWN"]

    # 1. New format: <prediction>LABEL</prediction>  (most reliable)
    match = re.search(r"<PREDICTION>\s*(UP|DOWN|FLAT)\s*</PREDICTION>", text)
    if match:
        label = match.group(1)
        if task == "binary" and label == "FLAT":
            return None
        return label

    # 2. Legacy format: PREDICTION: LABEL
    match = re.search(r"PREDICTION:\s*(UP|DOWN|FLAT)", text)
    if match:
        label = match.group(1)
        if task == "binary" and label == "FLAT":
            return None
        return label

    # 3. Last non-empty line (fallback for zero-shot direct responses)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        last = re.sub(r"<[^>]+>", "", lines[-1]).strip()
        for label in valid_labels:
            if re.search(rf"\b{label}\b", last):
                return label

    # 4. Last occurrence anywhere in response
    last_pos = -1
    found_label = None
    for label in valid_labels:
        pos = text.rfind(label)
        if pos > last_pos:
            last_pos = pos
            found_label = label

    return found_label
