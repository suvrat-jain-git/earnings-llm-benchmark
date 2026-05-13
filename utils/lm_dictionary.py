"""
Loughran-McDonald Financial Sentiment Dictionary loader and scorer.

The L-M dictionary uses year-based encoding: a non-zero value (e.g., 2009) in
a sentiment column means the word is flagged in that category.
"""
import pandas as pd
import numpy as np
from pathlib import Path

SENTIMENT_CATEGORIES = [
    "Negative", "Positive", "Uncertainty", "Litigious",
    "Strong_Modal", "Weak_Modal", "Constraining",
]


class LMDictionary:
    """Load and query the Loughran-McDonald financial sentiment dictionary."""

    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        # Build {category: set(UPPERCASE_WORDS)} – non-zero year value → flagged
        self.word_sets: dict[str, set[str]] = {}
        for cat in SENTIMENT_CATEGORIES:
            words = df.loc[df[cat] != 0, "Word"].str.upper().tolist()
            self.word_sets[cat] = set(words)

        # Combined set for quick membership check
        self.all_words = set()
        for s in self.word_sets.values():
            self.all_words |= s

    def score_tokens(self, tokens: list[str]) -> dict[str, float]:
        """
        Score a list of (already-tokenized, uppercased) tokens.
        Returns normalized counts per category + net_sentiment_ratio.
        """
        n = len(tokens)
        if n == 0:
            result = {cat: 0.0 for cat in SENTIMENT_CATEGORIES}
            result["net_sentiment_ratio"] = 0.0
            return result

        upper_tokens = [t.upper() for t in tokens]

        counts = {}
        for cat in SENTIMENT_CATEGORIES:
            counts[cat] = sum(1 for t in upper_tokens if t in self.word_sets[cat])

        # Normalize by document length
        result = {cat: counts[cat] / n for cat in SENTIMENT_CATEGORIES}

        # Net sentiment ratio: (positive - negative) / (positive + negative + 1)
        pos = counts["Positive"]
        neg = counts["Negative"]
        result["net_sentiment_ratio"] = (pos - neg) / (pos + neg + 1)

        # Also store raw counts (useful for feature injection text)
        for cat in SENTIMENT_CATEGORIES:
            result[f"{cat}_raw"] = counts[cat]

        return result
