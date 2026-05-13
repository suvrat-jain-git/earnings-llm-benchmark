"""
Feature extraction utilities: TF-IDF, BoW, POS, NER, readability,
statistical, word shape, syntactic features.
"""
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import textstat

from utils.transcript_parser import preprocess_spacy, tokenize_simple
from utils.lm_dictionary import LMDictionary


# ── POS Tags (Universal Dependencies) ───────────────────────────────────────
POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE",
]

# ── NER Entity Types (spaCy) ────────────────────────────────────────────────
NER_TYPES = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "ORDINAL", "CARDINAL",
]


def extract_pos_features(doc) -> dict:
    """Normalized POS tag distribution from spaCy Doc."""
    counts = Counter(token.pos_ for token in doc if not token.is_space)
    total = sum(counts.values()) or 1
    return {f"pos_{tag}": counts.get(tag, 0) / total for tag in POS_TAGS}


def extract_ner_features(doc) -> dict:
    """NER entity type counts (normalized by doc length) from spaCy Doc."""
    counts = Counter(ent.label_ for ent in doc.ents)
    n_tokens = len(doc) or 1
    return {f"ner_{etype}": counts.get(etype, 0) / n_tokens for etype in NER_TYPES}


def extract_readability_features(text: str) -> dict:
    """Readability scores via textstat."""
    if not text or len(text.split()) < 10:
        return {"flesch_kincaid": 0.0, "gunning_fog": 0.0, "coleman_liau": 0.0}
    return {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "coleman_liau": textstat.coleman_liau_index(text),
    }


def extract_statistical_features(text: str, tokens: list[str]) -> dict:
    """Document-level statistical features."""
    n_tokens = len(tokens) or 1
    unique_tokens = set(t.lower() for t in tokens)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = len(sentences) or 1
    avg_sent_len = n_tokens / n_sentences

    return {
        "doc_length": n_tokens,
        "type_token_ratio": len(unique_tokens) / n_tokens,
        "avg_sentence_length": avg_sent_len,
        "vocabulary_richness": len(unique_tokens),
        "n_sentences": n_sentences,
    }


def extract_word_shape_features(tokens: list[str]) -> dict:
    """Word shape features: fraction uppercase, numeric, $, %."""
    n = len(tokens) or 1
    n_upper = sum(1 for t in tokens if t.isupper() and len(t) > 1)
    n_numeric = sum(1 for t in tokens if any(c.isdigit() for c in t))
    n_dollar = sum(1 for t in tokens if "$" in t)
    n_percent = sum(1 for t in tokens if "%" in t)

    return {
        "frac_uppercase": n_upper / n,
        "frac_numeric": n_numeric / n,
        "frac_dollar": n_dollar / n,
        "frac_percent": n_percent / n,
    }


def extract_syntactic_features(doc) -> dict:
    """Avg dependency tree depth and noun phrase count."""
    def tree_depth(token):
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
            if depth > 100:
                break
        return depth

    # Sample to avoid slow computation on long docs
    sample_tokens = list(doc)[:5000]
    depths = [tree_depth(t) for t in sample_tokens if not t.is_space]
    avg_depth = np.mean(depths) if depths else 0.0
    n_noun_chunks = len(list(doc.noun_chunks))
    n_tokens = len(doc) or 1

    return {
        "avg_dep_tree_depth": avg_depth,
        "noun_phrase_density": n_noun_chunks / n_tokens,
    }


def extract_all_features_for_text(text: str, lm_dict: LMDictionary,
                                  spacy_model: str = None,
                                  prefix: str = "") -> dict:
    """
    Extract all non-BoW/TF-IDF features for a single text segment.

    Args:
        text: The text to extract features from
        lm_dict: LMDictionary instance
        spacy_model: spaCy model name. Defaults to config.SPACY_MODEL when None.
        prefix: Column prefix (e.g., "mgmt_", "analyst_", "full_")

    Returns:
        dict of feature_name → value
    """
    import config as _config
    spacy_model = spacy_model or _config.SPACY_MODEL
    if not text or len(text.strip()) < 10:
        # Return zeros for empty segments
        features = {}
        for tag in POS_TAGS:
            features[f"{prefix}pos_{tag}"] = 0.0
        for etype in NER_TYPES:
            features[f"{prefix}ner_{etype}"] = 0.0
        for cat in ["Negative", "Positive", "Uncertainty", "Litigious",
                     "Strong_Modal", "Weak_Modal", "Constraining",
                     "net_sentiment_ratio"]:
            features[f"{prefix}lm_{cat.lower()}"] = 0.0
        features.update({
            f"{prefix}flesch_kincaid": 0.0, f"{prefix}gunning_fog": 0.0,
            f"{prefix}coleman_liau": 0.0, f"{prefix}doc_length": 0,
            f"{prefix}type_token_ratio": 0.0, f"{prefix}avg_sentence_length": 0.0,
            f"{prefix}vocabulary_richness": 0, f"{prefix}n_sentences": 0,
            f"{prefix}frac_uppercase": 0.0, f"{prefix}frac_numeric": 0.0,
            f"{prefix}frac_dollar": 0.0, f"{prefix}frac_percent": 0.0,
            f"{prefix}avg_dep_tree_depth": 0.0, f"{prefix}noun_phrase_density": 0.0,
        })
        return features

    # Tokenize
    tokens = tokenize_simple(text)
    doc = preprocess_spacy(text, spacy_model)

    features = {}

    # POS
    pos = extract_pos_features(doc)
    features.update({f"{prefix}{k}": v for k, v in pos.items()})

    # NER
    ner = extract_ner_features(doc)
    features.update({f"{prefix}{k}": v for k, v in ner.items()})

    # L-M Sentiment
    lm_scores = lm_dict.score_tokens(tokens)
    for cat in ["Negative", "Positive", "Uncertainty", "Litigious",
                "Strong_Modal", "Weak_Modal", "Constraining", "net_sentiment_ratio"]:
        features[f"{prefix}lm_{cat.lower()}"] = lm_scores[cat]

    # Readability
    read = extract_readability_features(text)
    features.update({f"{prefix}{k}": v for k, v in read.items()})

    # Statistical
    stat = extract_statistical_features(text, tokens)
    features.update({f"{prefix}{k}": v for k, v in stat.items()})

    # Word shape
    ws = extract_word_shape_features(tokens)
    features.update({f"{prefix}{k}": v for k, v in ws.items()})

    # Syntactic
    syn = extract_syntactic_features(doc)
    features.update({f"{prefix}{k}": v for k, v in syn.items()})

    return features


def build_tfidf_svd(texts: list[str], max_features: int = 5000,
                    n_components: int = 50, prefix: str = "tfidf_svd_"):
    """
    Fit TF-IDF + TruncatedSVD on a list of texts.
    Returns: (feature_matrix, vectorizer, svd_model, feature_names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    feature_names = [f"{prefix}{i}" for i in range(n_components)]
    return svd_matrix, vectorizer, svd, feature_names


def build_bow_svd(texts: list[str], max_features: int = 2000,
                  n_components: int = 30, prefix: str = "bow_svd_"):
    """
    Fit BoW CountVectorizer + TruncatedSVD on a list of texts.
    Returns: (feature_matrix, vectorizer, svd_model, feature_names)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    bow_matrix = vectorizer.fit_transform(texts)

    n_components = min(n_components, bow_matrix.shape[0] - 1, bow_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_matrix = svd.fit_transform(bow_matrix)

    feature_names = [f"{prefix}{i}" for i in range(n_components)]
    return svd_matrix, vectorizer, svd, feature_names
