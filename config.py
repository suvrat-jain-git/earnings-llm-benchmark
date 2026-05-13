"""
Central configuration for the Hybrid Pre-DL + LLM Earnings Call Sentiment Research.
All constants, paths, thresholds, and parameters in one place.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_TRANSCRIPTS_DIR = DATA_DIR / "raw_transcripts"
PRICES_DIR = DATA_DIR / "prices"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"

LM_DICT_PATH = ROOT_DIR / "L_McD_Dict_Words.csv"
ENV_FILE_PATH = ROOT_DIR / "env.env"

# Create directories
for d in [DATA_DIR, RAW_TRANSCRIPTS_DIR, PRICES_DIR, FEATURES_DIR,
          LABELS_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Company Selection ────────────────────────────────────────────────────────
N_COMPANIES_LLM = 100          # Companies for LLM experiments
HF_DATASET_TRANSCRIPTS = "kurry/sp500_earnings_transcripts"
HF_DATASET_SECTORS = "glopardo/sp500-earnings-transcripts"

# ── Label Construction ───────────────────────────────────────────────────────
LABEL_THRESHOLD = 0.005        # ±0.5% for UP/DOWN/FLAT boundary
PRICE_WINDOW_DAYS = 10         # Trading days before/after call to fetch prices

# NYSE/NASDAQ regular session hours (ET)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# ── Feature Extraction ───────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_SVD_COMPONENTS = 50
BOW_MAX_FEATURES = 2000
BOW_SVD_COMPONENTS = 30
# spaCy model for POS tagging, NER, and dependency parsing in feature extraction.
# "en_core_web_sm"  — fast, ~12MB,  lower NER/POS accuracy (averaged word vectors)
# "en_core_web_md"  — balanced, ~43MB, GloVe 20k vectors
# "en_core_web_lg"  — best quality, ~741MB, GloVe 685k vectors, more accurate NER
# For research / publication quality, use en_core_web_lg.
# Install with: python -m spacy download en_core_web_lg
SPACY_MODEL = "en_core_web_lg"

# ── XGBoost ──────────────────────────────────────────────────────────────────
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 3
OPTUNA_N_TRIALS = 50
SHAP_TOP_K = 5                 # Top K SHAP features for GPT injection

# ── Azure OpenAI / GPT ──────────────────────────────────────────────────────
MAX_TRANSCRIPT_TOKENS = 12000  # Truncate beyond this
GPT_TEMPERATURE = 0.0
GPT_MAX_TOKENS = 500
GPT_FEW_SHOT_N = 3            # Number of few-shot exemplars (1 per class)
API_RETRY_MAX = 5
API_RETRY_BASE_DELAY = 2.0    # Seconds, exponential backoff

# ── Two-Stage Extraction ─────────────────────────────────────────────────────
EXTRACTIONS_PATH = FEATURES_DIR / "extractions_100.json"
EXTRACTION_MAX_TOKENS = 600    # Max output tokens for Stage 1 (extraction)

# ── Evaluation ───────────────────────────────────────────────────────────────
BOOTSTRAP_N_ITERATIONS = 2000
SIGNIFICANCE_ALPHA = 0.05

# ── Label Sensitivity & Multi-Day Returns ────────────────────────────────────
SENSITIVITY_THRESHOLDS = [0.0025, 0.005, 0.01, 0.02]
RETURN_HORIZONS = [1, 2, 3]           # trading days post-call
LABEL_MODE = "raw"                     # "raw" or "excess" (market-adjusted)

# ── GPT Calibration ──────────────────────────────────────────────────────────
GPT_CALIBRATION_RUNS = 3
GPT_CALIBRATION_TEMPERATURE = 0.3

# ── Baselines ─────────────────────────────────────────────────────────────────
RANDOM_BASELINE_RUNS = 1000      # Number of random-prediction Monte Carlo runs

# ── Diebold-Mariano Test ──────────────────────────────────────────────────────
DM_LOSS = "01"                   # "01" (0-1 loss) or "se" (squared error on return)

# ── Expected Calibration Error ────────────────────────────────────────────────
ECE_N_BINS = 10                  # Number of confidence bins for reliability diagram

# ── Temporal Leakage Gate (05b) ───────────────────────────────────────────────
# When True, XGB-fullcorpus training only uses transcripts whose date is
# strictly BEFORE the earliest date in the 100-company eval set.
XGB_FULLCORPUS_TEMPORAL_GATE = True

# ── XGBoost Feature Set ──────────────────────────────────────────────────────
XGB_CANONICAL_FEATURE_SET = "sentiment_only"  # "sentiment_only", "full", or "pca"
XGB_PCA_COMPONENTS = 20

# ── Experiment Names ─────────────────────────────────────────────────────────
EXPERIMENTS = [
    # Classical ML baselines
    "XGB-full",
    "XGB-sentiment-only",
    "XGB-fullcorpus",
    # Deep Learning baseline
    "FinBERT-finetuned",
    # Zero-shot LLM
    "GPT-zero",
    "GPT-zero-CoT",
    "GPT-zero-calibrated",
    # Few-shot LLM
    "GPT-few",
    "GPT-few-CoT",
    "GPT-rag-few",
    "GPT-few-sector",
    "GPT-few-CoT-sector",
    # Two-stage LLM
    "GPT-two-stage",
    "GPT-two-stage-CoT",
    # Hybrid (feature injection)
    "GPT-feat-inject",
    "GPT-feat-only",
    "GPT-speaker-seg",
    "GPT-contrastive",
    "GPT-xgb-inject",
    # Ensembles
    "GPT-core-ensemble",
    "GPT-hybrid-ensemble",
    "GPT-enhanced-ensemble",
    "GPT-full-ensemble",
    "GPT-disagreement-ensemble",
]

TASKS = ["binary", "ternary"]
