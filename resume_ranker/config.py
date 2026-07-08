"""
Central configuration for the Resume Ranking System.
All parameters, paths, weights, and model settings are controlled here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ─── Project Root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
RANKINGS_DIR = ROOT_DIR / "outputs" / "rankings"

# Ensure dirs exist
for _d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR, RANKINGS_DIR, PROCESSED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Model Selection ─────────────────────────────────────────────────────────
# Options: "tfidf", "bert", "logistic_regression", "random_forest"
ACTIVE_MODEL = "bert"

# ─── Section Weights ─────────────────────────────────────────────────────────
SECTION_WEIGHTS: Dict[str, float] = {
    "skills": 0.50,
    "experience": 0.30,
    "education": 0.15,
    "certifications": 0.05,
}

# ─── Scoring Combination Weights ─────────────────────────────────────────────
SCORE_COMBINATION_WEIGHTS: Dict[str, float] = {
    "semantic_similarity": 0.45,
    "skill_overlap": 0.40,
    "section_weighted": 0.15,
}

# ─── BERT / Sentence Transformer ─────────────────────────────────────────────
BERT_MODEL_NAME = "all-MiniLM-L6-v2"
BERT_BATCH_SIZE = 16
BERT_MAX_SEQ_LENGTH = 512

# ─── TF-IDF ──────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1

# ─── ML Models ───────────────────────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS: Dict = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42,
    "solver": "lbfgs",
    "multi_class": "auto",
}

RANDOM_FOREST_PARAMS: Dict = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
    "min_samples_split": 5,
}

# ─── Skill Extraction ────────────────────────────────────────────────────────
SKILL_ONTOLOGY_PATH = ROOT_DIR / "skill_extraction" / "skill_ontology.json"
SPACY_MODEL = "en_core_web_sm"

# ─── Text Preprocessing ──────────────────────────────────────────────────────
NLTK_STOPWORDS_LANG = "english"
LEMMATIZE = True
REMOVE_STOPWORDS = True
MIN_TOKEN_LENGTH = 2

# ─── Evaluation ──────────────────────────────────────────────────────────────
EVAL_K_VALUES: List[int] = [1, 3, 5, 10]
RELEVANCE_THRESHOLD = 0.5  # score above this = relevant

# ─── Explainability ──────────────────────────────────────────────────────────
SHAP_MAX_DISPLAY = 20
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 500

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = False
MAX_UPLOAD_SIZE_MB = 10

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = ROOT_DIR / "outputs" / "system.log"

# ─── MLflow ──────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = str(ROOT_DIR / "outputs" / "mlruns")
MLFLOW_EXPERIMENT_NAME = "resume_ranking"

# ─── Fairness ────────────────────────────────────────────────────────────────
FAIRNESS_SCORE_VARIANCE_THRESHOLD = 0.15
FAIRNESS_LOG_PATH = ROOT_DIR / "outputs" / "reports" / "fairness_report.json"

# ─── PDF Parsing ─────────────────────────────────────────────────────────────
PDF_PARSER = "pdfplumber"  # "pdfplumber" or "pymupdf"

# ─── Save / Load Paths ───────────────────────────────────────────────────────
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
BERT_EMBEDDINGS_CACHE = MODELS_DIR / "bert_embeddings_cache.pkl"


@dataclass
class Config:
    """Dataclass for runtime config override."""
    active_model: str = ACTIVE_MODEL
    section_weights: Dict[str, float] = field(default_factory=lambda: SECTION_WEIGHTS.copy())
    score_combination_weights: Dict[str, float] = field(default_factory=lambda: SCORE_COMBINATION_WEIGHTS.copy())
    bert_model_name: str = BERT_MODEL_NAME
    tfidf_max_features: int = TFIDF_MAX_FEATURES
    eval_k_values: List[int] = field(default_factory=lambda: EVAL_K_VALUES.copy())
    shap_max_display: int = SHAP_MAX_DISPLAY
    lime_num_features: int = LIME_NUM_FEATURES

    def validate(self):
        assert self.active_model in {"tfidf", "bert", "logistic_regression", "random_forest"}, \
            f"Invalid model: {self.active_model}"
        assert abs(sum(self.section_weights.values()) - 1.0) < 1e-6, \
            "Section weights must sum to 1.0"
        assert abs(sum(self.score_combination_weights.values()) - 1.0) < 1e-6, \
            "Score combination weights must sum to 1.0"
        return self


DEFAULT_CONFIG = Config()
