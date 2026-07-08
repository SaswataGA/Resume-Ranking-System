"""
ML models: Logistic Regression and Random Forest for resume ranking.
These models use feature vectors built from TF-IDF + skill overlap + section scores.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from utils.helpers import save_joblib, load_joblib
from utils.logger import log
import config


FEATURE_NAMES = [
    "tfidf_similarity",
    "bert_similarity",
    "skill_jaccard",
    "skill_overlap_count",
    "skill_precision",
    "skill_recall",
    "skill_f1",
    "skills_section_sim",
    "experience_section_sim",
    "education_section_sim",
    "cert_section_sim",
    "resume_length",
    "num_candidate_skills",
    "num_job_skills",
    "years_experience_est",
]


def build_feature_vector(
    tfidf_sim: float,
    bert_sim: float,
    skill_overlap: Dict,
    section_sims: Dict[str, float],
    resume_meta: Dict,
) -> np.ndarray:
    """
    Build a feature vector for ML models.
    
    Args:
        tfidf_sim: TF-IDF cosine similarity score
        bert_sim: BERT semantic similarity score
        skill_overlap: Output from SkillExtractor.skill_overlap()
        section_sims: Per-section similarity scores
        resume_meta: Metadata (resume_length, num_skills, etc.)
        
    Returns:
        Feature vector as numpy array
    """
    features = [
        float(tfidf_sim),
        float(bert_sim),
        float(skill_overlap.get("jaccard", 0.0)),
        float(skill_overlap.get("overlap_count", 0)),
        float(skill_overlap.get("precision", 0.0)),
        float(skill_overlap.get("recall", 0.0)),
        float(skill_overlap.get("f1", 0.0)),
        float(section_sims.get("skills", 0.0)),
        float(section_sims.get("experience", 0.0)),
        float(section_sims.get("education", 0.0)),
        float(section_sims.get("certifications", 0.0)),
        float(resume_meta.get("resume_length", 0)),
        float(resume_meta.get("num_candidate_skills", 0)),
        float(resume_meta.get("num_job_skills", 0)),
        float(resume_meta.get("years_experience_est", 0)),
    ]
    return np.array(features, dtype=np.float32)


class LogisticRegressionModel:
    """Logistic Regression for binary/ranked resume classification."""

    def __init__(self, params: Dict = None):
        self.params = params or config.LOGISTIC_REGRESSION_PARAMS
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**self.params)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        """Train the logistic regression model."""
        log.info(f"Training Logistic Regression on {X.shape[0]} samples")
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_fitted = True

        # Cross-validation score
        cv_scores = cross_val_score(self.pipeline, X, y, cv=min(5, len(y)), scoring="accuracy")
        log.info(f"LR CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of being a good match."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        proba = self.pipeline.predict_proba(X)
        # Return probability of positive class
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, -1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.pipeline.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.pipeline.score(X, y)

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get feature importance from logistic regression coefficients."""
        if not self.is_fitted:
            return []
        clf = self.pipeline.named_steps["clf"]
        coefs = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
        importance = list(zip(self.feature_names, coefs.tolist()))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return importance

    def save(self, path: Path = None) -> None:
        path = path or config.LR_MODEL_PATH
        save_joblib({"pipeline": self.pipeline, "is_fitted": self.is_fitted}, path)

    def load(self, path: Path = None) -> "LogisticRegressionModel":
        path = path or config.LR_MODEL_PATH
        data = load_joblib(path)
        self.pipeline = data["pipeline"]
        self.is_fitted = data["is_fitted"]
        return self


class RandomForestModel:
    """Random Forest for resume ranking with built-in feature importance."""

    def __init__(self, params: Dict = None):
        self.params = params or config.RANDOM_FOREST_PARAMS
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**self.params)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train the random forest model."""
        log.info(f"Training Random Forest on {X.shape[0]} samples")
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_fitted = True

        cv_scores = cross_val_score(self.pipeline, X, y, cv=min(5, len(y)), scoring="accuracy")
        log.info(f"RF CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        proba = self.pipeline.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, -1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.pipeline.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.pipeline.score(X, y)

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get Gini-based feature importance from Random Forest."""
        if not self.is_fitted:
            return []
        clf = self.pipeline.named_steps["clf"]
        importance = list(zip(self.feature_names, clf.feature_importances_.tolist()))
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance

    def save(self, path: Path = None) -> None:
        path = path or config.RF_MODEL_PATH
        save_joblib({"pipeline": self.pipeline, "is_fitted": self.is_fitted}, path)

    def load(self, path: Path = None) -> "RandomForestModel":
        path = path or config.RF_MODEL_PATH
        data = load_joblib(path)
        self.pipeline = data["pipeline"]
        self.is_fitted = data["is_fitted"]
        return self


def get_model(model_name: str) -> "LogisticRegressionModel | RandomForestModel":
    """Factory function to get ML model by name."""
    if model_name == "logistic_regression":
        return LogisticRegressionModel()
    elif model_name == "random_forest":
        return RandomForestModel()
    else:
        raise ValueError(f"Unknown ML model: {model_name}. Choose 'logistic_regression' or 'random_forest'")
