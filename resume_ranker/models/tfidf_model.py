"""
TF-IDF + Cosine Similarity baseline model for resume ranking.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import save_joblib, load_joblib
from utils.logger import log
import config


class TFIDFModel:
    """
    TF-IDF vectorizer with cosine similarity scoring.
    Baseline model for resume-job description matching.
    """

    def __init__(
        self,
        max_features: int = None,
        ngram_range: Tuple[int, int] = None,
        min_df: int = None,
    ):
        self.max_features = max_features or config.TFIDF_MAX_FEATURES
        self.ngram_range = ngram_range or config.TFIDF_NGRAM_RANGE
        self.min_df = min_df or config.TFIDF_MIN_DF
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_fitted = False

    def fit(self, corpus: List[str]) -> "TFIDFModel":
        """Fit TF-IDF on a corpus of documents."""
        log.info(f"Fitting TF-IDF on {len(corpus)} documents")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            analyzer="word",
            strip_accents="unicode",
        )
        self.vectorizer.fit(corpus)
        self.is_fitted = True
        log.info(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF matrix."""
        if not self.is_fitted:
            raise RuntimeError("TFIDFModel must be fitted before transform")
        return self.vectorizer.transform(texts)

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(corpus)
        return self.transform(corpus)

    def score(self, job_text: str, resume_texts: List[str]) -> np.ndarray:
        """
        Compute cosine similarity between job description and each resume.
        
        Args:
            job_text: Preprocessed job description
            resume_texts: List of preprocessed resume texts
            
        Returns:
            Array of similarity scores [0, 1]
        """
        all_texts = [job_text] + resume_texts

        if not self.is_fitted:
            self.fit(all_texts)

        vectors = self.transform(all_texts)
        job_vector = vectors[0]
        resume_vectors = vectors[1:]

        similarities = cosine_similarity(job_vector, resume_vectors).flatten()
        log.debug(f"TF-IDF scores: min={similarities.min():.4f}, max={similarities.max():.4f}")
        return similarities

    def get_feature_importance(self, text: str) -> List[Tuple[str, float]]:
        """Get TF-IDF feature scores for a given text."""
        if not self.is_fitted:
            return []
        vector = self.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = vector.toarray().flatten()
        importance = [(feature_names[i], scores[i]) for i in np.argsort(scores)[::-1] if scores[i] > 0]
        return importance[:50]  # top 50

    def save(self, path: Path = None) -> None:
        path = path or config.TFIDF_MODEL_PATH
        save_joblib({"vectorizer": self.vectorizer, "is_fitted": self.is_fitted}, path)

    def load(self, path: Path = None) -> "TFIDFModel":
        path = path or config.TFIDF_MODEL_PATH
        data = load_joblib(path)
        self.vectorizer = data["vectorizer"]
        self.is_fitted = data["is_fitted"]
        return self

    @classmethod
    def load_from_file(cls, path: Path = None) -> "TFIDFModel":
        model = cls()
        return model.load(path)
