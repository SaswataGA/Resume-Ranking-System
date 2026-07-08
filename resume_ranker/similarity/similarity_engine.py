"""
Similarity Engine: unified interface for all similarity models.
Computes TF-IDF, BERT, and section-based similarities.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from models.tfidf_model import TFIDFModel
from models.bert_model import BERTSimilarityModel
from preprocessing.text_cleaner import preprocess_for_tfidf, preprocess_for_bert
from utils.logger import log
import config


class SimilarityEngine:
    """
    Unified interface for computing document similarity.
    Manages TF-IDF and BERT models.
    """

    def __init__(self):
        self._tfidf: Optional[TFIDFModel] = None
        self._bert: Optional[BERTSimilarityModel] = None

    @property
    def tfidf(self) -> TFIDFModel:
        if self._tfidf is None:
            self._tfidf = TFIDFModel()
        return self._tfidf

    @property
    def bert(self) -> BERTSimilarityModel:
        if self._bert is None:
            self._bert = BERTSimilarityModel()
        return self._bert

    def compute_tfidf_similarity(
        self,
        job_text: str,
        resume_texts: List[str],
    ) -> np.ndarray:
        """Compute TF-IDF cosine similarity scores."""
        preprocessed_job = preprocess_for_tfidf(job_text)
        preprocessed_resumes = [preprocess_for_tfidf(r) for r in resume_texts]
        scores = self.tfidf.score(preprocessed_job, preprocessed_resumes)
        return scores

    def compute_bert_similarity(
        self,
        job_text: str,
        resume_texts: List[str],
    ) -> np.ndarray:
        """Compute BERT semantic similarity scores."""
        cleaned_job = preprocess_for_bert(job_text)
        cleaned_resumes = [preprocess_for_bert(r) for r in resume_texts]
        scores = self.bert.score(cleaned_job, cleaned_resumes)
        return scores

    def compute_section_similarity(
        self,
        job_sections: Dict[str, str],
        resume_sections_list: List[Dict[str, str]],
        weights: Dict[str, float] = None,
    ) -> np.ndarray:
        """
        Compute weighted section-by-section BERT similarity.
        
        Returns:
            Array of section-weighted scores per resume
        """
        weights = weights or config.SECTION_WEIGHTS
        scores = []

        for resume_sections in resume_sections_list:
            section_score = 0.0
            total_weight = 0.0

            for section, weight in weights.items():
                job_sec = preprocess_for_bert(job_sections.get(section, ""))
                resume_sec = preprocess_for_bert(resume_sections.get(section, ""))

                if not job_sec.strip() or not resume_sec.strip():
                    continue

                sim = self.bert.score(job_sec, [resume_sec])[0]
                section_score += sim * weight
                total_weight += weight

            scores.append(section_score / total_weight if total_weight > 0 else 0.0)

        return np.array(scores)

    def compute_per_section_scores(
        self,
        job_sections: Dict[str, str],
        resume_sections: Dict[str, str],
    ) -> Dict[str, float]:
        """Compute similarity per section for a single resume."""
        result = {}
        for section in config.SECTION_WEIGHTS:
            job_sec = preprocess_for_bert(job_sections.get(section, ""))
            resume_sec = preprocess_for_bert(resume_sections.get(section, ""))

            if not job_sec.strip() or not resume_sec.strip():
                result[section] = 0.0
                continue

            sim = self.bert.score(job_sec, [resume_sec])[0]
            result[section] = float(sim)

        return result


# Module-level singleton
_engine: Optional[SimilarityEngine] = None


def get_similarity_engine() -> SimilarityEngine:
    """Get or create singleton SimilarityEngine."""
    global _engine
    if _engine is None:
        _engine = SimilarityEngine()
    return _engine
