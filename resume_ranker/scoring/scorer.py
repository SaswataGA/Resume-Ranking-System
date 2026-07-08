"""
Resume Scoring Engine.
Combines TF-IDF, BERT, and skill overlap scores with configurable weights.
Produces the final ranking score for each resume.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprocessing.text_cleaner import preprocess_for_tfidf, preprocess_for_bert
from preprocessing.section_extractor import extract_sections
from preprocessing.pdf_parser import parse_resume_file
from skill_extraction.skill_extractor import get_skill_extractor
from similarity.similarity_engine import get_similarity_engine
from models.tfidf_model import TFIDFModel
from models.bert_model import BERTSimilarityModel
from utils.helpers import normalize_scores
from utils.logger import log
import config


@dataclass
class ResumeRecord:
    """Represents a parsed resume with all extracted features."""
    resume_id: str
    file_name: str
    raw_text: str
    cleaned_text: str
    sections: Dict[str, str]
    skills: List[str]
    tfidf_score: float = 0.0
    bert_score: float = 0.0
    skill_overlap_score: float = 0.0
    section_weighted_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    skill_overlap_details: Dict = field(default_factory=dict)
    section_scores: Dict[str, float] = field(default_factory=dict)
    feature_vector: Optional[np.ndarray] = None
    years_experience_est: float = 0.0


@dataclass
class JobRecord:
    """Represents a parsed job description."""
    raw_text: str
    cleaned_text: str
    sections: Dict[str, str]
    required_skills: List[str]


def estimate_years_experience(text: str) -> float:
    """Heuristic to estimate years of experience from text."""
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:work\s*)?experience",
        r"(\d+)\+?\s*years?\s*(?:in\s*)?(?:the\s*)?(?:industry|field|domain)",
        r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
    ]
    years = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            try:
                years.append(float(m))
            except ValueError:
                pass
    return max(years) if years else 0.0


def parse_resume(file_path: Path, resume_id: str = None) -> ResumeRecord:
    """Parse a resume file into a ResumeRecord."""
    file_path = Path(file_path)
    rid = resume_id or file_path.stem

    raw_text = parse_resume_file(file_path)
    if not raw_text:
        log.warning(f"Empty text for resume: {file_path.name}")
        raw_text = ""

    cleaned_text = preprocess_for_tfidf(raw_text)
    sections = extract_sections(raw_text)

    extractor = get_skill_extractor()
    skills = extractor.extract_skills(raw_text)
    years = estimate_years_experience(raw_text)

    return ResumeRecord(
        resume_id=rid,
        file_name=file_path.name,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        sections=sections,
        skills=skills,
        years_experience_est=years,
    )


def parse_job_description(text: str) -> JobRecord:
    """Parse job description text into a JobRecord."""
    cleaned = preprocess_for_tfidf(text)
    sections = extract_sections(text)

    extractor = get_skill_extractor()
    required_skills = extractor.extract_skills(text)

    return JobRecord(
        raw_text=text,
        cleaned_text=cleaned,
        sections=sections,
        required_skills=required_skills,
    )


class ResumeScorer:
    """
    Core scoring engine that ranks resumes against a job description.
    
    Supports model selection: tfidf, bert, logistic_regression, random_forest
    Combines multiple signals with configurable weights.
    """

    def __init__(
        self,
        model_name: str = None,
        score_weights: Dict[str, float] = None,
        section_weights: Dict[str, float] = None,
    ):
        self.model_name = model_name or config.ACTIVE_MODEL
        self.score_weights = score_weights or config.SCORE_COMBINATION_WEIGHTS
        self.section_weights = section_weights or config.SECTION_WEIGHTS

        self._similarity_engine = get_similarity_engine()
        self._skill_extractor = get_skill_extractor()
        self._ml_model = None

        log.info(f"ResumeScorer initialized with model: {self.model_name}")

    def _load_ml_model(self, model_name: str):
        """Load ML model (lazy)."""
        from models.ml_models import get_model
        if self._ml_model is None:
            model_path = config.LR_MODEL_PATH if model_name == "logistic_regression" else config.RF_MODEL_PATH
            model = get_model(model_name)
            if model_path.exists():
                model.load(model_path)
                log.info(f"Loaded saved {model_name} model")
            else:
                log.warning(f"No saved {model_name} model found. Model will score using raw features.")
            self._ml_model = model
        return self._ml_model

    def score_resumes(
        self,
        job: JobRecord,
        resumes: List[ResumeRecord],
    ) -> List[ResumeRecord]:
        """
        Score and rank all resumes against the job description.
        
        Args:
            job: Parsed job description
            resumes: List of parsed resumes
            
        Returns:
            List of resumes with scores filled in, sorted by final_score descending
        """
        if not resumes:
            return []

        log.info(f"Scoring {len(resumes)} resumes using model: {self.model_name}")

        # Step 1: Compute TF-IDF similarities
        resume_texts_tfidf = [r.cleaned_text for r in resumes]
        tfidf_scores = self._similarity_engine.compute_tfidf_similarity(
            job.cleaned_text, resume_texts_tfidf
        )
        tfidf_scores = normalize_scores(tfidf_scores)

        # Step 2: Compute BERT similarities (or skip if tfidf-only)
        if self.model_name != "tfidf":
            resume_texts_bert = [preprocess_for_bert(r.raw_text) for r in resumes]
            bert_scores = self._similarity_engine.compute_bert_similarity(
                preprocess_for_bert(job.raw_text), resume_texts_bert
            )
        else:
            bert_scores = tfidf_scores.copy()

        # Step 3: Compute skill overlap scores
        skill_overlaps = []
        for resume in resumes:
            overlap = self._skill_extractor.skill_overlap(
                resume.skills, job.required_skills
            )
            skill_overlaps.append(overlap)

        skill_scores = np.array([o.get("f1", 0.0) for o in skill_overlaps])

        # Step 4: Compute section-weighted scores
        job_sections = job.sections
        section_scores_list = []
        per_section_scores = []
        for resume in resumes:
            sec_scores = self._similarity_engine.compute_per_section_scores(
                job_sections, resume.sections
            )
            per_section_scores.append(sec_scores)
            weighted = sum(
                sec_scores.get(sec, 0.0) * w
                for sec, w in self.section_weights.items()
            )
            denom = sum(
                w for sec, w in self.section_weights.items()
                if sec_scores.get(sec, 0.0) > 0 or job_sections.get(sec, "").strip()
            )
            section_scores_list.append(weighted / denom if denom > 0 else 0.0)

        section_score_arr = np.array(section_scores_list)

        # Step 5: Compute ML model scores if applicable
        ml_scores = None
        if self.model_name in {"logistic_regression", "random_forest"}:
            ml_scores = self._compute_ml_scores(
                resumes, job, tfidf_scores, bert_scores, skill_overlaps, per_section_scores
            )

        # Step 6: Combine scores
        final_scores = self._combine_scores(
            tfidf_scores, bert_scores, skill_scores, section_score_arr, ml_scores
        )

        # Step 7: Fill results into ResumeRecord objects
        for i, resume in enumerate(resumes):
            resume.tfidf_score = float(tfidf_scores[i])
            resume.bert_score = float(bert_scores[i])
            resume.skill_overlap_score = float(skill_scores[i])
            resume.section_weighted_score = float(section_score_arr[i])
            resume.final_score = float(final_scores[i])
            resume.skill_overlap_details = skill_overlaps[i]
            resume.section_scores = per_section_scores[i]

        # Step 8: Sort and assign ranks
        resumes.sort(key=lambda r: r.final_score, reverse=True)
        for rank, resume in enumerate(resumes, start=1):
            resume.rank = rank

        log.info(f"Top resume: {resumes[0].file_name} (score: {resumes[0].final_score:.4f})")
        return resumes

    def _compute_ml_scores(
        self,
        resumes: List[ResumeRecord],
        job: JobRecord,
        tfidf_scores: np.ndarray,
        bert_scores: np.ndarray,
        skill_overlaps: List[Dict],
        per_section_scores: List[Dict],
    ) -> np.ndarray:
        """Build feature vectors and score with ML model."""
        from models.ml_models import build_feature_vector

        model = self._load_ml_model(self.model_name)

        feature_vectors = []
        for i, resume in enumerate(resumes):
            fv = build_feature_vector(
                tfidf_sim=float(tfidf_scores[i]),
                bert_sim=float(bert_scores[i]),
                skill_overlap=skill_overlaps[i],
                section_sims=per_section_scores[i],
                resume_meta={
                    "resume_length": len(resume.raw_text),
                    "num_candidate_skills": len(resume.skills),
                    "num_job_skills": len(job.required_skills),
                    "years_experience_est": resume.years_experience_est,
                },
            )
            feature_vectors.append(fv)
            resume.feature_vector = fv

        X = np.stack(feature_vectors)

        if model.is_fitted:
            ml_scores = model.predict_proba(X)
        else:
            # If not fitted, use weighted combination of base features as proxy
            log.warning("ML model not fitted, using base feature combination as proxy")
            ml_scores = (
                0.4 * tfidf_scores
                + 0.4 * bert_scores
                + 0.2 * np.array([o.get("f1", 0.0) for o in skill_overlaps])
            )

        return normalize_scores(ml_scores)

    def _combine_scores(
        self,
        tfidf_scores: np.ndarray,
        bert_scores: np.ndarray,
        skill_scores: np.ndarray,
        section_scores: np.ndarray,
        ml_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combine multiple score signals into a final score."""
        w = self.score_weights

        if self.model_name == "tfidf":
            return (
                w.get("semantic_similarity", 0.45) * tfidf_scores
                + w.get("skill_overlap", 0.40) * skill_scores
                + w.get("section_weighted", 0.15) * section_scores
            )
        elif self.model_name == "bert":
            return (
                w.get("semantic_similarity", 0.45) * bert_scores
                + w.get("skill_overlap", 0.40) * skill_scores
                + w.get("section_weighted", 0.15) * section_scores
            )
        elif ml_scores is not None:
            # For ML models, blend ML score with semantic signals
            return (
                0.40 * ml_scores
                + 0.25 * bert_scores
                + 0.25 * skill_scores
                + 0.10 * section_scores
            )
        else:
            return (
                w.get("semantic_similarity", 0.45) * bert_scores
                + w.get("skill_overlap", 0.40) * skill_scores
                + w.get("section_weighted", 0.15) * section_scores
            )

    def to_dataframe(self, resumes: List[ResumeRecord]) -> pd.DataFrame:
        """Convert scored resumes to a ranked DataFrame."""
        rows = []
        for r in resumes:
            rows.append({
                "rank": r.rank,
                "resume_id": r.resume_id,
                "file_name": r.file_name,
                "final_score": round(r.final_score, 4),
                "tfidf_score": round(r.tfidf_score, 4),
                "bert_score": round(r.bert_score, 4),
                "skill_overlap_score": round(r.skill_overlap_score, 4),
                "section_weighted_score": round(r.section_weighted_score, 4),
                "matched_skills": ", ".join(r.skill_overlap_details.get("matched", [])),
                "missing_skills": ", ".join(r.skill_overlap_details.get("missing", [])),
                "num_skills": len(r.skills),
                "years_experience": r.years_experience_est,
                "skills": ", ".join(r.skills),
            })
        return pd.DataFrame(rows)
