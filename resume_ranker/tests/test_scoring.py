"""Unit tests for scoring module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSkillExtractor:
    def test_extract_skills_basic(self):
        from skill_extraction.skill_extractor import get_skill_extractor
        extractor = get_skill_extractor()
        text = "I have experience with Python, machine learning, and AWS"
        skills = extractor.extract_skills(text)
        assert isinstance(skills, list)
        assert "python" in skills
        assert "machine learning" in skills
        assert "aws" in skills

    def test_extract_skills_aliases(self):
        from skill_extraction.skill_extractor import get_skill_extractor
        extractor = get_skill_extractor()
        # Test alias normalization
        text = "Expert in Py programming and sklearn models"
        skills = extractor.extract_skills(text)
        # "Py" should map to "python", "sklearn" to "scikit-learn"
        assert "python" in skills
        assert "scikit-learn" in skills

    def test_extract_skills_empty_text(self):
        from skill_extraction.skill_extractor import get_skill_extractor
        extractor = get_skill_extractor()
        skills = extractor.extract_skills("")
        assert skills == []

    def test_skill_overlap_perfect(self):
        from skill_extraction.skill_extractor import get_skill_extractor
        extractor = get_skill_extractor()
        skills_a = ["python", "machine learning", "sql"]
        skills_b = ["python", "machine learning", "sql"]
        result = extractor.skill_overlap(skills_a, skills_b)
        assert result["jaccard"] == 1.0
        assert result["f1"] == 1.0
        assert set(result["matched"]) == {"python", "machine learning", "sql"}

    def test_skill_overlap_no_match(self):
        from skill_extraction.skill_extractor import get_skill_extractor
        extractor = get_skill_extractor()
        skills_a = ["python", "machine learning"]
        skills_b = ["java", "spring"]
        result = extractor.skill_overlap(skills_a, skills_b)
        assert result["jaccard"] == 0.0
        assert result["overlap_count"] == 0
        assert result["matched"] == []

    def test_skill_overlap_partial(self):
        from skill_extraction.skill_extractor import get_skill_extractor
        extractor = get_skill_extractor()
        skills_a = ["python", "machine learning", "sql"]
        skills_b = ["python", "java", "sql"]
        result = extractor.skill_overlap(skills_a, skills_b)
        assert 0 < result["jaccard"] < 1
        assert result["overlap_count"] == 2
        assert "python" in result["matched"]
        assert "sql" in result["matched"]

    def test_skill_overlap_empty(self):
        from skill_extraction.skill_extractor import get_skill_extractor
        extractor = get_skill_extractor()
        result = extractor.skill_overlap([], [])
        assert result["jaccard"] == 0.0
        assert result["overlap_count"] == 0

    def test_normalize_skill(self):
        from skill_extraction.skill_extractor import SkillOntology
        ontology = SkillOntology()
        assert ontology.normalize("sklearn") == "scikit-learn"
        assert ontology.normalize("py") == "python"
        assert ontology.normalize("nodejs") == "javascript"
        assert ontology.normalize("k8s") == "kubernetes"


class TestTFIDFModel:
    def test_fit_and_score(self):
        from models.tfidf_model import TFIDFModel
        model = TFIDFModel()
        texts = [
            "Python machine learning data science",
            "Java spring boot backend development",
            "Python deep learning neural networks",
        ]
        scores = model.score(texts[0], texts[1:])
        assert len(scores) == 2
        assert all(0 <= s <= 1 for s in scores)
        # Python/ML text should be more similar to Python/DL text
        assert scores[1] > scores[0]

    def test_fit_transform(self):
        from models.tfidf_model import TFIDFModel
        model = TFIDFModel()
        corpus = ["python machine learning", "java web development", "python data science"]
        matrix = model.fit_transform(corpus)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 10000

    def test_tfidf_save_load(self, tmp_path):
        from models.tfidf_model import TFIDFModel
        model = TFIDFModel()
        corpus = ["python machine learning", "java web development"]
        model.fit(corpus)

        save_path = tmp_path / "tfidf.joblib"
        model.save(save_path)

        loaded = TFIDFModel()
        loaded.load(save_path)
        assert loaded.is_fitted

        scores1 = model.score("python", ["machine learning"])
        scores2 = loaded.score("python", ["machine learning"])
        assert abs(scores1[0] - scores2[0]) < 1e-6


class TestMLModels:
    def test_logistic_regression_fit_predict(self):
        from models.ml_models import LogisticRegressionModel
        import numpy as np
        X = np.random.randn(50, 15)
        y = np.random.randint(0, 2, 50)
        # Ensure both classes present
        y[:25] = 0
        y[25:] = 1

        model = LogisticRegressionModel()
        model.fit(X, y)
        assert model.is_fitted

        proba = model.predict_proba(X)
        assert len(proba) == 50
        assert all(0 <= p <= 1 for p in proba)

    def test_random_forest_fit_predict(self):
        from models.ml_models import RandomForestModel
        import numpy as np
        X = np.random.randn(50, 15)
        y = np.array([0] * 25 + [1] * 25)

        model = RandomForestModel()
        model.fit(X, y)
        assert model.is_fitted

        proba = model.predict_proba(X)
        assert len(proba) == 50

    def test_feature_importance_lr(self):
        from models.ml_models import LogisticRegressionModel
        import numpy as np
        X = np.random.randn(30, 15)
        y = np.array([0] * 15 + [1] * 15)
        model = LogisticRegressionModel()
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert len(importance) == 15
        assert all(isinstance(f, str) and isinstance(v, float) for f, v in importance)

    def test_feature_importance_rf(self):
        from models.ml_models import RandomForestModel
        import numpy as np
        X = np.random.randn(30, 15)
        y = np.array([0] * 15 + [1] * 15)
        model = RandomForestModel()
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert len(importance) == 15
        # RF importances sum to ~1
        total = sum(v for _, v in importance)
        assert abs(total - 1.0) < 0.01

    def test_build_feature_vector(self):
        from models.ml_models import build_feature_vector
        fv = build_feature_vector(
            tfidf_sim=0.7,
            bert_sim=0.8,
            skill_overlap={"jaccard": 0.5, "overlap_count": 3, "precision": 0.6, "recall": 0.4, "f1": 0.48},
            section_sims={"skills": 0.9, "experience": 0.7, "education": 0.5, "certifications": 0.3},
            resume_meta={"resume_length": 1500, "num_candidate_skills": 12, "num_job_skills": 15, "years_experience_est": 5},
        )
        assert fv.shape == (15,)
        assert fv[0] == pytest.approx(0.7)
        assert fv[1] == pytest.approx(0.8)


class TestScorer:
    def test_parse_job_description(self):
        from scoring.scorer import parse_job_description
        text = "Looking for Python developer with machine learning and AWS experience"
        job = parse_job_description(text)
        assert job.raw_text == text
        assert isinstance(job.required_skills, list)
        assert "python" in job.required_skills

    def test_parse_resume_from_file(self, tmp_path):
        from scoring.scorer import parse_resume
        resume_file = tmp_path / "test.txt"
        resume_file.write_text("""
John Doe
SKILLS
Python, Machine Learning, TensorFlow, AWS
EXPERIENCE
ML Engineer at TechCorp 2020-2023: Built ML pipelines
EDUCATION
MS Computer Science, Stanford 2020
""")
        record = parse_resume(resume_file)
        assert record.file_name == "test.txt"
        assert isinstance(record.skills, list)
        assert "python" in record.skills
        assert isinstance(record.sections, dict)

    def test_scorer_tfidf_ranking(self, tmp_path):
        from scoring.scorer import ResumeScorer, parse_job_description, parse_resume

        job_text = "Python machine learning engineer with scikit-learn and pandas"
        job = parse_job_description(job_text)

        r1 = tmp_path / "r1.txt"
        r1.write_text("Python data scientist with scikit-learn, pandas, machine learning experience")
        r2 = tmp_path / "r2.txt"
        r2.write_text("Java web developer Spring Boot REST API development")

        records = [parse_resume(r1), parse_resume(r2)]
        scorer = ResumeScorer(model_name="tfidf")
        scored = scorer.score_resumes(job, records)

        assert len(scored) == 2
        assert scored[0].rank == 1
        assert scored[1].rank == 2
        # Python/ML resume should rank higher
        assert scored[0].final_score >= scored[1].final_score
        # r1 should be top
        assert "r1" in scored[0].file_name

    def test_scorer_to_dataframe(self, tmp_path):
        from scoring.scorer import ResumeScorer, parse_job_description, parse_resume
        import pandas as pd

        job = parse_job_description("Python developer")
        r1 = tmp_path / "r1.txt"
        r1.write_text("Python developer with 3 years experience")
        records = [parse_resume(r1)]

        scorer = ResumeScorer(model_name="tfidf")
        scored = scorer.score_resumes(job, records)
        df = scorer.to_dataframe(scored)

        assert isinstance(df, pd.DataFrame)
        assert "rank" in df.columns
        assert "final_score" in df.columns
        assert "file_name" in df.columns
        assert len(df) == 1

    def test_estimate_years_experience(self):
        from scoring.scorer import estimate_years_experience
        text1 = "I have 5 years of experience in machine learning"
        assert estimate_years_experience(text1) == 5.0

        text2 = "10+ years experience in software engineering"
        assert estimate_years_experience(text2) == 10.0

        text3 = "Recent graduate with no experience"
        assert estimate_years_experience(text3) == 0.0
