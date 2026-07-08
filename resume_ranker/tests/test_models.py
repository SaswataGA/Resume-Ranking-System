"""Unit tests for model outputs and save/load functionality."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTFIDFSaveLoad:
    def test_save_and_load_tfidf(self, tmp_path):
        from models.tfidf_model import TFIDFModel
        corpus = [
            "python machine learning data science",
            "java spring boot web development",
            "python deep learning tensorflow pytorch",
        ]
        model = TFIDFModel(max_features=100)
        model.fit(corpus)

        save_path = tmp_path / "tfidf.joblib"
        model.save(save_path)

        loaded = TFIDFModel.load_from_file(save_path)
        assert loaded.is_fitted

        # Results must be identical after reload
        s1 = model.score("python machine learning", ["deep learning pytorch"])
        s2 = loaded.score("python machine learning", ["deep learning pytorch"])
        np.testing.assert_allclose(s1, s2, rtol=1e-5)

    def test_tfidf_score_range(self):
        from models.tfidf_model import TFIDFModel
        model = TFIDFModel()
        job = "python machine learning engineer"
        resumes = [
            "python developer scikit-learn pandas",
            "java spring boot angular",
            "python tensorflow deep learning",
        ]
        scores = model.score(job, resumes)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_tfidf_feature_importance(self):
        from models.tfidf_model import TFIDFModel
        model = TFIDFModel()
        model.fit(["python machine learning", "java web development"])
        importance = model.get_feature_importance("python machine learning")
        assert isinstance(importance, list)
        # Each item is (feature_name, score)
        for name, score in importance:
            assert isinstance(name, str)
            assert isinstance(score, float)
            assert score >= 0


class TestMLModelSaveLoad:
    def test_lr_save_load(self, tmp_path):
        from models.ml_models import LogisticRegressionModel
        X = np.random.randn(40, 15)
        y = np.array([0] * 20 + [1] * 20)

        model = LogisticRegressionModel()
        model.fit(X, y)

        save_path = tmp_path / "lr.joblib"
        model.save(save_path)

        loaded = LogisticRegressionModel()
        loaded.load(save_path)
        assert loaded.is_fitted

        p1 = model.predict_proba(X)
        p2 = loaded.predict_proba(X)
        np.testing.assert_allclose(p1, p2, rtol=1e-5)

    def test_rf_save_load(self, tmp_path):
        from models.ml_models import RandomForestModel
        X = np.random.randn(40, 15)
        y = np.array([0] * 20 + [1] * 20)

        model = RandomForestModel()
        model.fit(X, y)

        save_path = tmp_path / "rf.joblib"
        model.save(save_path)

        loaded = RandomForestModel()
        loaded.load(save_path)
        assert loaded.is_fitted

        p1 = model.predict_proba(X)
        p2 = loaded.predict_proba(X)
        np.testing.assert_allclose(p1, p2, rtol=1e-5)

    def test_lr_predict_proba_range(self):
        from models.ml_models import LogisticRegressionModel
        X = np.random.randn(20, 15)
        y = np.array([0] * 10 + [1] * 10)
        model = LogisticRegressionModel()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_rf_predict_proba_range(self):
        from models.ml_models import RandomForestModel
        X = np.random.randn(20, 15)
        y = np.array([0] * 10 + [1] * 10)
        model = RandomForestModel()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_get_model_factory(self):
        from models.ml_models import get_model, LogisticRegressionModel, RandomForestModel
        lr = get_model("logistic_regression")
        assert isinstance(lr, LogisticRegressionModel)

        rf = get_model("random_forest")
        assert isinstance(rf, RandomForestModel)

        with pytest.raises(ValueError):
            get_model("invalid_model")

    def test_unfitted_model_raises(self):
        from models.ml_models import LogisticRegressionModel
        import numpy as np
        model = LogisticRegressionModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.array([[0.5] * 15]))


class TestSimilarityEngine:
    def test_tfidf_similarity_returns_array(self):
        from similarity.similarity_engine import SimilarityEngine
        engine = SimilarityEngine()
        scores = engine.compute_tfidf_similarity(
            "python machine learning",
            ["python developer", "java engineer", "ml scientist"],
        )
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_tfidf_similarity_relevant_higher(self):
        from similarity.similarity_engine import SimilarityEngine
        engine = SimilarityEngine()
        job = "python machine learning scikit-learn pandas numpy"
        resumes = [
            "python data scientist sklearn pandas numpy tensorflow",
            "javascript react frontend developer html css",
        ]
        scores = engine.compute_tfidf_similarity(job, resumes)
        assert scores[0] > scores[1], "Python ML resume should score higher than JS frontend"

    def test_per_section_scores(self):
        from similarity.similarity_engine import SimilarityEngine
        engine = SimilarityEngine()
        job_sections = {
            "skills": "Python machine learning TensorFlow",
            "experience": "5 years ML engineering",
            "education": "MS Computer Science",
            "certifications": "",
        }
        resume_sections = {
            "skills": "Python scikit-learn machine learning",
            "experience": "4 years data science",
            "education": "BS Computer Science",
            "certifications": "",
        }
        result = engine.compute_per_section_scores(job_sections, resume_sections)
        assert isinstance(result, dict)
        for section in ["skills", "experience", "education"]:
            assert section in result
            assert 0.0 <= result[section] <= 1.0


class TestDataGenerator:
    def test_generate_training_features(self):
        from data.sample.data_generator import generate_training_features
        X, y = generate_training_features()
        assert X.ndim == 2
        assert X.shape[1] == 15
        assert len(y) == X.shape[0]
        assert set(np.unique(y)) == {0, 1}

    def test_class_balance(self):
        from data.sample.data_generator import generate_training_features
        X, y = generate_training_features()
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        # Should have both positive and negative samples
        assert pos_count > 0
        assert neg_count > 0
