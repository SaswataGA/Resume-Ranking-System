"""
Shared pytest fixtures for the Resume Ranking System test suite.
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Sample Texts ─────────────────────────────────────────────────────────────

SAMPLE_JOB_TEXT = """
Senior Machine Learning Engineer

We are seeking a Python machine learning engineer with 5+ years of experience.
Required skills: Python, TensorFlow, PyTorch, scikit-learn, AWS, Docker, Kubernetes,
SQL, MongoDB, Apache Spark. Strong background in NLP, deep learning, and MLOps.
Experience with BERT, Airflow, and Kafka is a plus.
Bachelor or Master degree in Computer Science required.
AWS Certified Machine Learning Specialty preferred.
"""

SAMPLE_RESUME_STRONG = """
Alice Johnson
alice@example.com

SKILLS
Python, TensorFlow, PyTorch, scikit-learn, AWS, Docker, Kubernetes, SQL,
MongoDB, Apache Spark, NLP, machine learning, BERT, Airflow, Kafka, MLflow

EXPERIENCE
Senior ML Engineer | TechCorp | 2019 - Present
- Built production NLP pipelines using BERT and PyTorch on AWS
- Deployed models with Docker and Kubernetes
- Managed data pipelines with Kafka and Airflow
- 6 years of experience in machine learning engineering

EDUCATION
MS Computer Science, Stanford University, 2018

CERTIFICATIONS
AWS Certified Machine Learning Specialty (2022)
"""

SAMPLE_RESUME_WEAK = """
David Lee
david@example.com

SKILLS
HTML, CSS, JavaScript, jQuery, Photoshop

EXPERIENCE
Web Designer | Small Agency | 2022 - Present
- Designed website layouts with HTML and CSS
- Created graphics in Photoshop

EDUCATION
BS Graphic Design, 2022
"""

SAMPLE_RESUME_MEDIUM = """
Carol Williams
carol@example.com

SKILLS
Python, scikit-learn, pandas, NumPy, SQL, Tableau, Excel

EXPERIENCE
Data Analyst | Retail Corp | 2020 - Present
- Built forecasting models with Python and scikit-learn
- Analyzed data with SQL and pandas
- Created dashboards with Tableau

EDUCATION
BS Statistics, 2020
"""


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_job_text():
    return SAMPLE_JOB_TEXT


@pytest.fixture(scope="session")
def sample_resume_strong():
    return SAMPLE_RESUME_STRONG


@pytest.fixture(scope="session")
def sample_resume_weak():
    return SAMPLE_RESUME_WEAK


@pytest.fixture(scope="session")
def sample_resume_medium():
    return SAMPLE_RESUME_MEDIUM


@pytest.fixture
def tmp_resume_files(tmp_path):
    """Create temporary .txt resume files for testing."""
    r_strong = tmp_path / "resume_strong.txt"
    r_medium = tmp_path / "resume_medium.txt"
    r_weak   = tmp_path / "resume_weak.txt"
    r_strong.write_text(SAMPLE_RESUME_STRONG, encoding="utf-8")
    r_medium.write_text(SAMPLE_RESUME_MEDIUM, encoding="utf-8")
    r_weak.write_text(SAMPLE_RESUME_WEAK,   encoding="utf-8")
    return [r_strong, r_medium, r_weak]


@pytest.fixture
def tmp_job_file(tmp_path):
    """Create a temporary job description file."""
    jd = tmp_path / "job_description.txt"
    jd.write_text(SAMPLE_JOB_TEXT, encoding="utf-8")
    return jd


@pytest.fixture(scope="session")
def parsed_job():
    from scoring.scorer import parse_job_description
    return parse_job_description(SAMPLE_JOB_TEXT)


@pytest.fixture(scope="session")
def parsed_resumes(tmp_path_factory):
    from scoring.scorer import parse_resume
    tmp = tmp_path_factory.mktemp("resumes")
    paths = []
    for name, text in [
        ("strong.txt", SAMPLE_RESUME_STRONG),
        ("medium.txt", SAMPLE_RESUME_MEDIUM),
        ("weak.txt",   SAMPLE_RESUME_WEAK),
    ]:
        p = tmp / name
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return [parse_resume(p) for p in paths]


@pytest.fixture(scope="session")
def synthetic_feature_matrix():
    """Return a fixed synthetic feature matrix (X, y) for ML model tests."""
    np.random.seed(42)
    n_pos, n_neg, n_feat = 60, 90, 15
    X_pos = np.clip(np.random.randn(n_pos, n_feat) * 0.3 + 0.7, 0, 1)
    X_neg = np.clip(np.random.randn(n_neg, n_feat) * 0.3 + 0.3, 0, 1)
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


@pytest.fixture
def ground_truth_labels():
    return {
        "ranked_ids":   ["resume_strong.txt", "resume_medium.txt", "resume_weak.txt"],
        "relevant_ids": ["resume_strong.txt", "resume_medium.txt"],
    }
