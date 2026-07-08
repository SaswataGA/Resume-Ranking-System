"""Integration tests for the FastAPI endpoints."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_JOB = """
Senior Machine Learning Engineer

We are seeking a Python machine learning engineer with experience in
TensorFlow, PyTorch, AWS, Docker, and Kubernetes. Strong skills in NLP,
deep learning, and data engineering required. SQL, MongoDB, and Spark a plus.
"""

SAMPLE_RESUME = """
Alice Johnson
alice@example.com

SKILLS
Python, TensorFlow, PyTorch, AWS, Docker, Kubernetes, NLP, Machine Learning, SQL, MongoDB

EXPERIENCE
Senior ML Engineer | TechCorp | 2020-2023
- Built deep learning models with TensorFlow and PyTorch
- Deployed on AWS using Docker and Kubernetes

EDUCATION
MS Computer Science, Stanford, 2020
"""

SAMPLE_RESUME_WEAK = """
Bob Builder
bob@example.com

SKILLS
HTML, CSS, JavaScript, Photoshop

EXPERIENCE
Web Designer | Creative Agency | 2019-2023
- Designed websites and graphics

EDUCATION
BS Art Design, 2019
"""


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_models_list(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "tfidf" in data["available_models"]
        assert "bert" in data["available_models"]

    def test_config_endpoint(self, client):
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "active_model" in data
        assert "section_weights" in data


class TestRankEndpoint:
    def test_rank_text_endpoint(self, client):
        import json
        resume_list = [
            {"id": "alice", "text": SAMPLE_RESUME},
            {"id": "bob", "text": SAMPLE_RESUME_WEAK},
        ]
        response = client.post(
            "/rank/text",
            data={
                "job_description": SAMPLE_JOB,
                "resume_texts": json.dumps(resume_list),
                "model_name": "tfidf",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "rankings" in data
        assert len(data["rankings"]) == 2
        # Alice (ML) should rank above Bob (web designer)
        rankings = data["rankings"]
        alice_rank = next(r["rank"] for r in rankings if r["resume_id"] == "alice")
        bob_rank = next(r["rank"] for r in rankings if r["resume_id"] == "bob")
        assert alice_rank < bob_rank

    def test_rank_text_invalid_json(self, client):
        response = client.post(
            "/rank/text",
            data={
                "job_description": SAMPLE_JOB,
                "resume_texts": "not valid json",
                "model_name": "tfidf",
            },
        )
        assert response.status_code == 400

    def test_rank_text_empty_job(self, client):
        import json
        response = client.post(
            "/rank/text",
            data={
                "job_description": "   ",
                "resume_texts": json.dumps([{"id": "r1", "text": "some resume"}]),
                "model_name": "tfidf",
            },
        )
        assert response.status_code == 400

    def test_rank_with_file_upload(self, client, tmp_path):
        # Create temp resume files
        r1 = tmp_path / "alice.txt"
        r1.write_text(SAMPLE_RESUME)
        r2 = tmp_path / "bob.txt"
        r2.write_text(SAMPLE_RESUME_WEAK)

        with open(r1, "rb") as f1, open(r2, "rb") as f2:
            response = client.post(
                "/rank",
                data={"job_description": SAMPLE_JOB, "model_name": "tfidf"},
                files=[
                    ("resumes", ("alice.txt", f1, "text/plain")),
                    ("resumes", ("bob.txt", f2, "text/plain")),
                ],
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["rankings"]) == 2

    def test_rank_invalid_model(self, client):
        import json
        response = client.post(
            "/rank/text",
            data={
                "job_description": SAMPLE_JOB,
                "resume_texts": json.dumps([{"id": "r1", "text": SAMPLE_RESUME}]),
                "model_name": "invalid_model_xyz",
            },
        )
        assert response.status_code == 400

    def test_rank_response_structure(self, client):
        import json
        response = client.post(
            "/rank/text",
            data={
                "job_description": SAMPLE_JOB,
                "resume_texts": json.dumps([{"id": "r1", "text": SAMPLE_RESUME}]),
                "model_name": "tfidf",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "rankings" in data
        ranking = data["rankings"][0]
        required_fields = ["rank", "resume_id", "final_score"]
        for field in required_fields:
            assert field in ranking


class TestExplainEndpoint:
    def test_explain_endpoint(self, client, tmp_path):
        r1 = tmp_path / "resume.txt"
        r1.write_text(SAMPLE_RESUME)

        with open(r1, "rb") as f:
            response = client.post(
                "/explain",
                data={"job_description": SAMPLE_JOB, "model_name": "tfidf"},
                files=[("resume", ("resume.txt", f, "text/plain"))],
            )

        assert response.status_code == 200
        data = response.json()
        assert "resume_id" in data
        assert "final_score" in data
        assert "skill_analysis" in data
        assert "narrative" in data
        assert 0.0 <= data["final_score"] <= 1.0
