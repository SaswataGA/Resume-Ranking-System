"""
FastAPI Application for Resume Ranking System.
Endpoints:
  POST /rank          - Rank resumes against a job description
  POST /explain       - Get explanations for a specific resume
  GET  /models        - List available models
  GET  /health        - Health check
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.scorer import ResumeScorer, parse_job_description, parse_resume
from explainability.explainer_pipeline import ExplainabilityPipeline
from evaluation.metrics import compute_ranking_metrics, format_metrics_report
from evaluation.fairness import run_fairness_analysis
from utils.logger import log
import config

app = FastAPI(
    title="Resume Ranking API",
    description="Skill-based Resume Ranking System with Explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class RankRequest(BaseModel):
    job_description: str
    model_name: Optional[str] = None
    weights: Optional[Dict[str, float]] = None


class RankedResume(BaseModel):
    rank: int
    resume_id: str
    file_name: str
    final_score: float
    tfidf_score: float
    bert_score: float
    skill_overlap_score: float
    section_weighted_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    candidate_skills: List[str]
    narrative: str


class RankResponse(BaseModel):
    status: str
    model_used: str
    processing_time_seconds: float
    num_resumes: int
    rankings: List[RankedResume]
    job_required_skills: List[str]


class ExplainResponse(BaseModel):
    resume_id: str
    rank: int
    final_score: float
    narrative: str
    skill_analysis: Dict
    section_analysis: Dict
    score_breakdown: Dict
    lime_text: Optional[str] = None


# ─── Helper ─────────────────────────────────────────────────────────────────

def _save_upload(upload_file: UploadFile, tmp_dir: str) -> Path:
    """Save uploaded file to temp directory."""
    safe_name = Path(upload_file.filename).name
    dest = Path(tmp_dir) / safe_name
    content = upload_file.file.read()
    dest.write_bytes(content)
    return dest


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0", "active_model": config.ACTIVE_MODEL}


@app.get("/models")
def list_models():
    return {
        "available_models": ["tfidf", "bert", "logistic_regression", "random_forest"],
        "active_model": config.ACTIVE_MODEL,
        "descriptions": {
            "tfidf": "TF-IDF + Cosine Similarity (fast, keyword-based baseline)",
            "bert": "Sentence-BERT semantic similarity (recommended)",
            "logistic_regression": "Logistic Regression on composite feature vector",
            "random_forest": "Random Forest on composite feature vector",
        },
    }


@app.post("/rank", response_model=RankResponse)
async def rank_resumes(
    job_description: str = Form(...),
    model_name: str = Form(default=None),
    resumes: List[UploadFile] = File(...),
):
    """
    Rank uploaded resumes against a job description.

    - **job_description**: Full job description text
    - **model_name**: Model to use (tfidf, bert, logistic_regression, random_forest)
    - **resumes**: Resume files (PDF or TXT)
    """
    start_time = time.time()

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required")

    selected_model = model_name or config.ACTIVE_MODEL
    valid_models = {"tfidf", "bert", "logistic_regression", "random_forest"}
    if selected_model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {valid_models}")

    log.info(f"Ranking {len(resumes)} resumes using model={selected_model}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save uploaded files
        resume_paths = []
        for upload in resumes:
            try:
                path = _save_upload(upload, tmp_dir)
                resume_paths.append(path)
            except Exception as e:
                log.warning(f"Failed to save {upload.filename}: {e}")

        if not resume_paths:
            raise HTTPException(status_code=400, detail="No valid resume files received")

        # Parse job
        job = parse_job_description(job_description)

        # Parse resumes
        resume_records = []
        for path in resume_paths:
            try:
                record = parse_resume(path)
                resume_records.append(record)
            except Exception as e:
                log.warning(f"Failed to parse {path.name}: {e}")

        if not resume_records:
            raise HTTPException(status_code=422, detail="Could not parse any resume files")

        # Score
        scorer = ResumeScorer(model_name=selected_model)
        scored = scorer.score_resumes(job, resume_records)

        # Build explainability for narratives
        xai = ExplainabilityPipeline(model_type=selected_model)
        explanations_map = {}
        for resume in scored:
            exp = xai.explain_resume(resume, job, output_dir=config.PLOTS_DIR)
            explanations_map[resume.resume_id] = exp

    elapsed = time.time() - start_time

    # Build response
    rankings = []
    for r in scored:
        exp = explanations_map.get(r.resume_id, {})
        rankings.append(RankedResume(
            rank=r.rank,
            resume_id=r.resume_id,
            file_name=r.file_name,
            final_score=round(r.final_score, 4),
            tfidf_score=round(r.tfidf_score, 4),
            bert_score=round(r.bert_score, 4),
            skill_overlap_score=round(r.skill_overlap_score, 4),
            section_weighted_score=round(r.section_weighted_score, 4),
            matched_skills=r.skill_overlap_details.get("matched", []),
            missing_skills=r.skill_overlap_details.get("missing", []),
            candidate_skills=r.skills,
            narrative=exp.get("narrative", ""),
        ))

    return RankResponse(
        status="success",
        model_used=selected_model,
        processing_time_seconds=round(elapsed, 3),
        num_resumes=len(scored),
        rankings=rankings,
        job_required_skills=job.required_skills,
    )


@app.post("/explain")
async def explain_resume(
    job_description: str = Form(...),
    resume: UploadFile = File(...),
    model_name: str = Form(default=None),
):
    """
    Get detailed explanation for a single resume's ranking.
    Returns SHAP/LIME analysis, skill gap, and section breakdown.
    """
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    selected_model = model_name or config.ACTIVE_MODEL

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _save_upload(resume, tmp_dir)

        job = parse_job_description(job_description)
        resume_record = parse_resume(path)

        scorer = ResumeScorer(model_name=selected_model)
        scored = scorer.score_resumes(job, [resume_record])
        resume_record = scored[0]

        xai = ExplainabilityPipeline(model_type=selected_model)
        explanation = xai.explain_resume(resume_record, job, output_dir=config.PLOTS_DIR)

    return ExplainResponse(
        resume_id=resume_record.resume_id,
        rank=resume_record.rank,
        final_score=round(resume_record.final_score, 4),
        narrative=explanation.get("narrative", ""),
        skill_analysis=explanation.get("skill_analysis", {}),
        section_analysis=explanation.get("section_analysis", {}),
        score_breakdown=explanation.get("score_breakdown", {}),
        lime_text=explanation.get("lime_text"),
    )


@app.post("/rank/text")
async def rank_resumes_text(
    job_description: str = Form(...),
    resume_texts: str = Form(...),  # JSON list of dicts: [{id, text}, ...]
    model_name: str = Form(default=None),
):
    """
    Rank resumes provided as plain text (JSON array).
    Useful for programmatic API access without file uploads.
    """
    start_time = time.time()
    selected_model = model_name or config.ACTIVE_MODEL

    try:
        resume_list = json.loads(resume_texts)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="resume_texts must be valid JSON")

    if not isinstance(resume_list, list) or not resume_list:
        raise HTTPException(status_code=400, detail="resume_texts must be a non-empty JSON array")

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp_dir:
        resume_records = []
        for item in resume_list:
            rid = item.get("id", f"resume_{len(resume_records)}")
            text = item.get("text", "")
            if not text.strip():
                continue
            path = Path(tmp_dir) / f"{rid}.txt"
            path.write_text(text, encoding="utf-8")
            record = parse_resume(path, resume_id=rid)
            resume_records.append(record)

    if not resume_records:
        raise HTTPException(status_code=422, detail="No valid resume texts provided")

    job = parse_job_description(job_description)
    scorer = ResumeScorer(model_name=selected_model)
    scored = scorer.score_resumes(job, resume_records)

    elapsed = time.time() - start_time
    df = scorer.to_dataframe(scored)

    return {
        "status": "success",
        "model_used": selected_model,
        "processing_time_seconds": round(elapsed, 3),
        "rankings": df.to_dict(orient="records"),
        "job_required_skills": job.required_skills,
    }


@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """Serve a generated plot by filename."""
    plot_path = config.PLOTS_DIR / filename
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(str(plot_path), media_type="image/png")


@app.get("/config")
def get_config():
    """Return current system configuration."""
    return {
        "active_model": config.ACTIVE_MODEL,
        "section_weights": config.SECTION_WEIGHTS,
        "score_combination_weights": config.SCORE_COMBINATION_WEIGHTS,
        "bert_model": config.BERT_MODEL_NAME,
        "eval_k_values": config.EVAL_K_VALUES,
    }
