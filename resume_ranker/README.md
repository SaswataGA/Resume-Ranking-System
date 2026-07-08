# Resume Ranking System with Explainable AI

A production-grade, research-quality machine learning system that ranks resumes against job descriptions using skill-based matching, semantic similarity, and transparent Explainable AI (SHAP + LIME).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [FastAPI Server](#fastapi-server)
  - [Streamlit Web App](#streamlit-web-app)
  - [Jupyter Notebook](#jupyter-notebook)
- [Models](#models)
- [Explainability](#explainability)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Example Output](#example-output)
- [Research Notes](#research-notes)

---

## Overview

This system provides end-to-end automated resume screening with:

- **Skill-based matching** using a curated skill ontology (100+ canonical skills, 400+ aliases)
- **Semantic similarity** via Sentence-BERT embeddings
- **Multiple ML models** (TF-IDF, BERT, Logistic Regression, Random Forest) — all manually selectable
- **Transparent explanations** — SHAP global importance + LIME per-resume explanations
- **Section-weighted scoring** — Skills, Experience, Education, Certifications with configurable weights
- **Fairness analysis** — Score distribution and bias detection
- **REST API** (FastAPI) + **Interactive Web UI** (Streamlit)

---

## Features

| Feature | Details |
|---|---|
| PDF + TXT parsing | pdfplumber (primary), PyMuPDF (fallback) |
| Text preprocessing | NLTK tokenisation, stopword removal, lemmatisation |
| Section extraction | Skills, Experience, Education, Certifications, Summary |
| Skill normalisation | Ontology-based alias mapping (e.g. `py` → `python`, `k8s` → `kubernetes`) |
| TF-IDF baseline | Cosine similarity on cleaned, lemmatised text |
| BERT semantic | `all-MiniLM-L6-v2` Sentence Transformer |
| ML ranking | Logistic Regression + Random Forest on 15-feature vector |
| Explainability | SHAP (TreeExplainer / LinearExplainer) + LIME (tabular) |
| Evaluation | Precision@K, Recall@K, MRR, nDCG, MAP, Accuracy, F1 |
| Ablation study | 5-configuration comparison table |
| Fairness analysis | Score distribution stats + disparate impact check |
| API | FastAPI with `/rank`, `/explain`, `/health`, `/models` |
| UI | Streamlit app with interactive plots |
| Logging | Loguru with file rotation |
| Experiment tracking | MLflow integration |
| Model persistence | joblib save/load for all models |

---

## Project Structure

```
resume_ranker/
│
├── data/
│   ├── raw/                    # Raw uploaded resumes
│   ├── processed/              # Preprocessed cache
│   └── sample/
│       ├── job_description.txt          # Sample job description
│       ├── resume_alice_johnson.txt     # Sample resume (strong)
│       ├── resume_eva_chen.txt          # Sample resume (strong, NLP)
│       ├── resume_bob_smith.txt         # Sample resume (good)
│       ├── resume_frank_martinez.txt    # Sample resume (DevOps)
│       ├── resume_carol_williams.txt    # Sample resume (moderate)
│       ├── resume_david_lee.txt         # Sample resume (weak)
│       ├── data_generator.py            # Synthetic data + ground truth
│       └── ground_truth.json            # Generated ground truth labels
│
├── preprocessing/
│   ├── pdf_parser.py           # PDF/TXT/DOCX parsing (robust)
│   ├── text_cleaner.py         # Cleaning, tokenisation, lemmatisation
│   └── section_extractor.py    # Section detection (Skills, Experience, etc.)
│
├── skill_extraction/
│   ├── skill_ontology.json     # 100+ canonical skills, 400+ aliases
│   └── skill_extractor.py      # Regex-based extraction + normalisation
│
├── models/
│   ├── tfidf_model.py          # TF-IDF + Cosine Similarity
│   ├── bert_model.py           # Sentence-BERT embeddings
│   └── ml_models.py            # Logistic Regression + Random Forest
│
├── similarity/
│   └── similarity_engine.py    # Unified similarity interface
│
├── scoring/
│   └── scorer.py               # ResumeScorer: full pipeline orchestration
│
├── explainability/
│   ├── shap_explainer.py       # SHAP global + local explanations
│   ├── lime_explainer.py       # LIME local explanations
│   └── explainer_pipeline.py   # Unified XAI pipeline + visualisations
│
├── evaluation/
│   ├── metrics.py              # Precision@K, Recall@K, MRR, nDCG, F1
│   └── fairness.py             # Bias detection + score distribution analysis
│
├── api/
│   └── app.py                  # FastAPI application
│
├── app/
│   └── streamlit_app.py        # Streamlit web UI
│
├── utils/
│   ├── logger.py               # Loguru logger setup
│   └── helpers.py              # save/load, normalise, misc utilities
│
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_preprocessing.py   # Tests: cleaning, parsing, sections
│   ├── test_scoring.py         # Tests: skill extraction, TF-IDF, scorer
│   ├── test_models.py          # Tests: save/load, ML model outputs
│   ├── test_evaluation.py      # Tests: metrics, fairness, helpers
│   └── test_api.py             # Tests: FastAPI endpoints
│
├── notebooks/
│   └── resume_ranking_demo.ipynb   # End-to-end Jupyter demo
│
├── outputs/
│   ├── models/                 # Saved model files (.joblib, .pkl)
│   ├── plots/                  # Generated SHAP, LIME, skill analysis plots
│   ├── rankings/               # CSV ranking outputs
│   └── reports/                # Evaluation + fairness reports
│
├── main.py                     # CLI entry point
├── config.py                   # Central configuration (all settings)
└── requirements.txt            # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1 — Clone / extract the project

```bash
cd resume_ranker
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Verify installation

```bash
python -c "import sentence_transformers, sklearn, shap, lime, fastapi; print('All OK')"
```

---

## Configuration

All settings live in `config.py`. Key options:

```python
# Model selection (override on the CLI with --model)
ACTIVE_MODEL = "bert"   # "tfidf" | "bert" | "logistic_regression" | "random_forest"

# Section weights (must sum to 1.0)
SECTION_WEIGHTS = {
    "skills":         0.50,
    "experience":     0.30,
    "education":      0.15,
    "certifications": 0.05,
}

# Score combination weights (must sum to 1.0)
SCORE_COMBINATION_WEIGHTS = {
    "semantic_similarity": 0.45,
    "skill_overlap":       0.40,
    "section_weighted":    0.15,
}

# BERT model name
BERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Evaluation K values
EVAL_K_VALUES = [1, 3, 5, 10]
```

---

## Usage

### Command-Line Interface

#### Run with BERT (default model, sample data)

```bash
python main.py --model bert
```

#### Run with TF-IDF baseline

```bash
python main.py --model tfidf
```

#### Use custom job description and resume folder

```bash
python main.py \
    --model bert \
    --job path/to/job_description.txt \
    --resumes path/to/resumes/
```

#### Train ML models first, then rank

```bash
python main.py --model logistic_regression --train-ml
python main.py --model random_forest --train-ml
```

#### Run with evaluation metrics

```bash
python main.py --model bert --evaluate
```

#### Run ablation study

```bash
python main.py --ablation
```

#### Skip explanations for faster batch processing

```bash
python main.py --model tfidf --no-explain
```

#### All CLI options

```
usage: main.py [-h]
               [--model {tfidf,bert,logistic_regression,random_forest}]
               [--job JOB]
               [--resumes RESUMES]
               [--train-ml]
               [--ablation]
               [--evaluate]
               [--serve]
               [--port PORT]
               [--no-explain]
               [--output OUTPUT]
               [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

---

### FastAPI Server

#### Start the server

```bash
python main.py --serve --port 8000
```

Or directly with uvicorn:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Interactive API docs

Open in your browser: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

Opens at [http://localhost:8501](http://localhost:8501)

Features:
- Paste job description or use sample
- Upload PDF/TXT resumes or use sample data
- Select model and adjust weights via sidebar sliders
- View ranked results, score breakdown, skill gap analysis
- View SHAP global importance and LIME per-resume explanations
- Download rankings as CSV

---

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook resume_ranking_demo.ipynb
```

The notebook walks through all 16 steps: preprocessing → skill extraction → ranking with all 4 models → SHAP → LIME → evaluation → ablation → fairness.

---

## Models

### A. TF-IDF + Cosine Similarity (`--model tfidf`)

- Baseline model
- Text is cleaned, tokenised, lemmatised, stopwords removed
- `TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)`
- Fast; good keyword matching but misses semantic meaning

### B. Sentence-BERT (`--model bert`)

- `all-MiniLM-L6-v2` from `sentence-transformers`
- 384-dim dense embeddings; cosine similarity
- Embedding cache for repeated texts
- Recommended for production use

### C. Logistic Regression (`--model logistic_regression`)

- 15-feature vector: TF-IDF sim, BERT sim, skill Jaccard/F1/precision/recall, per-section scores, resume length, skill counts, years experience
- Standard scaler + `LogisticRegression(C=1.0, max_iter=1000)`
- Requires `--train-ml` on first run

### D. Random Forest (`--model random_forest`)

- Same 15-feature vector as LR
- `RandomForestClassifier(n_estimators=200, max_depth=10)`
- Provides native Gini feature importance
- Requires `--train-ml` on first run

### Final Score Combination

```
final_score = w_semantic * semantic_sim
            + w_skill    * skill_overlap_f1
            + w_section  * section_weighted_sim
```

Weights are configurable in `config.py` or via Streamlit sliders.

---

## Explainability

### SHAP (Global + Local)

- **Global**: Mean absolute SHAP values across all resumes → which features matter most overall
- **Local**: Per-resume SHAP bar chart → which features pushed a specific resume up or down
- Uses `TreeExplainer` for Random Forest, `LinearExplainer` for Logistic Regression

### LIME (Local)

- `LimeTabularExplainer` on the 15-feature vector
- Shows which features positively/negatively contributed to the score
- Human-readable text output + bar chart visualisation

### Skill Analysis Plots

Every resume gets a 6-panel visualisation:
1. Score breakdown (BERT, TF-IDF, Skill Overlap, Section, Final)
2. Per-section similarity scores
3. Matched skills (green)
4. Missing skills (red)
5. Skill distribution pie chart
6. Narrative explanation

---

## Evaluation

### Ranking Metrics

| Metric | Description |
|---|---|
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of relevant resumes in top-K |
| nDCG@K | Normalised Discounted Cumulative Gain |
| MRR | Mean Reciprocal Rank (position of first relevant) |
| MAP | Mean Average Precision |

### Classification Metrics (ML models)

Accuracy, Precision, Recall, F1, ROC-AUC (on training data via cross-validation)

### Ablation Study

Compares 5 configurations:

| Configuration | Description |
|---|---|
| TF-IDF only | Baseline keyword matching |
| BERT only | Semantic similarity only |
| TF-IDF + Skills | Baseline + skill overlap signal |
| BERT + Skills | Semantic + skill overlap signal |
| Full Pipeline | BERT + Skills + section weighting |

Results saved to `outputs/reports/ablation_study.csv`

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run a specific test file

```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_scoring.py -v
pytest tests/test_models.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_api.py -v
```

### Run with coverage

```bash
pip install pytest-cov
pytest tests/ --cov=. --cov-report=html
```

---

## API Reference

### `GET /health`

Returns system health and active model.

```json
{"status": "ok", "version": "1.0.0", "active_model": "bert"}
```

### `GET /models`

Lists available models and descriptions.

### `GET /config`

Returns current configuration (weights, model, parameters).

### `POST /rank`

Upload job description + resume files; get ranked results.

**Form fields:**
- `job_description` (string) — full job description text
- `model_name` (string, optional) — one of `tfidf`, `bert`, `logistic_regression`, `random_forest`
- `resumes` (files) — one or more PDF or TXT files

**Response:**
```json
{
  "status": "success",
  "model_used": "bert",
  "processing_time_seconds": 1.23,
  "num_resumes": 6,
  "rankings": [
    {
      "rank": 1,
      "resume_id": "resume_alice_johnson",
      "file_name": "resume_alice_johnson.txt",
      "final_score": 0.8742,
      "bert_score": 0.8901,
      "tfidf_score": 0.7234,
      "skill_overlap_score": 0.8500,
      "matched_skills": ["python", "pytorch", "bert", "aws", "docker"],
      "missing_skills": [],
      "narrative": "🥇 TOP CANDIDATE — ranked #1 ..."
    }
  ],
  "job_required_skills": ["python", "tensorflow", "pytorch", "aws", "docker"]
}
```

### `POST /rank/text`

Rank resumes provided as JSON text (no file upload needed).

**Form fields:**
- `job_description` (string)
- `resume_texts` (JSON string) — array of `{"id": "...", "text": "..."}`
- `model_name` (string, optional)

### `POST /explain`

Get detailed SHAP/LIME explanation for a single resume.

**Form fields:**
- `job_description` (string)
- `resume` (file) — single PDF or TXT
- `model_name` (string, optional)

**Response:**
```json
{
  "resume_id": "resume_alice_johnson",
  "rank": 1,
  "final_score": 0.8742,
  "narrative": "🥇 TOP CANDIDATE ...",
  "skill_analysis": {
    "matched_skills": ["python", "pytorch"],
    "missing_skills": [],
    "overlap_score": 0.85
  },
  "section_analysis": {
    "skills": {"score": 0.91, "weight": 0.50, "has_content": true},
    "experience": {"score": 0.88, "weight": 0.30, "has_content": true}
  },
  "score_breakdown": {
    "tfidf_score": 0.72,
    "bert_score": 0.89,
    "skill_overlap_score": 0.85,
    "section_weighted_score": 0.87
  },
  "lime_text": "=== LIME Explanation: resume_alice_johnson ===\n✅ POSITIVE contributors ..."
}
```

### `GET /plots/{filename}`

Serve a generated plot PNG by filename from `outputs/plots/`.

---

## Example Output

```
============================================================
RESUME RANKING PIPELINE
Model: bert | Resumes: 6
============================================================
Job required skills (18): python, machine learning, pytorch, tensorflow, nlp ...

Parsing resumes...
  resume_alice_johnson.txt : 22 skills extracted
  resume_eva_chen.txt      : 20 skills extracted
  resume_bob_smith.txt     : 16 skills extracted
  resume_frank_martinez.txt: 12 skills extracted
  resume_carol_williams.txt:  7 skills extracted
  resume_david_lee.txt     :  2 skills extracted

Scoring complete in 4.31s

============================================================
RANKING RESULTS
============================================================
 #1 | resume_alice_johnson.txt          | Score: 0.8742 | BERT: 0.891 | Skills: 0.851 | Matched: 15
 #2 | resume_eva_chen.txt               | Score: 0.8615 | BERT: 0.879 | Skills: 0.832 | Matched: 14
 #3 | resume_bob_smith.txt              | Score: 0.7204 | BERT: 0.741 | Skills: 0.700 | Matched: 10
 #4 | resume_frank_martinez.txt         | Score: 0.5013 | BERT: 0.531 | Skills: 0.460 | Matched:  6
 #5 | resume_carol_williams.txt         | Score: 0.4102 | BERT: 0.420 | Skills: 0.390 | Matched:  4
 #6 | resume_david_lee.txt              | Score: 0.1543 | BERT: 0.180 | Skills: 0.110 | Matched:  1
```

---

## Research Notes

### Skill Ontology

The `skill_extraction/skill_ontology.json` contains:
- 100+ canonical skill names
- 400+ alias mappings
- Skills categorised into: programming languages, databases, ML frameworks, data science, big data, cloud, DevOps, web frameworks, soft skills

To extend: add entries to `canonical_skills` in `skill_ontology.json` — no code changes required.

### Feature Vector (15 dimensions)

| # | Feature | Description |
|---|---|---|
| 0 | `tfidf_similarity` | TF-IDF cosine similarity |
| 1 | `bert_similarity` | BERT semantic similarity |
| 2 | `skill_jaccard` | Jaccard index of skill sets |
| 3 | `skill_overlap_count` | Raw count of matching skills |
| 4 | `skill_precision` | Skills candidate has that job needs |
| 5 | `skill_recall` | Job skills covered by candidate |
| 6 | `skill_f1` | Harmonic mean of skill precision/recall |
| 7 | `skills_section_sim` | BERT sim on Skills section only |
| 8 | `experience_section_sim` | BERT sim on Experience section |
| 9 | `education_section_sim` | BERT sim on Education section |
| 10 | `cert_section_sim` | BERT sim on Certifications section |
| 11 | `resume_length` | Total character count |
| 12 | `num_candidate_skills` | Number of skills extracted from resume |
| 13 | `num_job_skills` | Number of skills in job description |
| 14 | `years_experience_est` | Heuristic years-of-experience estimate |

### Extending the System

- **New model**: Subclass from `TFIDFModel` or `BERTSimilarityModel`, add to `ResumeScorer._combine_scores`
- **New skill**: Add to `skill_ontology.json` under `canonical_skills`
- **New section**: Add regex patterns to `SECTION_PATTERNS` in `section_extractor.py`
- **New metric**: Add function to `evaluation/metrics.py` and include in `compute_ranking_metrics`

---

## License

MIT License — free to use, modify, and distribute for research and commercial purposes.
