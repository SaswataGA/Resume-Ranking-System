**Here's a clean, professional, and well-formatted `README.md`** for your project:

---

```markdown
# Resume Ranking System with Explainable AI

**A production-grade Machine Learning system** that ranks resumes against job descriptions using skill-based matching, semantic similarity, and transparent Explainable AI (SHAP + LIME).

---

## ✨ Features

- **Multi-Model Ranking**: TF-IDF, Sentence-BERT, Logistic Regression, Random Forest
- **Skill Ontology**: 100+ canonical skills with 400+ aliases
- **Section-weighted Scoring**: Skills, Experience, Education, Certifications
- **Explainable AI**: SHAP (global + local) + LIME explanations
- **Web Interface**: Beautiful Streamlit UI
- **REST API**: FastAPI with Swagger docs
- **Evaluation**: Precision@K, nDCG, MRR, MAP, Fairness Analysis
- **PDF + TXT Support**: Robust parsing with fallback

---

## Project Structure

```
resume_ranker/
├── data/sample/              # Sample job & resumes
├── preprocessing/            # PDF parsing & cleaning
├── skill_extraction/         # Ontology-based skill extractor
├── models/                   # TF-IDF, BERT, ML models
├── scoring/                  # Core ranking engine
├── explainability/           # SHAP + LIME
├── api/                      # FastAPI backend
├── app/                      # Streamlit UI
├── evaluation/               # Metrics & fairness
├── notebooks/                # Demo notebook
├── outputs/                  # Models, plots, reports
├── main.py                   # CLI entry point
├── config.py                 # All settings
└── requirements.txt
```

---

## Installation

### 1. Clone / Extract Project

```bash
cd resume_ranker
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Verify

```bash
python -c "import nltk, spacy, sentence_transformers, sklearn, shap, lime; print('✅ All good!')"
```

---

## Usage

### Command Line

```bash
# Quick test with BERT (recommended)
python main.py --model bert

# Use TF-IDF baseline
python main.py --model tfidf

# Train ML models
python main.py --model logistic_regression --train-ml
```

### Streamlit Web App (Recommended)

```bash
streamlit run app/streamlit_app.py
```

### FastAPI Server

```bash
python main.py --serve --port 8000
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

### Jupyter Demo

```bash
jupyter notebook notebooks/resume_ranking_demo.ipynb
```

---

## Available Models

| Model                    | Type               | Speed | Best For                  |
|-------------------------|--------------------|-------|---------------------------|
| `bert`                  | Semantic           | Medium| Production (Recommended) |
| `tfidf`                 | Keyword            | Fast  | Baseline                 |
| `logistic_regression`   | ML + Features      | Medium| Interpretable            |
| `random_forest`         | ML + Features      | Medium| Feature Importance       |

---

## Configuration (`config.py`)

Key settings you can modify:
- `ACTIVE_MODEL`
- `SECTION_WEIGHTS`
- `SCORE_COMBINATION_WEIGHTS`
- `BERT_MODEL_NAME`

---

## API Endpoints

- `POST /rank` — Rank resumes
- `POST /explain` — Get SHAP/LIME explanation
- `GET /health`, `GET /models`

---

## Output Folders

- `outputs/models/` — Saved trained models
- `outputs/plots/` — SHAP, LIME, skill charts
- `outputs/rankings/` — CSV results
- `outputs/reports/` — Evaluation reports

---

## Testing

```bash
pytest tests/ -v
```

---

## Contributing

1. Fork the repo
2. Create feature branch
3. Make changes + tests
4. Submit PR

---

## License

MIT License

---

**Made with ❤️ for smarter hiring**
