"""
Main entry point for the Resume Ranking System.
Runs the full pipeline: parse → score → explain → evaluate → report.

Usage:
    python main.py --help
    python main.py --model bert --job data/sample/job_description.txt --resumes data/sample/
    python main.py --model tfidf --train-ml
    python main.py --ablation
    python main.py --serve
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# ─── Ensure project root is on path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import log, setup_logger
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume Ranking System with Explainable AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model bert
  python main.py --model tfidf --job data/sample/job_description.txt
  python main.py --model logistic_regression --train-ml
  python main.py --model random_forest --train-ml
  python main.py --ablation
  python main.py --serve --port 8000
  python main.py --evaluate
        """,
    )

    parser.add_argument(
        "--model",
        choices=["tfidf", "bert", "logistic_regression", "random_forest"],
        default=config.ACTIVE_MODEL,
        help="Model to use for ranking (default: %(default)s)",
    )
    parser.add_argument(
        "--job",
        type=Path,
        default=config.SAMPLE_DIR / "job_description.txt",
        help="Path to job description file",
    )
    parser.add_argument(
        "--resumes",
        type=Path,
        default=config.SAMPLE_DIR,
        help="Path to directory containing resume files or single resume file",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train ML models (logistic_regression/random_forest) before ranking",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study comparing all model configurations",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation against ground truth labels",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help="Port for FastAPI server (default: %(default)s)",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip explainability (faster for large batches)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RANKINGS_DIR / "rankings.csv",
        help="Path to save ranking results CSV",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def load_resume_files(resumes_path: Path) -> list:
    """Load all resume files from a directory or single file."""
    supported = {".txt", ".pdf", ".docx"}
    if resumes_path.is_file():
        return [resumes_path]
    elif resumes_path.is_dir():
        files = [
            f for f in resumes_path.iterdir()
            if f.suffix.lower() in supported and not f.name.startswith("job_") and not f.name.startswith("data_")
        ]
        log.info(f"Found {len(files)} resume files in {resumes_path}")
        return sorted(files)
    else:
        log.error(f"Resumes path not found: {resumes_path}")
        return []


def train_ml_models(resume_files: list, job_text: str) -> None:
    """Train and save ML models using synthetic feature vectors."""
    sys.path.insert(0, str(config.ROOT_DIR))
    from data.sample.data_generator import generate_training_features
    from models.ml_models import LogisticRegressionModel, RandomForestModel

    log.info("Training ML models...")
    X, y = generate_training_features()
    log.info(f"Training data: X={X.shape}, positives={int(y.sum())}, negatives={int((1-y).sum())}")

    lr = LogisticRegressionModel()
    lr.fit(X, y)
    lr.save()
    log.info("Logistic Regression trained and saved")

    rf = RandomForestModel()
    rf.fit(X, y)
    rf.save()
    log.info("Random Forest trained and saved")

    log.info("LR Feature Importance (top 5):")
    for feat, imp in lr.get_feature_importance()[:5]:
        log.info(f"  {feat}: {imp:.4f}")

    log.info("RF Feature Importance (top 5):")
    for feat, imp in rf.get_feature_importance()[:5]:
        log.info(f"  {feat}: {imp:.4f}")


def run_ranking_pipeline(
    job_text: str,
    resume_files: list,
    model_name: str,
    run_explain: bool = True,
    output_path: Path = None,
) -> pd.DataFrame:
    """Run the full ranking pipeline."""
    from scoring.scorer import ResumeScorer, parse_job_description, parse_resume
    from explainability.explainer_pipeline import ExplainabilityPipeline
    from evaluation.fairness import run_fairness_analysis

    log.info(f"\n{'='*60}")
    log.info(f"RESUME RANKING PIPELINE")
    log.info(f"Model: {model_name} | Resumes: {len(resume_files)}")
    log.info(f"{'='*60}")

    start = time.time()

    # Parse job
    job = parse_job_description(job_text)
    log.info(f"Job required skills ({len(job.required_skills)}): {', '.join(job.required_skills[:10])}")

    # Parse resumes
    log.info("Parsing resumes...")
    resumes = []
    for rf in resume_files:
        r = parse_resume(rf)
        resumes.append(r)
        log.info(f"  {r.file_name}: {len(r.skills)} skills extracted")

    if not resumes:
        log.error("No resumes parsed successfully")
        return pd.DataFrame()

    # Score
    scorer = ResumeScorer(model_name=model_name)
    scored = scorer.score_resumes(job, resumes)
    df = scorer.to_dataframe(scored)

    parse_time = time.time() - start
    log.info(f"\nScoring complete in {parse_time:.2f}s")

    log.info("\n" + "="*60)
    log.info("RANKING RESULTS")
    log.info("="*60)
    for r in scored:
        log.info(
            f"#{r.rank:2d} | {r.file_name:38s} | Score: {r.final_score:.4f} "
            f"| BERT: {r.bert_score:.3f} | Skills: {r.skill_overlap_score:.3f} "
            f"| Matched: {len(r.skill_overlap_details.get('matched', []))}"
        )

    # Fairness analysis
    log.info("\nRunning fairness analysis...")
    fairness = run_fairness_analysis(
        df,
        output_path=config.FAIRNESS_LOG_PATH,
        plot_path=config.PLOTS_DIR / "fairness_distribution.png",
    )
    for obs in fairness.get("overall_observations", []):
        log.info(f"  {obs}")

    # Explainability
    if run_explain:
        log.info("\nGenerating explanations (SHAP/LIME/Skill Analysis)...")
        xai = ExplainabilityPipeline(model_type=model_name)
        explanations = xai.explain_all(scored, job, output_dir=config.PLOTS_DIR)

        log.info("\n" + "="*60)
        log.info("EXPLANATION SUMMARIES (Top 3)")
        log.info("="*60)
        for resume in scored[:min(3, len(scored))]:
            exp = explanations.get(resume.resume_id, {})
            narrative = exp.get("narrative", "No narrative generated")
            log.info(f"\n{resume.file_name}:")
            log.info(narrative)
            skill_plot = exp.get("skill_plot")
            if skill_plot:
                log.info(f"  Analysis plot: {skill_plot}")

    # Save results
    output_path = output_path or config.RANKINGS_DIR / "rankings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"\nRankings saved to: {output_path}")

    total_time = time.time() - start
    log.info(f"Total pipeline time: {total_time:.2f}s")

    return df


def run_evaluation(ranked_df: pd.DataFrame, job_txt_name: str = "job_description.txt") -> None:
    """Evaluate ranking against ground truth."""
    from evaluation.metrics import compute_ranking_metrics, format_metrics_report

    ground_truth_path = config.SAMPLE_DIR / "ground_truth.json"
    if not ground_truth_path.exists():
        sys.path.insert(0, str(config.ROOT_DIR))
        from data.sample.data_generator import save_ground_truth
        save_ground_truth()

    with open(ground_truth_path) as f:
        gt = json.load(f)

    relevant = gt["ground_truth"].get(job_txt_name, {}).get("relevant_resumes", [])
    if not relevant:
        log.warning("No ground truth relevant resumes found for evaluation")
        return

    ranked_ids = ranked_df["file_name"].tolist()
    metrics = compute_ranking_metrics(ranked_ids, relevant, config.EVAL_K_VALUES)

    log.info("\n" + format_metrics_report(metrics))

    report_path = config.REPORTS_DIR / "eval_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Evaluation metrics saved to {report_path}")


def run_ablation(job_text: str, resume_files: list) -> None:
    """Run ablation study across model configurations."""
    from evaluation.metrics import ablation_study
    from data.sample.data_generator import GROUND_TRUTH

    relevant_ids = GROUND_TRUTH["job_description.txt"]["relevant_resumes"]
    output_path = config.REPORTS_DIR / "ablation_study.csv"

    log.info("\n" + "="*60)
    log.info("ABLATION STUDY")
    log.info("="*60)

    df = ablation_study(job_text, resume_files, relevant_ids, output_path=output_path)

    log.info("\nAblation Results:")
    log.info(df.to_string())
    log.info(f"\nAblation study saved to {output_path}")


def serve_api(port: int = None) -> None:
    """Start the FastAPI server."""
    import uvicorn
    port = port or config.API_PORT
    log.info(f"Starting FastAPI server on http://localhost:{port}")
    log.info(f"API docs: http://localhost:{port}/docs")
    uvicorn.run(
        "api.app:app",
        host=config.API_HOST,
        port=port,
        reload=False,
        log_level="info",
    )


def main():
    args = parse_args()
    setup_logger(log_level=args.log_level)

    log.info("Resume Ranking System v1.0.0")
    log.info(f"Active model: {args.model}")

    if args.serve:
        serve_api(port=args.port)
        return

    if not args.job.exists():
        log.error(f"Job description file not found: {args.job}")
        sys.exit(1)

    job_text = args.job.read_text(encoding="utf-8")
    log.info(f"Job description loaded: {args.job.name} ({len(job_text)} chars)")

    resume_files = load_resume_files(args.resumes)
    if not resume_files:
        log.error("No resume files found")
        sys.exit(1)

    # Generate ground truth
    gt_path = config.SAMPLE_DIR / "ground_truth.json"
    if not gt_path.exists():
        sys.path.insert(0, str(config.ROOT_DIR))
        from data.sample.data_generator import save_ground_truth
        save_ground_truth()

    if args.train_ml:
        train_ml_models(resume_files, job_text)

    if args.ablation:
        run_ablation(job_text, resume_files)
        return

    ranked_df = run_ranking_pipeline(
        job_text=job_text,
        resume_files=resume_files,
        model_name=args.model,
        run_explain=not args.no_explain,
        output_path=args.output,
    )

    if args.evaluate and not ranked_df.empty:
        run_evaluation(ranked_df, job_txt_name=args.job.name)

    log.info("\nPipeline complete!")


if __name__ == "__main__":
    main()
