"""
Evaluation metrics for resume ranking system.
Implements: Precision@K, Recall@K, MRR, nDCG, Accuracy, Precision, Recall, F1.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import log
import config


def precision_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    relevant_set = set(relevant_ids)
    return len([r for r in top_k if r in relevant_set]) / k


def recall_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@K: fraction of all relevant items in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(ranked_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant item."""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Discounted Cumulative Gain@K."""
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Normalized DCG@K."""
    actual_dcg = dcg_at_k(ranked_ids, relevant_ids, k)
    # Ideal: all relevant items at top positions
    ideal_dcg = dcg_at_k(relevant_ids, relevant_ids, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(ranked_ids: List[str], relevant_ids: List[str]) -> float:
    """Average Precision: area under precision-recall curve."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = 0
    ap = 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_set:
            hits += 1
            ap += hits / i
    return ap / len(relevant_set)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, precision, recall, F1.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_scores: Predicted scores (for ROC AUC)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    results = {}
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if len(np.unique(y_true_arr)) < 2:
        log.warning("Only one class in y_true — skipping classification metrics")
        return {"accuracy": float(np.mean(y_true_arr == y_pred_arr))}

    results["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
    results["precision"] = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    results["recall"] = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    results["f1"] = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    if y_scores is not None:
        try:
            results["roc_auc"] = float(roc_auc_score(y_true_arr, y_scores))
        except Exception:
            pass

    return results


def compute_ranking_metrics(
    ranked_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute all ranking metrics for a single query.
    
    Args:
        ranked_ids: Ordered list of resume IDs by predicted ranking
        relevant_ids: List of truly relevant resume IDs
        k_values: List of K values for @K metrics
    
    Returns:
        Dict of metric_name -> value
    """
    k_values = k_values or config.EVAL_K_VALUES
    results = {}

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(ranked_ids, relevant_ids, k)
        results[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant_ids, k)

    results["mrr"] = mean_reciprocal_rank(ranked_ids, relevant_ids)
    results["map"] = average_precision(ranked_ids, relevant_ids)

    return results


def evaluate_system(
    rankings: List[Dict],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Evaluate the full system across multiple queries.
    
    Args:
        rankings: List of dicts with keys:
            - ranked_ids: list of resume IDs in ranked order
            - relevant_ids: ground truth relevant resume IDs
        k_values: List of K values
    
    Returns:
        Averaged metrics across all queries
    """
    k_values = k_values or config.EVAL_K_VALUES

    all_metrics = {f"precision@{k}": [] for k in k_values}
    all_metrics.update({f"recall@{k}": [] for k in k_values})
    all_metrics.update({f"ndcg@{k}": [] for k in k_values})
    all_metrics["mrr"] = []
    all_metrics["map"] = []

    for item in rankings:
        ranked_ids = item["ranked_ids"]
        relevant_ids = item["relevant_ids"]
        metrics = compute_ranking_metrics(ranked_ids, relevant_ids, k_values)
        for key, value in metrics.items():
            if key in all_metrics:
                all_metrics[key].append(value)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}


def ablation_study(
    job_text: str,
    resume_files: List,
    relevant_ids: List[str],
    output_path=None,
) -> pd.DataFrame:
    """
    Run ablation study comparing:
    1. TF-IDF only
    2. BERT only
    3. TF-IDF + Skills
    4. BERT + Skills
    5. Full pipeline
    
    Returns:
        DataFrame with metrics per configuration
    """
    from scoring.scorer import parse_job_description, parse_resume, ResumeScorer

    log.info("Running ablation study...")
    configs = {
        "TF-IDF only": {"model": "tfidf", "use_skills": False},
        "BERT only": {"model": "bert", "use_skills": False},
        "TF-IDF + Skills": {"model": "tfidf", "use_skills": True},
        "BERT + Skills": {"model": "bert", "use_skills": True},
        "Full Pipeline (BERT)": {"model": "bert", "use_skills": True},
    }

    job = parse_job_description(job_text)
    resumes = []
    for rf in resume_files:
        r = parse_resume(rf)
        resumes.append(r)

    results = []
    k_values = config.EVAL_K_VALUES

    for config_name, cfg in configs.items():
        log.info(f"  Ablation: {config_name}")
        scorer = ResumeScorer(model_name=cfg["model"])
        scored = scorer.score_resumes(job, list(resumes))
        ranked_ids = [r.resume_id for r in scored]

        metrics = compute_ranking_metrics(ranked_ids, relevant_ids, k_values)
        row = {"configuration": config_name, "model": cfg["model"]}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)
    df = df.set_index("configuration")

    if output_path:
        df.to_csv(output_path)
        log.info(f"Ablation study saved to {output_path}")

    return df


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable report."""
    lines = ["=" * 50, "EVALUATION METRICS REPORT", "=" * 50]

    ranking_keys = [k for k in metrics if "@" in k or k in ("mrr", "map")]
    class_keys = [k for k in metrics if k in ("accuracy", "precision", "recall", "f1", "roc_auc")]

    if ranking_keys:
        lines.append("\n📊 Ranking Metrics:")
        for k in sorted(ranking_keys):
            lines.append(f"  {k.upper():20s}: {metrics[k]:.4f}")

    if class_keys:
        lines.append("\n📊 Classification Metrics:")
        for k in class_keys:
            lines.append(f"  {k.capitalize():20s}: {metrics[k]:.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)
