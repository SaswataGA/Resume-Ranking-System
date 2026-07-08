"""
Fairness analysis module for resume ranking system.
Detects potential bias in ranking distributions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import log
import config


def compute_score_statistics(scores: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics for score distribution."""
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
        "iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
        "variance": float(np.var(scores)),
    }


def detect_score_clustering(scores: np.ndarray, threshold: float = None) -> Dict:
    """
    Detect whether scores cluster unusually (potential bias indicator).
    
    Returns observations about score distribution shape.
    """
    threshold = threshold or config.FAIRNESS_SCORE_VARIANCE_THRESHOLD
    stats = compute_score_statistics(scores)
    observations = []

    if stats["variance"] < threshold * threshold:
        observations.append(
            f"⚠ Score variance ({stats['variance']:.4f}) is very low — "
            "scores may not distinguish candidates effectively."
        )

    if stats["max"] - stats["min"] < 0.1:
        observations.append(
            f"⚠ Score range is very narrow ({stats['min']:.3f} to {stats['max']:.3f}) — "
            "ranking may not be meaningful."
        )

    top_bottom_gap = scores[np.argsort(scores)[-1]] - scores[np.argsort(scores)[0]]
    if top_bottom_gap > 0.9:
        observations.append(
            f"ℹ Large gap between top and bottom candidates ({top_bottom_gap:.3f}) — "
            "clear differentiation in ranking."
        )

    if not observations:
        observations.append("✅ Score distribution appears reasonable.")

    return {
        "statistics": stats,
        "observations": observations,
        "variance_flag": stats["variance"] < threshold * threshold,
    }


def analyze_group_fairness(
    rankings_df: pd.DataFrame,
    group_column: Optional[str] = None,
) -> Dict:
    """
    Analyze fairness across groups if group information is available.
    
    Since resumes often don't contain explicit demographic info,
    this analyzes proxy signals like name-based inferences if present.
    
    Args:
        rankings_df: DataFrame with resume rankings and scores
        group_column: Column name for group (if available)
    
    Returns:
        Fairness analysis results
    """
    report = {
        "has_group_data": False,
        "group_analysis": {},
        "observations": [],
        "recommendation": "",
    }

    if group_column and group_column in rankings_df.columns:
        report["has_group_data"] = True
        groups = rankings_df[group_column].unique()

        for group in groups:
            group_data = rankings_df[rankings_df[group_column] == group]
            group_scores = group_data["final_score"].values
            report["group_analysis"][str(group)] = compute_score_statistics(group_scores)

        # Check for disparate impact
        if len(groups) >= 2:
            group_means = {
                g: rankings_df[rankings_df[group_column] == g]["final_score"].mean()
                for g in groups
            }
            max_mean = max(group_means.values())
            min_mean = min(group_means.values())
            ratio = min_mean / max_mean if max_mean > 0 else 0

            report["disparate_impact_ratio"] = ratio
            if ratio < 0.8:
                report["observations"].append(
                    f"⚠ Potential disparate impact detected: group mean score ratio = {ratio:.3f} "
                    f"(below 0.8 threshold). Groups: {group_means}"
                )
            else:
                report["observations"].append(
                    f"✅ No significant disparate impact: ratio = {ratio:.3f}"
                )
    else:
        report["observations"].append(
            "ℹ No demographic/group data available in resumes. "
            "Fairness analysis is limited to score distribution analysis."
        )
        report["observations"].append(
            "Recommendation: Manually review top-ranked candidates to check for "
            "name, school, or geography-based patterns."
        )

    return report


def rank_distribution_analysis(scores: np.ndarray, file_names: List[str]) -> Dict:
    """Analyze the distribution of rankings for potential anomalies."""
    n = len(scores)
    if n == 0:
        return {}

    sorted_indices = np.argsort(scores)[::-1]
    score_gaps = []
    for i in range(len(sorted_indices) - 1):
        gap = scores[sorted_indices[i]] - scores[sorted_indices[i + 1]]
        score_gaps.append(float(gap))

    observations = []

    # Check for cliff between rank 1 and rest
    if score_gaps and score_gaps[0] > 0.2:
        observations.append(
            f"📊 Large score gap between rank 1 and rank 2 ({score_gaps[0]:.3f}). "
            "Top candidate clearly differentiated."
        )

    # Check for ties or near-ties
    near_ties = sum(1 for g in score_gaps if g < 0.01)
    if near_ties > n * 0.3:
        observations.append(
            f"⚠ {near_ties} near-ties detected (score gap < 0.01). "
            "Consider additional tie-breaking criteria."
        )

    return {
        "n_candidates": n,
        "score_gaps": score_gaps[:10],
        "near_tie_count": near_ties,
        "observations": observations,
    }


def run_fairness_analysis(
    rankings_df: pd.DataFrame,
    output_path: Path = None,
    plot_path: Path = None,
) -> Dict:
    """
    Run full fairness analysis on ranking results.
    
    Returns:
        Complete fairness report dict
    """
    scores = rankings_df["final_score"].values
    file_names = rankings_df["file_name"].tolist()

    report = {
        "score_clustering": detect_score_clustering(scores),
        "rank_distribution": rank_distribution_analysis(scores, file_names),
        "group_fairness": analyze_group_fairness(rankings_df),
        "overall_observations": [],
    }

    # Aggregate observations
    for section in ["score_clustering", "rank_distribution", "group_fairness"]:
        obs = report[section].get("observations", [])
        report["overall_observations"].extend(obs)

    # Save report
    output_path = output_path or config.FAIRNESS_LOG_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Fairness report saved: {output_path}")

    # Generate distribution plot
    if plot_path is None:
        plot_path = config.PLOTS_DIR / "fairness_score_distribution.png"

    _plot_score_distribution(scores, file_names, plot_path)
    report["distribution_plot"] = str(plot_path)

    return report


def _plot_score_distribution(
    scores: np.ndarray,
    labels: List[str],
    save_path: Path,
) -> None:
    """Plot score distribution for fairness visualization."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Distribution histogram
        axes[0].hist(scores, bins=min(20, len(scores)), color="#3498db", alpha=0.7, edgecolor="white")
        axes[0].axvline(np.mean(scores), color="#e74c3c", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
        axes[0].axvline(np.median(scores), color="#27ae60", linestyle="--", label=f"Median: {np.median(scores):.3f}")
        axes[0].set_xlabel("Final Score", fontsize=10)
        axes[0].set_ylabel("Count", fontsize=10)
        axes[0].set_title("Score Distribution", fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # Bar chart of individual scores
        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_idx]
        sorted_labels = [labels[i] for i in sorted_idx]

        display_n = min(15, len(sorted_scores))
        colors = plt.cm.RdYlGn(sorted_scores[:display_n])
        axes[1].barh(range(display_n), sorted_scores[:display_n], color=colors, alpha=0.85)
        axes[1].set_yticks(range(display_n))
        axes[1].set_yticklabels(sorted_labels[:display_n], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Final Score", fontsize=10)
        axes[1].set_title(f"Candidate Rankings (Top {display_n})", fontsize=11, fontweight="bold")
        axes[1].grid(axis="x", alpha=0.3)

        plt.suptitle("Fairness: Score Distribution Analysis", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Fairness distribution plot saved: {save_path}")
    except Exception as e:
        log.error(f"Fairness plot failed: {e}")


def format_fairness_report(report: Dict) -> str:
    """Format fairness report as readable text."""
    lines = ["=" * 55, "FAIRNESS ANALYSIS REPORT", "=" * 55]

    stats = report.get("score_clustering", {}).get("statistics", {})
    if stats:
        lines.append("\n📊 Score Statistics:")
        for k, v in stats.items():
            lines.append(f"  {k:12s}: {v:.4f}")

    lines.append("\n🔍 Observations:")
    for obs in report.get("overall_observations", []):
        lines.append(f"  {obs}")

    lines.append("=" * 55)
    return "\n".join(lines)
