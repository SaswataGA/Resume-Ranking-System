"""
Synthetic data generator for testing the resume ranking system.
Creates realistic resume-job pairs with ground truth relevance labels.
"""

import random
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


GROUND_TRUTH = {
    "job_description.txt": {
        "relevant_resumes": [
            "resume_alice_johnson.txt",
            "resume_eva_chen.txt",
            "resume_bob_smith.txt",
        ],
        "highly_relevant": [
            "resume_alice_johnson.txt",
            "resume_eva_chen.txt",
        ],
        "somewhat_relevant": [
            "resume_frank_martinez.txt",
            "resume_carol_williams.txt",
        ],
        "not_relevant": [
            "resume_david_lee.txt",
        ],
    }
}

RELEVANCE_SCORES = {
    "resume_alice_johnson.txt": 5,    # Highly relevant
    "resume_eva_chen.txt": 5,         # Highly relevant
    "resume_bob_smith.txt": 4,        # Relevant
    "resume_frank_martinez.txt": 2,   # Partially relevant
    "resume_carol_williams.txt": 2,   # Partially relevant
    "resume_david_lee.txt": 1,        # Not relevant
}


def save_ground_truth():
    """Save ground truth labels for evaluation."""
    output = {
        "ground_truth": GROUND_TRUTH,
        "relevance_scores": RELEVANCE_SCORES,
        "notes": {
            "scoring": "1=not relevant, 2=partially relevant, 3=relevant, 4=very relevant, 5=highly relevant",
            "threshold": "Scores >= 3 are considered relevant for Precision@K and Recall@K",
        }
    }
    path = SAMPLE_DIR / "ground_truth.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Ground truth saved to {path}")
    return output


def generate_training_features() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training feature vectors and labels for ML models.
    Used when no real labeled training data is available.

    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Binary labels (1 = relevant, 0 = not relevant)
    """
    n_pos = 80
    n_neg = 120
    n_features = 15  # Matches FEATURE_NAMES in ml_models.py

    # Positive samples (relevant resumes)
    X_pos = np.zeros((n_pos, n_features))
    X_pos[:, 0] = np.random.beta(5, 2, n_pos)          # tfidf_similarity (high)
    X_pos[:, 1] = np.random.beta(6, 2, n_pos)          # bert_similarity (high)
    X_pos[:, 2] = np.random.beta(4, 2, n_pos)          # skill_jaccard (high)
    X_pos[:, 3] = np.random.randint(3, 15, n_pos).astype(float)  # skill_overlap_count
    X_pos[:, 4] = np.random.beta(5, 2, n_pos)          # skill_precision
    X_pos[:, 5] = np.random.beta(4, 2, n_pos)          # skill_recall
    X_pos[:, 6] = np.random.beta(5, 2, n_pos)          # skill_f1
    X_pos[:, 7] = np.random.beta(5, 2, n_pos)          # skills_section_sim
    X_pos[:, 8] = np.random.beta(4, 2, n_pos)          # experience_section_sim
    X_pos[:, 9] = np.random.beta(4, 2, n_pos)          # education_section_sim
    X_pos[:, 10] = np.random.beta(3, 2, n_pos)         # cert_section_sim
    X_pos[:, 11] = np.random.randint(500, 2000, n_pos).astype(float)  # resume_length
    X_pos[:, 12] = np.random.randint(8, 25, n_pos).astype(float)      # num_candidate_skills
    X_pos[:, 13] = np.random.randint(10, 20, n_pos).astype(float)     # num_job_skills
    X_pos[:, 14] = np.random.uniform(3, 12, n_pos)     # years_experience_est

    # Negative samples (irrelevant resumes)
    X_neg = np.zeros((n_neg, n_features))
    X_neg[:, 0] = np.random.beta(2, 5, n_neg)          # tfidf_similarity (low)
    X_neg[:, 1] = np.random.beta(2, 6, n_neg)          # bert_similarity (low)
    X_neg[:, 2] = np.random.beta(1, 5, n_neg)          # skill_jaccard (low)
    X_neg[:, 3] = np.random.randint(0, 4, n_neg).astype(float)   # skill_overlap_count
    X_neg[:, 4] = np.random.beta(2, 5, n_neg)          # skill_precision
    X_neg[:, 5] = np.random.beta(1, 5, n_neg)          # skill_recall
    X_neg[:, 6] = np.random.beta(1, 5, n_neg)          # skill_f1
    X_neg[:, 7] = np.random.beta(2, 5, n_neg)          # skills_section_sim
    X_neg[:, 8] = np.random.beta(2, 5, n_neg)          # experience_section_sim
    X_neg[:, 9] = np.random.beta(2, 5, n_neg)          # education_section_sim
    X_neg[:, 10] = np.random.beta(1, 5, n_neg)         # cert_section_sim
    X_neg[:, 11] = np.random.randint(100, 800, n_neg).astype(float)   # resume_length
    X_neg[:, 12] = np.random.randint(1, 8, n_neg).astype(float)       # num_candidate_skills
    X_neg[:, 13] = np.random.randint(10, 20, n_neg).astype(float)     # num_job_skills
    X_neg[:, 14] = np.random.uniform(0, 3, n_neg)      # years_experience_est

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


if __name__ == "__main__":
    save_ground_truth()
    X, y = generate_training_features()
    print(f"Generated training data: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.sum()} positive, {(1-y).sum()} negative")
