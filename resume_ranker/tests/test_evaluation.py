"""Unit tests for evaluation and metrics module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRankingMetrics:
    def test_precision_at_k_perfect(self):
        from evaluation.metrics import precision_at_k
        ranked = ["a", "b", "c", "d"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(ranked, relevant, 3) == 1.0

    def test_precision_at_k_zero(self):
        from evaluation.metrics import precision_at_k
        ranked = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert precision_at_k(ranked, relevant, 3) == 0.0

    def test_precision_at_k_partial(self):
        from evaluation.metrics import precision_at_k
        ranked = ["a", "x", "b", "y"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(ranked, relevant, 2) == 0.5

    def test_precision_at_k_zero_k(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b"], ["a"], 0) == 0.0

    def test_recall_at_k_perfect(self):
        from evaluation.metrics import recall_at_k
        ranked = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(ranked, relevant, 3) == 1.0

    def test_recall_at_k_partial(self):
        from evaluation.metrics import recall_at_k
        ranked = ["a", "x", "b", "c"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)

    def test_recall_at_k_empty_relevant(self):
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b"], [], 2) == 0.0

    def test_mrr_first_is_relevant(self):
        from evaluation.metrics import mean_reciprocal_rank
        ranked = ["a", "b", "c"]
        relevant = ["a"]
        assert mean_reciprocal_rank(ranked, relevant) == 1.0

    def test_mrr_second_is_relevant(self):
        from evaluation.metrics import mean_reciprocal_rank
        ranked = ["x", "a", "b"]
        relevant = ["a"]
        assert mean_reciprocal_rank(ranked, relevant) == 0.5

    def test_mrr_none_relevant(self):
        from evaluation.metrics import mean_reciprocal_rank
        ranked = ["x", "y", "z"]
        relevant = ["a"]
        assert mean_reciprocal_rank(ranked, relevant) == 0.0

    def test_ndcg_at_k_perfect(self):
        from evaluation.metrics import ndcg_at_k
        ranked = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(1.0)

    def test_ndcg_at_k_worst_order(self):
        from evaluation.metrics import ndcg_at_k
        ranked = ["c", "b", "a"]
        relevant = ["a"]
        score_worst = ndcg_at_k(ranked, relevant, 3)
        ranked_best = ["a", "b", "c"]
        score_best = ndcg_at_k(ranked_best, relevant, 3)
        assert score_best > score_worst

    def test_average_precision(self):
        from evaluation.metrics import average_precision
        ranked = ["a", "x", "b", "y", "c"]
        relevant = ["a", "b", "c"]
        ap = average_precision(ranked, relevant)
        assert 0 < ap <= 1

    def test_compute_ranking_metrics_full(self):
        from evaluation.metrics import compute_ranking_metrics
        ranked = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b"]
        metrics = compute_ranking_metrics(ranked, relevant, k_values=[1, 2, 3])
        assert "precision@1" in metrics
        assert "recall@1" in metrics
        assert "ndcg@1" in metrics
        assert "mrr" in metrics
        assert "map" in metrics
        assert metrics["precision@2"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_compute_classification_metrics(self):
        from evaluation.metrics import compute_classification_metrics
        y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_system_multiple_queries(self):
        from evaluation.metrics import evaluate_system
        rankings = [
            {"ranked_ids": ["a", "b", "c"], "relevant_ids": ["a", "b"]},
            {"ranked_ids": ["x", "a", "b"], "relevant_ids": ["a"]},
        ]
        metrics = evaluate_system(rankings, k_values=[1, 2])
        assert "precision@1" in metrics
        assert "mrr" in metrics
        assert 0 <= metrics["mrr"] <= 1


class TestFairness:
    def test_compute_score_statistics(self):
        from evaluation.fairness import compute_score_statistics
        import numpy as np
        scores = np.array([0.8, 0.6, 0.7, 0.5, 0.9])
        stats = compute_score_statistics(scores)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert abs(stats["mean"] - 0.7) < 0.01

    def test_detect_score_clustering_normal(self):
        from evaluation.fairness import detect_score_clustering
        import numpy as np
        scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
        result = detect_score_clustering(scores)
        assert "statistics" in result
        assert "observations" in result
        assert isinstance(result["variance_flag"], bool)

    def test_run_fairness_analysis(self, tmp_path):
        from evaluation.fairness import run_fairness_analysis
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            "file_name": [f"resume_{i}.txt" for i in range(6)],
            "final_score": np.array([0.9, 0.75, 0.6, 0.5, 0.3, 0.2]),
            "rank": [1, 2, 3, 4, 5, 6],
        })

        report = run_fairness_analysis(
            df,
            output_path=tmp_path / "fairness.json",
            plot_path=tmp_path / "fairness.png",
        )

        assert "score_clustering" in report
        assert "rank_distribution" in report
        assert "overall_observations" in report

    def test_rank_distribution_analysis(self):
        from evaluation.fairness import rank_distribution_analysis
        import numpy as np
        scores = np.array([0.95, 0.60, 0.58, 0.55, 0.20])
        labels = [f"r{i}" for i in range(5)]
        result = rank_distribution_analysis(scores, labels)
        assert "n_candidates" in result
        assert result["n_candidates"] == 5
        assert "near_tie_count" in result


class TestHelpers:
    def test_normalize_scores(self):
        from utils.helpers import normalize_scores
        import numpy as np
        scores = np.array([2.0, 4.0, 6.0, 8.0])
        normed = normalize_scores(scores)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)

    def test_normalize_scores_all_same(self):
        from utils.helpers import normalize_scores
        import numpy as np
        scores = np.array([0.5, 0.5, 0.5])
        normed = normalize_scores(scores)
        assert all(n == pytest.approx(0.5) for n in normed)

    def test_ranks_from_scores(self):
        from utils.helpers import ranks_from_scores
        import numpy as np
        scores = np.array([0.3, 0.9, 0.6])
        ranks = ranks_from_scores(scores)
        # Highest score gets rank 1
        assert ranks[1] == 1
        assert ranks[2] == 2
        assert ranks[0] == 3

    def test_clean_text(self):
        from utils.helpers import clean_text
        text = "  Hello  World!  "
        result = clean_text(text)
        assert result == "hello world"

    def test_safe_divide(self):
        from utils.helpers import safe_divide
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=99.0) == 99.0

    def test_save_load_json(self, tmp_path):
        from utils.helpers import save_json, load_json
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_truncate_text(self):
        from utils.helpers import truncate_text
        text = "a" * 600
        result = truncate_text(text, max_chars=512)
        assert len(result) <= 515  # 512 + "..."
        assert result.endswith("...")

    def test_flatten_list(self):
        from utils.helpers import flatten_list
        nested = [[1, 2], [3, 4], [5]]
        result = flatten_list(nested)
        assert result == [1, 2, 3, 4, 5]
