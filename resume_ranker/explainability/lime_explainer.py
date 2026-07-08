"""
LIME-based local explanations for individual resume rankings.
Explains why a specific resume was ranked the way it was.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import log
import config


class LIMEExplainer:
    """
    LIME explainer for resume ranking decisions.
    Provides local, interpretable explanations for each candidate.
    """

    def __init__(
        self,
        feature_names: List[str] = None,
        num_features: int = None,
        num_samples: int = None,
    ):
        from models.ml_models import FEATURE_NAMES
        self.feature_names = feature_names or FEATURE_NAMES
        self.num_features = num_features or config.LIME_NUM_FEATURES
        self.num_samples = num_samples or config.LIME_NUM_SAMPLES
        self._explainer = None

    def _build_explainer(self, training_data: np.ndarray):
        """Build LIME tabular explainer."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            self._explainer = LimeTabularExplainer(
                training_data=training_data,
                feature_names=self.feature_names,
                mode="regression",
                discretize_continuous=True,
                verbose=False,
            )
            log.info("LIME explainer built successfully")
        except Exception as e:
            log.error(f"Failed to build LIME explainer: {e}")

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        training_data: np.ndarray = None,
        resume_id: str = "resume",
    ) -> Optional[Dict]:
        """
        Generate LIME explanation for a single resume.
        
        Args:
            instance: Feature vector for the resume
            predict_fn: Function that takes feature matrix and returns scores
            training_data: Background training data
            resume_id: Identifier for the resume
            
        Returns:
            Dict with explanation details
        """
        if training_data is None:
            # Generate synthetic background if not provided
            training_data = np.random.randn(100, len(self.feature_names))
            training_data = np.abs(training_data)

        if self._explainer is None:
            self._build_explainer(training_data)

        if self._explainer is None:
            return None

        try:
            explanation = self._explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=self.num_features,
                num_samples=self.num_samples,
            )

            # Extract feature contributions
            feature_contributions = explanation.as_list()

            result = {
                "resume_id": resume_id,
                "feature_contributions": feature_contributions,
                "local_prediction": explanation.local_pred[0] if hasattr(explanation, "local_pred") else None,
                "intercept": explanation.intercept[0] if hasattr(explanation, "intercept") else None,
                "score": explanation.score if hasattr(explanation, "score") else None,
                "positive_features": [(f, v) for f, v in feature_contributions if v > 0],
                "negative_features": [(f, v) for f, v in feature_contributions if v < 0],
            }

            return result
        except Exception as e:
            log.error(f"LIME explain_instance failed for {resume_id}: {e}")
            return None

    def explain_text(
        self,
        resume_text: str,
        job_text: str,
        score_fn: Callable,
        resume_id: str = "resume",
    ) -> Optional[Dict]:
        """
        LIME explanation for text-based scoring (TF-IDF features).
        Uses LIME text explainer.
        """
        try:
            from lime.lime_text import LimeTextExplainer

            text_explainer = LimeTextExplainer(
                class_names=["low_match", "high_match"],
                verbose=False,
            )

            def predict_wrapper(texts):
                from preprocessing.text_cleaner import preprocess_for_tfidf
                from models.tfidf_model import TFIDFModel
                tfidf = TFIDFModel()
                all_texts = [job_text] + texts
                scores = tfidf.score(preprocess_for_tfidf(job_text),
                                    [preprocess_for_tfidf(t) for t in texts])
                return np.column_stack([1 - scores, scores])

            explanation = text_explainer.explain_instance(
                resume_text,
                predict_wrapper,
                num_features=self.num_features,
                num_samples=self.num_samples,
            )

            return {
                "resume_id": resume_id,
                "type": "text",
                "feature_contributions": explanation.as_list(label=1),
            }
        except Exception as e:
            log.error(f"LIME text explanation failed: {e}")
            return None

    def plot_explanation(
        self,
        explanation: Dict,
        resume_id: str = None,
        save_path: Path = None,
    ) -> Optional[Path]:
        """Plot LIME feature contributions for a single resume."""
        if not explanation or "feature_contributions" not in explanation:
            return None

        try:
            rid = resume_id or explanation.get("resume_id", "resume")
            contributions = explanation["feature_contributions"]

            if not contributions:
                return None

            names = [c[0] for c in contributions]
            values = [c[1] for c in contributions]

            fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4 + 1)))

            colors = ["#27ae60" if v > 0 else "#e74c3c" for v in values]
            ax.barh(range(len(names)), values, color=colors, alpha=0.85, edgecolor="white")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()
            ax.axvline(0, color="black", linewidth=1.0)
            ax.set_xlabel("Feature Contribution to Score", fontsize=10)
            ax.set_title(f"LIME Local Explanation: {rid}", fontsize=12, fontweight="bold")

            # Annotate positive/negative
            ax.text(
                0.02, 0.98, "Green = boosts score  |  Red = lowers score",
                transform=ax.transAxes, fontsize=8, va="top", color="gray"
            )

            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()

            save_path = save_path or (config.PLOTS_DIR / f"lime_{rid}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"LIME plot saved: {save_path}")
            return save_path
        except Exception as e:
            log.error(f"LIME plot failed: {e}")
            return None

    def format_explanation_text(self, explanation: Dict) -> str:
        """Format explanation as human-readable text."""
        if not explanation:
            return "No explanation available."

        lines = [f"=== LIME Explanation: {explanation.get('resume_id', 'Resume')} ==="]

        pos = explanation.get("positive_features", [])
        neg = explanation.get("negative_features", [])

        if pos:
            lines.append("\n✅ POSITIVE contributors (why candidate ranked HIGHER):")
            for feat, val in sorted(pos, key=lambda x: -x[1]):
                lines.append(f"  + {feat}: +{val:.4f}")

        if neg:
            lines.append("\n❌ NEGATIVE contributors (why candidate ranked LOWER):")
            for feat, val in sorted(neg, key=lambda x: x[1]):
                lines.append(f"  - {feat}: {val:.4f}")

        if explanation.get("local_prediction") is not None:
            lines.append(f"\n📊 Local prediction score: {explanation['local_prediction']:.4f}")

        return "\n".join(lines)
