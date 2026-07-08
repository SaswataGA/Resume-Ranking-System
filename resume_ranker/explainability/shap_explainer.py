"""
SHAP-based global and local explanations for resume ranking.
Explains which features contribute most to ranking decisions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import log
import config


class SHAPExplainer:
    """
    SHAP explainer for resume ranking models.
    Supports TreeExplainer (Random Forest) and LinearExplainer (Logistic Regression).
    Falls back to KernelExplainer for any model.
    """

    def __init__(self, model=None, model_type: str = None):
        self.model = model
        self.model_type = model_type or config.ACTIVE_MODEL
        self._explainer = None
        self._shap_values = None
        self._feature_names = None

    def _get_sklearn_model(self):
        """Extract sklearn model from pipeline."""
        if hasattr(self.model, "pipeline"):
            return self.model.pipeline.named_steps.get("clf")
        return self.model

    def _get_scaler(self):
        """Extract scaler from pipeline."""
        if hasattr(self.model, "pipeline"):
            return self.model.pipeline.named_steps.get("scaler")
        return None

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
    ) -> "SHAPExplainer":
        """
        Fit SHAP explainer on background data.
        
        Args:
            X: Feature matrix (background data)
            feature_names: Names of features
        """
        try:
            import shap
        except ImportError:
            log.error("SHAP not installed. Run: pip install shap")
            return self

        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        sklearn_model = self._get_sklearn_model()
        scaler = self._get_scaler()

        # Scale data if scaler available
        X_scaled = scaler.transform(X) if scaler is not None else X

        log.info(f"Fitting SHAP explainer for {self.model_type}")

        try:
            if self.model_type == "random_forest":
                self._explainer = shap.TreeExplainer(sklearn_model)
            elif self.model_type == "logistic_regression":
                background = shap.sample(X_scaled, min(50, len(X_scaled)))
                self._explainer = shap.LinearExplainer(
                    sklearn_model,
                    background,
                    feature_perturbation="interventional",
                )
            else:
                # Generic KernelExplainer
                if hasattr(self.model, "predict_proba"):
                    predict_fn = lambda x: self.model.predict_proba(x)
                else:
                    predict_fn = lambda x: self.model.predict(x)
                background = shap.sample(X_scaled, min(50, len(X_scaled)))
                self._explainer = shap.KernelExplainer(predict_fn, background)

            log.info("SHAP explainer fitted successfully")
        except Exception as e:
            log.error(f"SHAP fit failed: {e}")

        return self

    def explain(
        self,
        X: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for input data.
        
        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        if self._explainer is None:
            log.error("SHAP explainer not fitted. Call fit() first.")
            return None

        try:
            import shap
            scaler = self._get_scaler()
            X_scaled = scaler.transform(X) if scaler is not None else X

            shap_values = self._explainer.shap_values(X_scaled)

            # For binary classifiers, take positive class shap values
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]

            self._shap_values = shap_values
            return shap_values
        except Exception as e:
            log.error(f"SHAP explain failed: {e}")
            return None

    def get_global_importance(self) -> List[Tuple[str, float]]:
        """Get global feature importance from SHAP values."""
        if self._shap_values is None or self._feature_names is None:
            return []
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        importance = list(zip(self._feature_names, mean_abs.tolist()))
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance

    def get_local_explanation(
        self,
        shap_values_row: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Get local explanation for a single instance."""
        if self._feature_names is None:
            return []
        explanation = list(zip(self._feature_names, shap_values_row.tolist()))
        explanation.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanation

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: Path = None,
        title: str = "SHAP Summary Plot",
    ) -> Optional[Path]:
        """Generate SHAP summary plot (global feature importance)."""
        try:
            import shap

            if self._shap_values is None:
                self.explain(X)

            if self._shap_values is None:
                return None

            fig, ax = plt.subplots(figsize=(10, 6))

            # Manual summary plot using bar chart
            importance = self.get_global_importance()
            if not importance:
                return None

            names = [i[0] for i in importance[:config.SHAP_MAX_DISPLAY]]
            values = [i[1] for i in importance[:config.SHAP_MAX_DISPLAY]]

            colors = ["#e74c3c" if v > np.mean(values) else "#3498db" for v in values]
            bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()

            save_path = save_path or (config.PLOTS_DIR / "shap_summary.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"SHAP summary plot saved: {save_path}")
            return save_path
        except Exception as e:
            log.error(f"SHAP summary plot failed: {e}")
            return None

    def plot_local(
        self,
        shap_values_row: np.ndarray,
        resume_id: str,
        save_path: Path = None,
    ) -> Optional[Path]:
        """Generate local SHAP explanation plot for one resume."""
        try:
            if self._feature_names is None:
                return None

            explanation = self.get_local_explanation(shap_values_row)
            top = explanation[:config.SHAP_MAX_DISPLAY]

            names = [e[0] for e in top]
            values = [e[1] for e in top]

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#27ae60" if v > 0 else "#e74c3c" for v in values]
            ax.barh(range(len(names)), values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on ranking score)", fontsize=10)
            ax.set_title(f"SHAP Local Explanation: {resume_id}", fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()

            save_path = save_path or (config.PLOTS_DIR / f"shap_local_{resume_id}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"SHAP local plot saved: {save_path}")
            return save_path
        except Exception as e:
            log.error(f"SHAP local plot failed: {e}")
            return None
