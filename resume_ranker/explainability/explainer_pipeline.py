"""
Unified explainability pipeline combining SHAP and LIME.
Generates comprehensive explanations for ranked resumes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from scoring.scorer import ResumeRecord, JobRecord
from utils.logger import log
import config


class ExplainabilityPipeline:
    """
    Full explainability pipeline for the resume ranking system.
    
    Generates:
    1. SHAP global feature importance (across all resumes)
    2. SHAP local explanations (per resume)
    3. LIME local explanations (per resume)
    4. Skill gap analysis
    5. Human-readable explanations
    """

    def __init__(self, model=None, model_type: str = None):
        self.model = model
        self.model_type = model_type or config.ACTIVE_MODEL
        self.shap = SHAPExplainer(model=model, model_type=model_type)
        self.lime = LIMEExplainer()
        self._X = None
        self._shap_values = None

    def fit(
        self,
        resumes: List[ResumeRecord],
    ) -> "ExplainabilityPipeline":
        """Fit explainers on resume feature vectors."""
        feature_vectors = [r.feature_vector for r in resumes if r.feature_vector is not None]
        if not feature_vectors:
            log.warning("No feature vectors available for explainability fitting")
            return self

        self._X = np.stack(feature_vectors)

        if self.model is not None and self.model.is_fitted:
            from models.ml_models import FEATURE_NAMES
            self.shap.fit(self._X, feature_names=FEATURE_NAMES)

        return self

    def explain_all(
        self,
        resumes: List[ResumeRecord],
        job: JobRecord,
        output_dir: Path = None,
    ) -> Dict:
        """
        Generate complete explanations for all resumes.
        
        Returns:
            Dict mapping resume_id -> explanation dict
        """
        output_dir = output_dir or config.PLOTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        explanations = {}

        for resume in resumes:
            exp = self.explain_resume(resume, job, output_dir)
            explanations[resume.resume_id] = exp

        # Global SHAP if we have ML model
        if self._X is not None and self.model is not None and self.model.is_fitted:
            self._shap_values = self.shap.explain(self._X)
            shap_plot_path = self.shap.plot_summary(
                self._X,
                save_path=output_dir / "shap_global_summary.png",
            )
            explanations["_global"] = {
                "shap_summary_plot": str(shap_plot_path) if shap_plot_path else None,
                "global_importance": self.shap.get_global_importance(),
            }

        return explanations

    def explain_resume(
        self,
        resume: ResumeRecord,
        job: JobRecord,
        output_dir: Path = None,
    ) -> Dict:
        """Generate explanation for a single resume."""
        output_dir = output_dir or config.PLOTS_DIR
        explanation = {
            "resume_id": resume.resume_id,
            "file_name": resume.file_name,
            "rank": resume.rank,
            "final_score": resume.final_score,
            "score_breakdown": {
                "tfidf_score": resume.tfidf_score,
                "bert_score": resume.bert_score,
                "skill_overlap_score": resume.skill_overlap_score,
                "section_weighted_score": resume.section_weighted_score,
            },
            "skill_analysis": self._explain_skills(resume, job),
            "section_analysis": self._explain_sections(resume),
            "narrative": self._generate_narrative(resume, job),
            "shap_local_plot": None,
            "lime_plot": None,
            "lime_explanation": None,
        }

        # SHAP local explanation
        if (
            resume.feature_vector is not None
            and self.model is not None
            and self.model.is_fitted
            and self._shap_values is not None
        ):
            # Find index in shap values
            try:
                fv_idx = self._find_resume_index(resume)
                if fv_idx is not None:
                    shap_row = self._shap_values[fv_idx]
                    shap_plot = self.shap.plot_local(
                        shap_row,
                        resume_id=resume.resume_id,
                        save_path=output_dir / f"shap_local_{resume.resume_id}.png",
                    )
                    explanation["shap_local_plot"] = str(shap_plot) if shap_plot else None
                    explanation["shap_local_values"] = self.shap.get_local_explanation(shap_row)
            except Exception as e:
                log.warning(f"SHAP local explanation failed for {resume.resume_id}: {e}")

        # LIME explanation
        if resume.feature_vector is not None and self.model is not None and self.model.is_fitted:
            try:
                lime_exp = self._compute_lime_explanation(resume)
                if lime_exp:
                    explanation["lime_explanation"] = lime_exp
                    lime_plot = self.lime.plot_explanation(
                        lime_exp,
                        resume_id=resume.resume_id,
                        save_path=output_dir / f"lime_{resume.resume_id}.png",
                    )
                    explanation["lime_plot"] = str(lime_plot) if lime_plot else None
                    explanation["lime_text"] = self.lime.format_explanation_text(lime_exp)
            except Exception as e:
                log.warning(f"LIME explanation failed for {resume.resume_id}: {e}")

        # Fallback: skill-based visualization
        skill_plot = self._plot_skill_analysis(resume, job, output_dir)
        explanation["skill_plot"] = str(skill_plot) if skill_plot else None

        return explanation

    def _explain_skills(self, resume: ResumeRecord, job: JobRecord) -> Dict:
        """Detailed skill analysis."""
        details = resume.skill_overlap_details
        return {
            "candidate_skills": resume.skills,
            "required_skills": job.required_skills,
            "matched_skills": details.get("matched", []),
            "missing_skills": details.get("missing", []),
            "extra_skills": details.get("extra", []),
            "overlap_score": details.get("f1", 0.0),
            "jaccard": details.get("jaccard", 0.0),
            "precision": details.get("precision", 0.0),
            "recall": details.get("recall", 0.0),
        }

    def _explain_sections(self, resume: ResumeRecord) -> Dict:
        """Section-level score breakdown."""
        return {
            section: {
                "score": resume.section_scores.get(section, 0.0),
                "weight": config.SECTION_WEIGHTS.get(section, 0.0),
                "has_content": bool(resume.sections.get(section, "").strip()),
                "content_preview": resume.sections.get(section, "")[:200],
            }
            for section in config.SECTION_WEIGHTS
        }

    def _generate_narrative(self, resume: ResumeRecord, job: JobRecord) -> str:
        """Generate human-readable explanation of the ranking."""
        details = resume.skill_overlap_details
        matched = details.get("matched", [])
        missing = details.get("missing", [])

        narrative_parts = []

        # Rank statement
        if resume.rank == 1:
            narrative_parts.append(f"🥇 TOP CANDIDATE — ranked #1 with score {resume.final_score:.3f}")
        elif resume.rank <= 3:
            narrative_parts.append(f"⭐ Strong candidate — ranked #{resume.rank} with score {resume.final_score:.3f}")
        else:
            narrative_parts.append(f"Candidate ranked #{resume.rank} with score {resume.final_score:.3f}")

        # Semantic match
        if resume.bert_score >= 0.75:
            narrative_parts.append(f"✅ Excellent semantic match with the job description (BERT: {resume.bert_score:.3f})")
        elif resume.bert_score >= 0.5:
            narrative_parts.append(f"✔ Good semantic alignment (BERT: {resume.bert_score:.3f})")
        else:
            narrative_parts.append(f"⚠ Limited semantic overlap (BERT: {resume.bert_score:.3f})")

        # Skill match
        if matched:
            narrative_parts.append(f"✅ Matching skills: {', '.join(matched[:8])}" +
                                   (f" (+{len(matched)-8} more)" if len(matched) > 8 else ""))
        else:
            narrative_parts.append("❌ No matching required skills found")

        if missing:
            narrative_parts.append(f"⚠ Missing skills: {', '.join(missing[:5])}" +
                                   (f" (+{len(missing)-5} more)" if len(missing) > 5 else ""))

        # Experience
        if resume.years_experience_est > 0:
            narrative_parts.append(f"📋 Estimated experience: {resume.years_experience_est:.0f} years")

        return "\n".join(narrative_parts)

    def _compute_lime_explanation(self, resume: ResumeRecord) -> Optional[Dict]:
        """Compute LIME explanation for one resume."""
        if self._X is None:
            return None

        def predict_fn(X_arr):
            return self.model.predict_proba(X_arr)

        return self.lime.explain_instance(
            instance=resume.feature_vector,
            predict_fn=predict_fn,
            training_data=self._X,
            resume_id=resume.resume_id,
        )

    def _find_resume_index(self, resume: ResumeRecord) -> Optional[int]:
        """Find index of resume in self._X."""
        if self._X is None or resume.feature_vector is None:
            return None
        for i, row in enumerate(self._X):
            if np.allclose(row, resume.feature_vector, atol=1e-6):
                return i
        return None

    def _plot_skill_analysis(
        self,
        resume: ResumeRecord,
        job: JobRecord,
        output_dir: Path,
    ) -> Optional[Path]:
        """Create a comprehensive skill gap visualization."""
        try:
            details = resume.skill_overlap_details
            matched = details.get("matched", [])
            missing = details.get("missing", [])
            extra = details.get("extra", [])

            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(f"Resume Analysis: {resume.file_name}\nRank #{resume.rank} | Score: {resume.final_score:.3f}",
                         fontsize=13, fontweight="bold", y=0.98)

            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

            # Score breakdown bar chart
            ax1 = fig.add_subplot(gs[0, :2])
            score_labels = ["BERT\nSimilarity", "TF-IDF\nSimilarity", "Skill\nOverlap", "Section\nWeighted", "Final\nScore"]
            score_values = [
                resume.bert_score,
                resume.tfidf_score,
                resume.skill_overlap_score,
                resume.section_weighted_score,
                resume.final_score,
            ]
            bar_colors = ["#3498db", "#9b59b6", "#27ae60", "#f39c12", "#e74c3c"]
            bars = ax1.bar(score_labels, score_values, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1.2)
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel("Score (0-1)", fontsize=9)
            ax1.set_title("Score Breakdown", fontsize=10, fontweight="bold")
            for bar, val in zip(bars, score_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            ax1.grid(axis="y", alpha=0.3)

            # Section scores radar-style
            ax2 = fig.add_subplot(gs[0, 2])
            sections = list(resume.section_scores.keys())
            sec_scores = [resume.section_scores.get(s, 0.0) for s in sections]
            sec_colors = ["#27ae60" if s > 0.5 else "#e74c3c" if s < 0.3 else "#f39c12" for s in sec_scores]
            ax2.barh(sections, sec_scores, color=sec_colors, alpha=0.8)
            ax2.set_xlim(0, 1)
            ax2.set_title("Section Scores", fontsize=10, fontweight="bold")
            ax2.grid(axis="x", alpha=0.3)
            for i, v in enumerate(sec_scores):
                ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

            # Matched skills
            ax3 = fig.add_subplot(gs[1, 0])
            if matched:
                display_matched = matched[:12]
                y_pos = range(len(display_matched))
                ax3.barh(y_pos, [0.8] * len(display_matched), color="#27ae60", alpha=0.7)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(display_matched, fontsize=8)
                ax3.set_title(f"✅ Matched Skills ({len(matched)})", fontsize=9, fontweight="bold", color="#27ae60")
                ax3.set_xlim(0, 1)
                ax3.axis("off")
                for i, skill in enumerate(display_matched):
                    ax3.text(0.05, i, f"✓ {skill}", va="center", fontsize=8, color="white", fontweight="bold")
            else:
                ax3.text(0.5, 0.5, "No matching skills", ha="center", va="center", fontsize=10, color="gray")
                ax3.set_title("✅ Matched Skills (0)", fontsize=9, color="#27ae60")
            ax3.axis("off")

            # Missing skills
            ax4 = fig.add_subplot(gs[1, 1])
            if missing:
                display_missing = missing[:12]
                for i, skill in enumerate(display_missing):
                    ax4.text(0.05, 1 - (i + 1) / (len(display_missing) + 1),
                             f"✗ {skill}", va="center", fontsize=8, color="#c0392b")
                ax4.set_title(f"❌ Missing Skills ({len(missing)})", fontsize=9, fontweight="bold", color="#e74c3c")
            else:
                ax4.text(0.5, 0.5, "No missing skills!", ha="center", va="center", fontsize=10, color="#27ae60")
                ax4.set_title("❌ Missing Skills (0)", fontsize=9, color="#27ae60")
            ax4.axis("off")

            # Skill counts summary
            ax5 = fig.add_subplot(gs[1, 2])
            labels = ["Matched", "Missing", "Extra"]
            sizes = [len(matched), len(missing), len(extra)]
            explode = (0.05, 0.05, 0.05)
            if sum(sizes) > 0:
                pie_colors = ["#27ae60", "#e74c3c", "#3498db"]
                wedges, texts, autotexts = ax5.pie(
                    sizes, labels=labels, colors=pie_colors,
                    autopct="%1.0f%%", explode=explode,
                    startangle=90, textprops={"fontsize": 8},
                )
                ax5.set_title("Skill Distribution", fontsize=9, fontweight="bold")
            else:
                ax5.text(0.5, 0.5, "No skills found", ha="center", va="center")
                ax5.set_title("Skill Distribution", fontsize=9)

            save_path = output_dir / f"analysis_{resume.resume_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"Skill analysis plot saved: {save_path}")
            return save_path
        except Exception as e:
            log.error(f"Skill analysis plot failed for {resume.resume_id}: {e}")
            return None
