"""
Streamlit web application for the Resume Ranking System.
Provides interactive UI for uploading resumes, viewing rankings,
and exploring SHAP/LIME explanations.
"""

import sys
import json
import tempfile
from pathlib import Path

# Must be first Streamlit call
import streamlit as st

st.set_page_config(
    page_title="Resume Ranker AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ACTIVE_MODEL, PLOTS_DIR, RANKINGS_DIR
from utils.logger import log


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 20px;
        padding: 4px 12px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .rank-1 { background-color: #FFD700; color: black; border-radius: 50%; padding: 4px 10px; font-weight: bold; }
    .rank-2 { background-color: #C0C0C0; color: black; border-radius: 50%; padding: 4px 10px; font-weight: bold; }
    .rank-3 { background-color: #CD7F32; color: white; border-radius: 50%; padding: 4px 10px; font-weight: bold; }
    .skill-match { background: #d4edda; color: #155724; border-radius: 4px; padding: 2px 8px; margin: 2px; display: inline-block; font-size: 0.8rem; }
    .skill-miss { background: #f8d7da; color: #721c24; border-radius: 4px; padding: 2px 8px; margin: 2px; display: inline-block; font-size: 0.8rem; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 12px; text-align: center; }
    .stProgress .st-bo { background-color: #667eea; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=70)
    st.title("⚙️ Configuration")

    selected_model = st.selectbox(
        "Ranking Model",
        options=["bert", "tfidf", "logistic_regression", "random_forest"],
        index=0,
        help="BERT: Semantic similarity (recommended)\nTF-IDF: Fast keyword baseline\nML Models: Feature-based ranking",
    )

    st.markdown("### Section Weights")
    skills_weight = st.slider("Skills Weight", 0.0, 1.0, 0.50, 0.05)
    exp_weight = st.slider("Experience Weight", 0.0, 1.0, 0.30, 0.05)
    edu_weight = st.slider("Education Weight", 0.0, 1.0, 0.15, 0.05)
    cert_weight = st.slider("Certifications Weight", 0.0, 1.0, 0.05, 0.05)

    total = skills_weight + exp_weight + edu_weight + cert_weight
    if abs(total - 1.0) > 0.01:
        st.warning(f"⚠ Weights sum to {total:.2f}. Normalizing automatically.")
        if total > 0:
            norm = total
            skills_weight /= norm
            exp_weight /= norm
            edu_weight /= norm
            cert_weight /= norm

    st.markdown("### Score Combination")
    sem_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.45, 0.05)
    skill_comb_weight = st.slider("Skill Overlap", 0.0, 1.0, 0.40, 0.05)
    sec_comb_weight = st.slider("Section Weighted", 0.0, 1.0, 0.15, 0.05)

    st.markdown("---")
    run_explainability = st.checkbox("Generate Explanations", value=True)
    run_fairness = st.checkbox("Fairness Analysis", value=True)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎯 Resume Ranking System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Skill-Based Candidate Ranking with Explainable AI</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📋 Rank Resumes", "🔍 Explanations", "📊 Evaluation", "📘 About"])

# ── Tab 1: Input & Ranking ────────────────────────────────────────────────────
with tab1:
    col_jd, col_res = st.columns([1, 1])

    with col_jd:
        st.subheader("📄 Job Description")
        jd_input_method = st.radio("Input method", ["Type / Paste", "Upload File"], horizontal=True)

        job_text = ""
        if jd_input_method == "Type / Paste":
            job_text = st.text_area(
                "Paste job description here",
                height=300,
                placeholder="Senior ML Engineer...\n\nRequired skills:\n- Python, PyTorch, TensorFlow...",
            )
        else:
            jd_file = st.file_uploader("Upload job description", type=["txt", "pdf"])
            if jd_file:
                if jd_file.name.endswith(".txt"):
                    job_text = jd_file.read().decode("utf-8", errors="replace")
                else:
                    st.info("PDF JD parsing will extract text automatically")
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(jd_file.read())
                        tmp_path = tmp.name
                    from preprocessing.pdf_parser import parse_resume_file
                    job_text = parse_resume_file(Path(tmp_path))
                    os.unlink(tmp_path)
                st.text_area("Loaded JD", job_text[:500] + "...", height=100, disabled=True)

        # Load sample
        if st.button("📂 Load Sample JD"):
            sample_jd = Path(__file__).parent.parent / "data" / "sample" / "job_description.txt"
            if sample_jd.exists():
                job_text = sample_jd.read_text(encoding="utf-8")
                st.success("Sample job description loaded!")
                st.text_area("JD Preview", job_text[:400] + "...", height=100, disabled=True)
            else:
                st.error("Sample JD not found")

    with col_res:
        st.subheader("📁 Upload Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload resume files (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )

        use_sample = st.checkbox("Use sample resumes (demo mode)")

        if uploaded_resumes:
            st.success(f"✅ {len(uploaded_resumes)} resume(s) uploaded")
            for f in uploaded_resumes:
                st.text(f"  📎 {f.name}")
        elif use_sample:
            sample_dir = Path(__file__).parent.parent / "data" / "sample"
            sample_files = list(sample_dir.glob("resume_*.txt"))
            st.info(f"Using {len(sample_files)} sample resumes from data/sample/")
            for f in sample_files:
                st.text(f"  📎 {f.name}")

    # Run button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        run_btn = st.button("🚀 RANK CANDIDATES", type="primary", use_container_width=True)

    if run_btn:
        if not job_text.strip():
            st.error("❌ Please enter a job description")
            st.stop()

        if not uploaded_resumes and not use_sample:
            st.error("❌ Please upload resumes or enable sample mode")
            st.stop()

        # Build file paths
        resume_paths = []
        tmp_dirs = []

        if use_sample and not uploaded_resumes:
            sample_dir = Path(__file__).parent.parent / "data" / "sample"
            resume_paths = list(sample_dir.glob("resume_*.txt"))
            if not resume_paths:
                st.error("No sample resumes found in data/sample/")
                st.stop()

        with st.spinner("🔄 Parsing and ranking resumes..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                if uploaded_resumes:
                    for uf in uploaded_resumes:
                        fpath = tmp_path / uf.name
                        with open(fpath, "wb") as f:
                            f.write(uf.read())
                        resume_paths.append(fpath)

                # Update config weights
                import config
                config.SECTION_WEIGHTS = {
                    "skills": skills_weight,
                    "experience": exp_weight,
                    "education": edu_weight,
                    "certifications": cert_weight,
                }
                config.SCORE_COMBINATION_WEIGHTS = {
                    "semantic_similarity": sem_weight,
                    "skill_overlap": skill_comb_weight,
                    "section_weighted": sec_comb_weight,
                }
                config.ACTIVE_MODEL = selected_model

                try:
                    from scoring.scorer import ResumeScorer, parse_resume, parse_job_description
                    from explainability.explainer_pipeline import ExplainabilityPipeline

                    job = parse_job_description(job_text)
                    resumes_parsed = []
                    parse_errors = []
                    for rp in resume_paths:
                        try:
                            r = parse_resume(rp)
                            resumes_parsed.append(r)
                        except Exception as e:
                            parse_errors.append(str(rp.name))

                    if parse_errors:
                        st.warning(f"⚠ Could not parse: {', '.join(parse_errors)}")

                    if not resumes_parsed:
                        st.error("❌ No resumes could be parsed")
                        st.stop()

                    scorer = ResumeScorer(model_name=selected_model)
                    ranked = scorer.score_resumes(job, resumes_parsed)
                    df = scorer.to_dataframe(ranked)

                    exp_data = {}
                    if run_explainability:
                        exp_pipeline = ExplainabilityPipeline(model_type=selected_model)
                        exp_pipeline.fit(ranked)
                        exp_data = exp_pipeline.explain_all(ranked, job, output_dir=PLOTS_DIR / "streamlit")

                    st.session_state["ranked"] = ranked
                    st.session_state["df"] = df
                    st.session_state["job"] = job
                    st.session_state["exp_data"] = exp_data

                except Exception as e:
                    st.error(f"❌ Ranking failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()

        st.success(f"✅ Ranked {len(ranked)} candidates using **{selected_model}** model!")

    # Display Results
    if "ranked" in st.session_state:
        ranked = st.session_state["ranked"]
        df = st.session_state["df"]
        job = st.session_state["job"]

        st.markdown("---")
        st.subheader("🏆 Ranking Results")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidates", len(ranked))
        m2.metric("Required Skills", len(job.required_skills))
        m3.metric("Top Score", f"{ranked[0].final_score:.3f}" if ranked else "N/A")
        m4.metric("Model Used", selected_model.upper())

        # Skill cloud
        if job.required_skills:
            st.markdown("**🎯 Required Skills:**")
            skills_html = " ".join(
                f'<span class="skill-match">{s}</span>' for s in job.required_skills
            )
            st.markdown(skills_html, unsafe_allow_html=True)

        st.markdown("---")

        # Ranking cards
        for resume in ranked:
            rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(resume.rank, f"#{resume.rank}")
            overlap = resume.skill_overlap_details

            with st.expander(
                f"{rank_emoji} **{resume.file_name}** — Score: {resume.final_score:.3f}",
                expanded=(resume.rank <= 2),
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Score", f"{resume.final_score:.3f}")
                c2.metric("BERT Similarity", f"{resume.bert_score:.3f}")
                c3.metric("Skill Match", f"{resume.skill_overlap_score:.3f}")
                c4.metric("Section Score", f"{resume.section_weighted_score:.3f}")

                col_skills, col_plot = st.columns([1, 1])
                with col_skills:
                    matched = overlap.get("matched", [])
                    missing = overlap.get("missing", [])

                    if matched:
                        st.markdown("**✅ Matched Skills:**")
                        matched_html = " ".join(
                            f'<span class="skill-match">{s}</span>' for s in matched
                        )
                        st.markdown(matched_html, unsafe_allow_html=True)

                    if missing:
                        st.markdown("**❌ Missing Skills:**")
                        missing_html = " ".join(
                            f'<span class="skill-miss">{s}</span>' for s in missing[:10]
                        )
                        st.markdown(missing_html, unsafe_allow_html=True)

                with col_plot:
                    # Show analysis plot if generated
                    exp = st.session_state.get("exp_data", {}).get(resume.resume_id, {})
                    skill_plot = exp.get("skill_plot")
                    if skill_plot and Path(skill_plot).exists():
                        st.image(skill_plot, use_container_width=True)

                # Score progress bars
                st.markdown("**Score Breakdown:**")
                score_items = [
                    ("BERT Similarity", resume.bert_score),
                    ("TF-IDF Similarity", resume.tfidf_score),
                    ("Skill Overlap", resume.skill_overlap_score),
                    ("Section Weighted", resume.section_weighted_score),
                ]
                for label, score in score_items:
                    st.write(f"{label}: {score:.3f}")
                    st.progress(min(float(score), 1.0))

        # CSV Download
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Rankings CSV",
            csv,
            "rankings.csv",
            "text/csv",
        )


# ── Tab 2: Explanations ───────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Explainable AI Insights")

    if "ranked" not in st.session_state:
        st.info("👆 Please run the ranking first (Tab 1)")
    else:
        ranked = st.session_state["ranked"]
        exp_data = st.session_state.get("exp_data", {})

        selected_candidate = st.selectbox(
            "Select candidate to explain",
            options=[f"#{r.rank} - {r.file_name}" for r in ranked],
        )

        if selected_candidate:
            rank_num = int(selected_candidate.split(" - ")[0].replace("#", ""))
            resume = next(r for r in ranked if r.rank == rank_num)
            exp = exp_data.get(resume.resume_id, {})

            st.markdown(f"### Analysis: {resume.file_name}")
            st.markdown(f"**Rank:** #{resume.rank} | **Score:** {resume.final_score:.4f}")

            # Narrative
            narrative = exp.get("narrative", "")
            if narrative:
                st.markdown("**🗣 Why this ranking:**")
                for line in narrative.split("\n"):
                    if line.strip():
                        st.markdown(line)

            col_shap, col_lime = st.columns(2)

            with col_shap:
                st.markdown("#### SHAP Local Explanation")
                shap_plot = exp.get("shap_local_plot")
                if shap_plot and Path(shap_plot).exists():
                    st.image(shap_plot, caption="SHAP: Feature contributions to ranking score")
                else:
                    # Show feature importance table instead
                    shap_vals = exp.get("shap_local_values", [])
                    if shap_vals:
                        st.markdown("**Feature Contributions:**")
                        for feat, val in shap_vals[:10]:
                            color = "🟢" if val > 0 else "🔴"
                            st.write(f"{color} {feat}: {val:+.4f}")
                    else:
                        st.info("SHAP: Available when using ML models (logistic_regression/random_forest)")

            with col_lime:
                st.markdown("#### LIME Local Explanation")
                lime_plot = exp.get("lime_plot")
                if lime_plot and Path(lime_plot).exists():
                    st.image(lime_plot, caption="LIME: Local feature contributions")
                else:
                    lime_text = exp.get("lime_text", "")
                    if lime_text:
                        st.code(lime_text)
                    else:
                        st.info("LIME: Available when using ML models (logistic_regression/random_forest)")

            # Skill analysis plot
            skill_plot = exp.get("skill_plot")
            if skill_plot and Path(skill_plot).exists():
                st.markdown("#### Comprehensive Skill Analysis")
                st.image(skill_plot, use_container_width=True)

            # Global SHAP
            global_exp = exp_data.get("_global", {})
            global_plot = global_exp.get("shap_summary_plot")
            if global_plot and Path(global_plot).exists():
                st.markdown("#### Global Feature Importance (SHAP)")
                st.image(global_plot, use_container_width=True)


# ── Tab 3: Evaluation ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Evaluation & Fairness Analysis")

    if "ranked" not in st.session_state:
        st.info("👆 Please run the ranking first (Tab 1)")
    else:
        ranked = st.session_state["ranked"]
        df = st.session_state["df"]

        col_eval, col_fair = st.columns(2)

        with col_eval:
            st.markdown("#### Evaluation Metrics")
            st.markdown("Enter ground truth relevant candidates:")
            rel_input = st.text_input(
                "Relevant resume IDs (comma-separated file stems)",
                placeholder="resume_alice_chen, resume_eva_rodriguez",
            )
            if rel_input:
                from evaluation.metrics import compute_ranking_metrics, format_metrics_report
                rel_ids = [r.strip() for r in rel_input.split(",")]
                ranked_ids = [r.resume_id for r in ranked]
                metrics = compute_ranking_metrics(ranked_ids, rel_ids)

                st.markdown("**Ranking Metrics:**")
                for k, v in metrics.items():
                    st.metric(k.upper(), f"{v:.4f}")

        with col_fair:
            st.markdown("#### Fairness Analysis")
            if run_fairness:
                from evaluation.fairness import run_fairness_analysis, format_fairness_report
                fairness = run_fairness_analysis(df)

                for obs in fairness.get("overall_observations", []):
                    st.write(obs)

                fair_plot = fairness.get("distribution_plot")
                if fair_plot and Path(fair_plot).exists():
                    st.image(fair_plot, use_container_width=True)
            else:
                st.info("Enable Fairness Analysis in sidebar")

        # Rankings table
        st.markdown("---")
        st.markdown("#### Full Rankings Table")
        display_df = df[["rank", "file_name", "final_score", "bert_score",
                          "skill_overlap_score", "num_skills", "matched_skills"]].copy()
        st.dataframe(display_df, use_container_width=True)


# ── Tab 4: About ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📘 About This System")
    st.markdown("""
    ## Resume Ranking System with Explainable AI

    This system ranks resumes against job descriptions using multiple AI approaches:

    ### 🤖 Models
    | Model | Description | Best For |
    |-------|-------------|----------|
    | **BERT** | Semantic similarity via Sentence Transformers | Production ranking |
    | **TF-IDF** | Term frequency-based cosine similarity | Fast baseline |
    | **Logistic Regression** | ML on engineered features | Interpretable ranking |
    | **Random Forest** | Ensemble on features with feature importance | Feature analysis |

    ### 🔍 Explainability
    - **SHAP**: Global and local feature importance
    - **LIME**: Local interpretable explanations per candidate
    - **Skill Analysis**: Visual skill gap and match breakdown

    ### 📊 Evaluation Metrics
    - Precision@K, Recall@K, nDCG@K
    - Mean Reciprocal Rank (MRR)
    - Mean Average Precision (MAP)

    ### ⚖️ Fairness
    - Score distribution analysis
    - Rank variance detection
    - Disparate impact analysis (when group data available)

    ### 🏗️ Architecture
    ```
    Resume Files (PDF/TXT/DOCX)
          ↓
    PDF Parser → Text Cleaner → Section Extractor
          ↓
    Skill Extractor (Ontology-based)
          ↓
    Similarity Engine (TF-IDF + BERT)
          ↓
    Scorer (Combined Weighted Score)
          ↓
    Explainability Pipeline (SHAP + LIME)
          ↓
    Rankings + Reports + Visualizations
    ```
    """)
