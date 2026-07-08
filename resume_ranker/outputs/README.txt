This directory stores generated outputs.

outputs/
├── models/      ← trained model files (.joblib, .pkl) — auto-created on first run
├── plots/       ← SHAP, LIME, skill analysis PNG charts
├── rankings/    ← rankings.csv results
└── reports/     ← evaluation metrics JSON, fairness report, ablation study CSV

All files here are auto-generated when you run the pipeline.
Do NOT commit the model .joblib files to Git — they are large binary files.
Use Git LFS or DVC for model versioning if needed.
