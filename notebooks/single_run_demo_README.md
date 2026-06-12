single_run_demo.ipynb — Quick Notebook Guide

Purpose
- Walk through the common Load → Validate → Preprocess → Feature → Train → Evaluate → Inspect workflow for Garmin running activities.

Sections / Cells
- Setup & imports: creates `OUT_DIR`, imports `src` helpers, and lists requirements (see `requirements.txt`).
- Load data: loads CSVs from `data/raw/` using `src.data_loader.load_all_data()`.
- Validate data: runs `src.validation.validate_and_coerce()` and writes a small report to `OUT_DIR`.
- Preprocess & features: calls `src.preprocessing.preprocess_data()` then `src.features.compute_features()`.
- Select features & train baseline: selects numeric features, fits `src.modelling.fit_sustainable_hr_model()` and writes `hr_model.joblib` to `OUT_DIR`.
- Cross-validation (quick): runs `src.evaluation.cross_validate_model_from_df()` (default 5 folds), saves `cv_hr_results.json` and `cv_fold_errors.png`.
- Save predictions and visualisations: saves `predictions.csv` and diagnostic PNGs (predicted_vs_actual, residuals_distribution, feature_importance).
- Scenario prediction example: builds a median-based scenario via `src.targets.build_scenario_template()` and predicts using the trained model.
- Next steps: short suggestions for improvements (more features, HPO, tests).

How to run
1. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Start Jupyter and open the notebook:

```powershell
python -m pip install jupyter
python -m jupyter notebook notebooks\single_run_demo.ipynb
```

Notes
- Outputs are written to `OUT_DIR` inside the notebook (defaults to `scratch_models` when set). Remove the folder to clean disposable artifacts.
- If your system Python lacks `pip` (MSYS/MinGW), run the `requirements.txt` install with the same interpreter you use to run Jupyter.
- See `docs/RUNNING.md` for a longer run guide and `scratch/predict_scenario.py` for a CLI-style example.

Contact
- If you'd like, I can open and run the notebook cells here, or add a rendered HTML summary for quick inspection.