# single_run_demo.ipynb — Notebook Guide

An end-to-end walkthrough of the analysis pipeline using Garmin running activity data. Covers data loading, validation, feature engineering, model training, cross-validation, and race predictions.

## Sections

| Section | Description |
|---|---|
| Setup & imports | Creates the output directory and imports `src` helpers |
| Load data | Loads CSVs from `data/raw/` using `data_loader.load_all_data()` |
| Validate data | Runs `validation.validate_and_coerce()` and saves a report to the output directory |
| Preprocess & features | Calls `preprocessing.preprocess_data()` then `features.compute_features()` |
| Select features & train | Fits `modelling.fit_sustainable_hr_model()` and saves `hr_model.joblib` |
| Cross-validation | Runs 5-fold CV via `evaluation.cross_validate_model_from_df()` and saves a fold-error plot |
| Save predictions & plots | Writes `predictions.csv` and diagnostic PNGs to the output directory |
| Scenario prediction | Builds a median-based scenario and predicts HR using the trained model |
| Training history | Plots cumulative distance and rolling 7-run training load |
| Inverse pace model | Trains `modelling.fit_pace_from_hr_model()` and evaluates metrics |
| Race predictions | Generates forward, inverse, and target-time predictions via `race_predictor.race_report()` |

Outputs are written to `scratch_models/` by default. Delete the folder to remove disposable artefacts.

## How to run

1. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install jupyter
```

2. Open the notebook:

```powershell
python -m jupyter notebook notebooks\single_run_demo.ipynb
```

3. Replace the `data/raw/` placeholder with your own `Activities.csv` export from Garmin Connect, then run all cells.

## Notes

- If `data/raw/` is empty, generate synthetic data first: `python -m src.train --synthetic`.
- For an interactive dashboard view, run `streamlit run dashboard.py` from the project root.
- See [docs/RUNNING.md](../docs/RUNNING.md) for the full CLI reference.
