# Running the Sustainable HR Tool

Commands to validate data, train models, launch the dashboard, and run predictions. All commands assume the virtual environment is active.

## Prerequisites

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dashboard

The easiest way to explore your data is via the Streamlit dashboard:

```powershell
streamlit run dashboard.py
```

The dashboard automatically loads data from `data/processed/` or `data/raw/`, trains models if none are saved, and opens a browser window with four tabs: Overview, Training History, Analysis, and Race Predictions. Click **Reload data** in the sidebar after adding or updating CSV files.

## Training pipeline (CLI)

Run the full pipeline — validation → cross-validation → model fitting → diagnostic plots:

```powershell
# Use CSVs in data/raw (or data/processed if present)
python -m src.train

# Use synthetic data for a quick experiment
python -m src.train --synthetic

# Auto-generate synthetic data if no CSVs are found (non-interactive)
python -m src.train --auto-synthetic --model-dir scratch_models
```

Models, CV results, and diagnostic plots are saved to `--model-dir` (defaults to `models/`).

## Validate only

Run schema validation without training. Writes `validation_report.json` to the model directory.

```powershell
python -m src.train --input data/raw --model-dir scratch_models --validate-only
```

## Data folders

| Folder | Purpose |
|---|---|
| `data/raw/` | Drop Garmin `Activities.csv` exports here |
| `data/processed/` | Pre-processed CSVs; takes priority over `data/raw/` when no `--input` is given |
| `models/` | Saved `.joblib` model files (persisted between runs) |
| `scratch_models/` | Disposable outputs for quick experiments (git-ignored) |

The `--synthetic` flag writes a processed CSV to `data/processed/processed_synthetic.csv` and a Garmin-style raw CSV to `data/raw/Activities_synthetic.csv`. If a real `Activities.csv` is present in `data/raw/`, its column layout is used as the template for the synthetic export.

## Behaviour when no data is present

- The CLI prefers CSVs in `data/processed/`, falling back to `data/raw/`.
- If neither folder contains CSVs, the CLI will prompt interactively to generate synthetic data.
- Use `--auto-synthetic` to skip the prompt and generate synthetic data automatically.

## Loading a saved model

```python
import joblib
from src import data_loader, preprocessing, features

model = joblib.load('models/hr_model.joblib')
df = data_loader.load_data('data/raw/Activities.csv')
df_proc = preprocessing.preprocess_data(df, drop_na_columns=['distance_km'])
df_feat = features.compute_features(df_proc)

feature_cols = [
    c for c in df_feat.select_dtypes(include=['number']).columns
    if c != 'avg_hr' and not c.startswith('predicted_')
]
df_feat['predicted_sustainable_hr'] = model.predict(df_feat[feature_cols])
```

## Install as editable package (optional)

To expose utility scripts as console commands on your PATH, install the project in editable mode from the project root:

```powershell
python -m pip install -e .
```

This provides two commands:

| Command | Description |
|---|---|
| `srht-validate` | Runs schema validation (`tools.validate_and_clean:main`) |
| `srht-predict` | Runs a scenario prediction (`tools.predict_scenario:main`) |

See [tools/README.md](../tools/README.md) for usage examples.

## Key source files

| File | Purpose |
|---|---|
| `src/train.py` | Training CLI |
| `src/data_loader.py` | CSV ingestion and column normalisation |
| `src/preprocessing.py` | Cleaning and type coercion |
| `src/features.py` | Feature engineering |
| `src/modelling.py` | Model training |
| `src/validation.py` | Schema validation and PII removal |
| `src/race_predictor.py` | Race prediction logic |
| `src/evaluation_visuals.py` | Diagnostic plots |
