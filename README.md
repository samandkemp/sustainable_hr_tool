# Sustainable HR Tool

A Python tool for analysing running data from Garmin watches and predicting sustainable heart rate and pace across standard race distances.

Drop your Garmin `Activities.csv` into `data/raw/`, launch the dashboard, and explore your training history alongside model-backed predictions for 5k, 10k, half marathon, and marathon efforts.

## Quick start

**1. Create a virtual environment and install dependencies**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**2. Add your Garmin data**

Export your activities from Garmin Connect (the file is labelled `Activities.csv`) and place it in `data/raw/`. If you don't have a Garmin export yet, the tool can generate synthetic data automatically — see [docs/RUNNING.md](docs/RUNNING.md).

**3. Launch the dashboard**

```powershell
streamlit run dashboard.py
```

The dashboard loads your data, trains models, and opens a browser window with training history, analysis charts, and race predictions. Click **Reload data** in the sidebar whenever you add new CSV files.

**4. Or run the pipeline from the command line**

```powershell
# Use data in data/raw (or data/processed if present)
python -m src.train

# Generate synthetic data and train on it
python -m src.train --synthetic
```

See [docs/RUNNING.md](docs/RUNNING.md) for the full CLI reference.

## Features

- **Data ingestion** — parses Garmin Activities CSV exports; normalises column names and units automatically
- **Validation** — checks required columns, coerces types, removes known PII fields
- **Feature engineering** — elevation-adjusted pace, effort scoring, fatigue proxies, rolling training load
- **Modelling** — linear baseline for sustainable HR prediction and inverse pace prediction
- **Race predictions** — forward (pace → HR), inverse (HR → pace), and arithmetic (target time → required pace) for 5k, 10k, half marathon, and marathon
- **Dashboard** — Streamlit app with four tabs: Overview, Training History, Analysis, Race Predictions
- **Synthetic data** — generates realistic Garmin-style CSVs for testing the pipeline without real data

## Project layout

```
sustainable_hr_tool/
├── dashboard.py               # Streamlit dashboard entrypoint
├── main.py                    # Lightweight training launcher
├── config.yaml                # Pipeline configuration
├── requirements.txt
├── src/
│   ├── data_loader.py         # CSV ingestion and column normalisation
│   ├── preprocessing.py       # Cleaning and type coercion
│   ├── features.py            # Feature engineering
│   ├── modelling.py           # Model training
│   ├── evaluation.py          # Metrics and cross-validation
│   ├── evaluation_visuals.py  # Diagnostic plots
│   ├── race_predictor.py      # Race prediction logic
│   ├── targets.py             # Scenario builders for model input
│   ├── utils.py               # Synthetic data generation
│   └── validation.py          # Schema validation and PII removal
├── data/
│   ├── raw/                   # Drop Garmin CSV exports here
│   └── processed/             # Pre-processed CSVs (optional)
├── models/                    # Saved model files (git-ignored)
├── notebooks/                 # Exploratory Jupyter notebooks
├── tools/                     # Small utility scripts
└── docs/                      # Extended documentation
```

## Configuration

Pipeline settings live in `config.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `paths` | Input and output folder locations |
| `dummy_data` | Synthetic data generation parameters |
| `preprocessing` | Columns to require and normalise |
| `features` | Rolling training load window size |
| `modelling` | Target columns and random seed |
| `evaluation` | Cross-validation folds and metrics |
| `race_prediction` | Race distances in km |
