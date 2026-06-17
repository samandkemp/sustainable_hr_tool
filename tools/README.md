# Tools

Small utility scripts for common pipeline operations. These replace ad-hoc scripts and are safe to include in automated workflows.

## Utilities

| Script | Description |
|---|---|
| `validate_and_clean.py` | Runs schema validation and optionally removes the output folder |
| `predict_scenario.py` | Loads a saved model and predicts a median-based scenario |

## Examples

**Validate data and keep outputs:**

```powershell
python tools/validate_and_clean.py --input data/raw --model-dir scratch_models
```

**Validate data and delete outputs afterwards:**

```powershell
python tools/validate_and_clean.py --input data/raw --model-dir scratch_models --clean
```

**Predict a scenario (defaults to `hr_model.joblib` in `--model-dir`):**

```powershell
python tools/predict_scenario.py --model-dir scratch_models --pace 5.0 --distance 10.0
```

**Predict with a custom model file and JSON output:**

```powershell
python tools/predict_scenario.py --model-dir scratch_models --model-file my_model.joblib --format json --output my_scenario.json --pace 5.0 --distance 10.0
```

## Install as console scripts (optional)

Install the project in editable mode to make these tools available as commands on your PATH:

```powershell
python -m pip install -e .
```

| Command | Runs |
|---|---|
| `srht-validate` | `tools.validate_and_clean:main` |
| `srht-predict` | `tools.predict_scenario:main` |

Uninstall with:

```powershell
python -m pip uninstall sustainable_hr_tool
```

> Run all tools using the same Python environment as the project (the `.venv` created with `python -m venv .venv`).
