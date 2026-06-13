Tools
-----

This folder contains small vetted utilities intended to replace ad-hoc scripts
previously placed under `scratch/`.

Utilities
- `validate_and_clean.py`: Run validation (`src.train --validate-only`) and optionally remove the output folder. Use the same Python interpreter to ensure consistency.
- `predict_scenario.py`: Load a saved model from a model directory and predict a median-based scenario. Supports custom model filenames and CSV/JSON outputs.

Examples

Run validation and keep outputs:

```powershell
python tools/validate_and_clean.py --input data/raw --model-dir scratch_models
```

Run validation and delete outputs after completion:

```powershell
python tools/validate_and_clean.py --input data/raw --model-dir scratch_models --clean
```

Predict a scenario (default expects `hr_model.joblib` in `--model-dir`):

```powershell
python tools/predict_scenario.py --model-dir scratch_models --pace 5.0 --distance 10.0
```

Specify a custom model filename and JSON output:

```powershell
python tools/predict_scenario.py --model-dir scratch_models --model-file my_model.joblib --format json --output my_scenario.json --pace 5.0 --distance 10.0
```

Notes
- Run these tools using the same Python environment where the project dependencies are installed (use the venv created with `python -m venv .venv` and `python -m pip install -r requirements.txt`).

Install as editable package (optional)

To make the tools available as console scripts on your PATH, install the project in editable mode:

```powershell
python -m pip install -e .
```

This installs two commands:

- `srht-validate` -> runs `tools.validate_and_clean:main`
- `srht-predict`  -> runs `tools.predict_scenario:main`

Uninstall with:

```powershell
python -m pip uninstall sustainable_hr_tool
```
