Running the Sustainable HR Tool
================================

This document lists the quick commands to validate data, train models, and run predictions. Use a virtual environment for reproducibility.

Prerequisites

- Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Validate data only

Run lightweight validation (no training). This writes `validation_report.json` to the model directory.

```powershell
# Validate all CSVs in the data/raw folder and save results to a disposable folder
python -m src.train --input data/raw --model-dir scratch_models --validate-only
```

Disposable helper

There is a disposable PowerShell helper under `scratch/` to run validation and optionally delete the scratch output:

```powershell
# Run validation and leave outputs
.\scratch\validate_and_clean.ps1 -input data/raw -modeldir scratch_models
# Run validation and delete the scratch_models folder afterwards
.\scratch\validate_and_clean.ps1 -input data/raw -modeldir scratch_models -clean
```

Train models

Run the full pipeline (validation -> CV -> final fit). Models, CV results and reports will be saved to the model directory.

```powershell
python -m src.train --input data/raw --model-dir scratch_models
```

Use synthetic data for quick experiments:

```powershell
python -m src.train --synthetic --model-dir scratch_models
```

Load a saved model and make predictions (example script snippet)

```python
import joblib
import pandas as pd
model = joblib.load('scratch_models/hr_model.joblib')
df = pd.read_csv('data/raw/Activities.csv')
# Preprocess and feature-engineer as in the pipeline
from src import data_loader, preprocessing, features
df = data_loader.load_data('data/raw/Activities.csv')
df_proc = preprocessing.preprocess_data(df, drop_na_columns=['distance_km'])
df_feat = features.compute_features(df_proc)
# Select features used by the model (example)
feature_cols = [c for c in df_feat.select_dtypes(include=['number']).columns if c != 'avg_hr' and not c.startswith('predicted_')]
preds = model.predict(df_feat[feature_cols])
df_feat['predicted_sustainable_hr'] = preds
```

Cleaning up

Remove disposable outputs and scripts by deleting the `scratch/` folder and any `scratch_models` directory you created:

```powershell
rmdir /s /q scratch_models
rmdir /s /q scratch
```

Files of interest

- Training CLI: [src/train.py](src/train.py)
- Data loading & parsing: [src/data_loader.py](src/data_loader.py)
- Preprocessing: [src/preprocessing.py](src/preprocessing.py)
- Feature engineering: [src/features.py](src/features.py)
- Models: [src/modelling.py](src/modelling.py)
- Validation helpers: [src/validation.py](src/validation.py)
- CV plotting: [src/evaluation_visuals.py](src/evaluation_visuals.py)
