"""Disposable scenario prediction helper.

Usage: adjust the `MODEL_DIR` and `SCENARIO` at the top and run.
This script is intentionally small and placed under `scratch/` so it can
be removed when no longer required.
"""
import joblib
from pathlib import Path
import pandas as pd
from src import data_loader, preprocessing, features, targets

# Configuration
MODEL_DIR = Path("scratch_models")
MODEL_FILE = MODEL_DIR / "hr_model.joblib"

# Example scenario: pace (min/km) and distance (km)
SCENARIO_PACE = 5.0
SCENARIO_DISTANCE = 10.0


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}. Run training first.")

    model = joblib.load(MODEL_FILE)

    # Load a sample of data to form a median-based template
    sample_path = Path("data/raw/Activities.csv")
    if sample_path.exists():
        df_sample = data_loader.load_data(str(sample_path))
    else:
        # fallback: use synthetic generator
        from src import utils
        df_sample = utils.generate_dummy_data(n_runs=50)

    scenario = targets.build_scenario_template(df_sample, SCENARIO_PACE, SCENARIO_DISTANCE)
    # light preprocess and features to match training pipeline
    scenario_proc = preprocessing.preprocess_data(scenario, compute_features=True)
    scenario_feat = features.compute_features(scenario_proc)

    # Select numeric feature columns used by the model (best-effort)
    feature_cols = [c for c in scenario_feat.select_dtypes(include=["number"]).columns if c != "avg_hr" and not c.startswith("predicted_")]
    preds = model.predict(scenario_feat[feature_cols])

    scenario_feat["predicted_sustainable_hr"] = preds
    print("Scenario input:")
    print(scenario_feat[feature_cols + ["predicted_sustainable_hr"]].to_dict(orient="records"))

    # Save to disk for inspection
    out_path = MODEL_DIR / "scenario_prediction.csv"
    scenario_feat.to_csv(out_path, index=False)
    print(f"Scenario prediction saved to: {out_path}")


if __name__ == "__main__":
    main()
