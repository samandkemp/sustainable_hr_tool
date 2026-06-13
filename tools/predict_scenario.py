"""Scenario prediction utility.

Usage example:
    python tools/predict_scenario.py --model-dir scratch_models --pace 5.0 --distance 10.0

This script loads a trained HR model from `--model-dir` and predicts HR for
a median-based scenario.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict scenario using saved model")
    parser.add_argument("--model-dir", default="scratch_models", help="Directory containing the model file")
    parser.add_argument("--model-file", default="hr_model.joblib", help="Model filename inside --model-dir")
    parser.add_argument("--pace", type=float, default=5.0, help="Target pace (min/km)")
    parser.add_argument("--distance", type=float, default=10.0, help="Distance (km)")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size to build median template")
    parser.add_argument("--output", default=None, help="Optional output path for scenario prediction")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")
    args = parser.parse_args()

    model_path = Path(args.model_dir) / args.model_file
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run training first.")
        return 2

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 3

    # Import pipeline pieces lazily to avoid import-time heavy deps when not needed
    try:
        from src import data_loader, preprocessing, features, targets
    except Exception as e:
        print("Failed to import project modules — ensure you run this with the project Python environment:", e)
        return 4

    # Prefer processed data
    proc_folder = Path("data/processed")
    raw_folder = Path("data/raw")
    if proc_folder.exists() and any(proc_folder.glob("*.csv")):
        df_sample = data_loader.load_all_data(str(proc_folder))
    elif raw_folder.exists() and any(raw_folder.glob("*.csv")):
        df_sample = data_loader.load_all_data(str(raw_folder))
    else:
        print("No data found in data/processed or data/raw — use synthetic generator to produce samples")
        from src import utils

        df_sample = utils.generate_dummy_data(n_runs=args.sample_size)

    scenario = targets.build_scenario_template(df_sample, pace_min_km=args.pace, distance_km=args.distance)
    scenario_proc = preprocessing.preprocess_data(scenario, compute_features=True)
    scenario_feat = features.compute_features(scenario_proc)

    feature_cols = [c for c in scenario_feat.select_dtypes(include=["number"]).columns if c != "avg_hr" and not c.startswith("predicted_")]
    preds = model.predict(scenario_feat[feature_cols])
    scenario_feat["predicted_sustainable_hr"] = preds

    out_path = Path(args.output) if args.output else Path(args.model_dir) / ("scenario_prediction." + args.format)

    if args.format == "csv":
        scenario_feat.to_csv(out_path, index=False)
    else:
        import json

        records = scenario_feat.to_dict(orient="records")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)

    print(f"Scenario prediction saved to: {out_path}")
    try:
        print(scenario_feat.to_dict(orient="records"))
    except Exception:
        # printing may fail for large frames; ignore
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
