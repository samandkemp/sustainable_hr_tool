"""Training pipeline
-------------------
Simple training entrypoint to run a baseline pipeline end-to-end.

Usage examples are in the module docstring and the project README.
"""

from pathlib import Path
import argparse
import sys

import pandas as pd
from . import data_loader
from . import preprocessing
from . import features
from . import modelling
from . import evaluation
from . import utils
from . import validation
from . import evaluation_visuals
from sklearn.linear_model import LinearRegression


def run_training(
    input_path: str = None,
    use_synthetic: bool = False,
    model_dir: str = "models",
    save_models: bool = True,
    validate_only: bool = False,
    auto_synthetic: bool = False,
):
    # Load data
    if use_synthetic:
        # Generate synthetic processed + raw-format files. Processed data is used
        # by the modelling pipeline; the raw-format CSV is useful as a template
        # for users to drop into `data/raw/` when collecting real activities.
        proc_path = Path("data/processed/processed_synthetic.csv")
        raw_path = Path("data/raw/Activities_synthetic.csv")
        df_proc, _ = utils.generate_synthetic_and_raw(processed_path=str(proc_path), raw_path=str(raw_path))
        df = df_proc
    else:
        # If the user did not specify input, prefer any processed CSVs under
        # `data/processed/` so sanitized data is used by default. Fall back to
        # `data/raw/` if no processed files are present.
        if input_path is None:
            proc_folder = Path("data/processed")
            raw_folder = Path("data/raw")
            if proc_folder.exists() and any(proc_folder.glob("*.csv")):
                df = data_loader.load_all_data(str(proc_folder))
            elif raw_folder.exists() and any(raw_folder.glob("*.csv")):
                df = data_loader.load_all_data(str(raw_folder))
            else:
                # No CSVs found; either auto-generate synthetic data, or prompt the user
                if auto_synthetic:
                    proc_path = Path("data/processed/processed_synthetic.csv")
                    raw_path = Path("data/raw/Activities_synthetic.csv")
                    df_proc, _ = utils.generate_synthetic_and_raw(processed_path=str(proc_path), raw_path=str(raw_path))
                    df = df_proc
                else:
                    # Interactive prompt when running in a TTY
                    if sys.stdin is not None and sys.stdin.isatty():
                        ans = input(
                            "No CSVs found in data/processed or data/raw. Generate synthetic data now? [y/N]: "
                        ).strip().lower()
                        if ans in ("y", "yes"):
                            proc_path = Path("data/processed/processed_synthetic.csv")
                            raw_path = Path("data/raw/Activities_synthetic.csv")
                            df_proc, _ = utils.generate_synthetic_and_raw(processed_path=str(proc_path), raw_path=str(raw_path))
                            df = df_proc
                        else:
                            print("No data available; aborting training.")
                            return None
                    else:
                        raise ValueError("No input provided and no CSVs found in data/processed or data/raw. Re-run with --synthetic or --auto-synthetic to generate synthetic data.")
        else:
            p = Path(input_path)
            if p.is_dir():
                df = data_loader.load_all_data(str(p))
            else:
                df = data_loader.load_data(str(p))

    # Prepare model dir early so validation artifacts can be saved there
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # Minimal validation & PII removal
    df_valid, val_report = validation.validate_and_coerce(df)
    validation_report_path = model_dir_path / "validation_report.json"
    import json
    with open(validation_report_path, "w") as fh:
        json.dump(val_report, fh, indent=2)
    print(f"Validation report saved to: {validation_report_path}")

    # Remove obvious PII columns
    df = validation.remove_pii_columns(df_valid)

    if validate_only:
        # Print a short human-readable validation summary and exit
        print("Validation summary:")
        print(f"  missing_columns: {val_report.get('missing_columns')}")
        print(f"  coerced_columns: {val_report.get('coerced_columns')}")
        print(f"  shape: {val_report.get('shape')}")
        return {"validation_report": val_report, "validation_path": str(validation_report_path)}

    # Preprocess
    df_proc = preprocessing.preprocess_data(df, drop_na_columns=["distance_km"])

    # Feature engineering
    df_feat = features.compute_features(df_proc)

    # Select numeric feature columns for HR model (exclude target)
    target = "avg_hr"
    numeric_cols = df_feat.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target and not c.startswith("predicted_")]

    # Cross-validate sustainable HR baseline before final fit

    try:
        cv_est = LinearRegression()
        cv_results = evaluation.cross_validate_model_from_df(cv_est, df_feat, feature_cols, target, cv=5)
        import json
        cv_report_path = model_dir_path / "cv_hr_results.json"
        with open(cv_report_path, "w") as fh:
            json.dump(cv_results, fh, indent=2)
        print(f"Cross-validation results saved to: {cv_report_path}")
        # Also create a simple plot summarising folds
        try:
            plot_path = model_dir_path / "cv_fold_errors.png"
            evaluation_visuals.plot_cv_metrics(cv_results, out_path=str(plot_path))
            print(f"CV fold errors plot saved to: {plot_path}")
        except Exception as e:
            print(f"Failed to create CV plot: {e}")
    except Exception as e:
        print(f"Cross-validation failed: {e}")

    # Train sustainable HR model
    hr_model_file = str(model_dir_path / "hr_model.joblib") if save_models else None
    hr_model, df_with_hr = modelling.fit_sustainable_hr_model(df_feat, features=feature_cols, target=target, model_file=hr_model_file)

    # Evaluate
    metrics = evaluation.evaluate_model(df_with_hr[target], df_with_hr["predicted_sustainable_hr"])

    print("HR model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Train pace-from-HR model (inverse)
    pace_model_file = str(model_dir_path / "pace_model.joblib") if save_models else None
    pace_model, df_with_pace = modelling.fit_pace_from_hr_model(df_feat, model_file=pace_model_file)

    # Evaluate pace model if available
    if "predicted_pace_min_km" in df_with_pace.columns:
        pace_metrics = evaluation.evaluate_model(df_with_pace["avg_pace_min_km"], df_with_pace["predicted_pace_min_km"])
        print("Pace model metrics:")
        for k, v in pace_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save a small report
    report_path = model_dir_path / "training_report.csv"
    report_df = pd.DataFrame([{"model": "hr_model", **metrics}])
    report_df.to_csv(report_path, index=False)
    print(f"Training report saved to: {report_path}")

    # Save predictions for easy inspection
    preds_path = model_dir_path / "predictions.csv"
    # Prefer the df that contains both HR and pace predictions when available
    to_save_df = df_with_pace if "predicted_pace_min_km" in df_with_pace.columns else df_with_hr
    try:
        to_save_df.to_csv(preds_path, index=False)
        print(f"Predictions saved to: {preds_path}")
    except Exception as e:
        print(f"Failed to save predictions: {e}")

    # Create residual visualisations saved to disk for quick inspection
    try:
        hr_plot_path = model_dir_path / "residuals_distribution_hr.png"
        evaluation_visuals.plot_residuals_distribution(df_with_hr[target], df_with_hr["predicted_sustainable_hr"], out_path=str(hr_plot_path))
        print(f"Residuals distribution saved to: {hr_plot_path}")
    except Exception as e:
        print(f"Failed to create residuals distribution plot: {e}")

    # Residuals vs distance (if available in hr predictions)
    try:
        if "distance_km" in df_with_hr.columns:
            rvf_path = model_dir_path / "residuals_vs_distance.png"
            evaluation_visuals.plot_residuals_vs_feature(df_with_hr, "avg_hr", "predicted_sustainable_hr", "distance_km", out_path=str(rvf_path))
            print(f"Residuals vs distance plot saved to: {rvf_path}")
    except Exception as e:
        print(f"Failed to create residuals vs feature plot: {e}")

    # Additional analytical plots for quick inspection
    try:
        pv_path = model_dir_path / "predicted_vs_actual.png"
        evaluation_visuals.plot_predicted_vs_actual(df_with_hr[target], df_with_hr["predicted_sustainable_hr"], out_path=str(pv_path))
        print(f"Predicted vs Actual plot saved to: {pv_path}")
    except Exception as e:
        print(f"Failed to create predicted vs actual plot: {e}")

    try:
        # Feature importance if linear model
        fi_path = model_dir_path / "feature_importance.png"
        evaluation_visuals.plot_feature_importance(hr_model, feature_cols, out_path=str(fi_path))
        print(f"Feature importance plot saved to: {fi_path}")
    except Exception as e:
        print(f"Feature importance plot skipped/failed: {e}")

    try:
        # Correlation heatmap for numeric features
        corr_cols = [c for c in df_feat.select_dtypes(include=["number"]).columns if c != target]
        if len(corr_cols) > 1:
            corr_path = model_dir_path / "feature_correlation.png"
            evaluation_visuals.plot_feature_correlation(df_feat, corr_cols, out_path=str(corr_path))
            print(f"Feature correlation plot saved to: {corr_path}")
    except Exception as e:
        print(f"Feature correlation plot skipped/failed: {e}")

    return {
        "hr_model": hr_model,
        "pace_model": pace_model,
        "metrics": {"hr": metrics, "pace": pace_metrics if 'pace_metrics' in locals() else None},
    }


def _cli():
    parser = argparse.ArgumentParser(description="Train baseline sustainable-HR and pace models.")
    parser.add_argument("--input", help="Path to activities CSV file or folder containing CSVs", default=None)
    parser.add_argument("--synthetic", help="Use synthetic data generator instead of real input", action="store_true")
    parser.add_argument("--auto-synthetic", help="Automatically generate synthetic data if no input CSVs are found", action="store_true")
    parser.add_argument("--model-dir", help="Directory to write models and reports", default="models")
    parser.add_argument("--no-save", help="Do not persist trained models", action="store_true")
    parser.add_argument("--validate-only", help="Only run validation and save a report, do not train", action="store_true")
    args = parser.parse_args()

    run_training(
        input_path=args.input,
        use_synthetic=args.synthetic,
        model_dir=args.model_dir,
        save_models=not args.no_save,
        validate_only=args.validate_only,
        auto_synthetic=args.auto_synthetic,
    )


if __name__ == "__main__":
    _cli()
