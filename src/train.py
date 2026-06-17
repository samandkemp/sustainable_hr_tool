"""Training pipeline for model
-------------------
End-to-end pipeline: load → validate → preprocess → feature engineer → train → evaluate → report.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import data_loader, evaluation, evaluation_visuals, features, modelling, preprocessing, utils, validation


def _load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def _save_plot(fn, *args, out_path, label="plot", **kwargs):
    """Call a plot function and save the output; print a one-line status."""
    try:
        fn(*args, out_path=str(out_path), **kwargs)
        print(f"{label} saved to: {out_path}")
    except Exception as exc:
        print(f"{label} skipped: {exc}")


def run_training(
    input_path: str = None,
    use_synthetic: bool = False,
    model_dir: str = "models",
    save_models: bool = True,
    validate_only: bool = False,
    auto_synthetic: bool = False,
):
    # ── Load data ────────────────────────────────────────────────────────────
    if use_synthetic:
        proc_path = Path("data/processed/processed_synthetic.csv")
        raw_path = Path("data/raw/Activities_synthetic.csv")
        df, _ = utils.generate_synthetic_and_raw(processed_path=str(proc_path), raw_path=str(raw_path))
    else:
        if input_path is None:
            proc_folder = Path("data/processed")
            raw_folder = Path("data/raw")
            if proc_folder.exists() and any(proc_folder.glob("*.csv")):
                df = data_loader.load_all_data(str(proc_folder))
            elif raw_folder.exists() and any(raw_folder.glob("*.csv")):
                df = data_loader.load_all_data(str(raw_folder))
            elif auto_synthetic:
                proc_path = Path("data/processed/processed_synthetic.csv")
                raw_path = Path("data/raw/Activities_synthetic.csv")
                df, _ = utils.generate_synthetic_and_raw(processed_path=str(proc_path), raw_path=str(raw_path))
            elif sys.stdin is not None and sys.stdin.isatty():
                ans = input("No CSVs found. Generate synthetic data now? [y/N]: ").strip().lower()
                if ans in ("y", "yes"):
                    proc_path = Path("data/processed/processed_synthetic.csv")
                    raw_path = Path("data/raw/Activities_synthetic.csv")
                    df, _ = utils.generate_synthetic_and_raw(processed_path=str(proc_path), raw_path=str(raw_path))
                else:
                    print("No data available; aborting.")
                    return None
            else:
                raise ValueError(
                    "No input and no CSVs found in data/processed or data/raw. "
                    "Re-run with --synthetic or --auto-synthetic."
                )
        else:
            p = Path(input_path)
            df = data_loader.load_all_data(str(p)) if p.is_dir() else data_loader.load_data(str(p))

    # ── Validate ─────────────────────────────────────────────────────────────
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    df_valid, val_report = validation.validate_and_coerce(df)
    validation_report_path = model_dir_path / "validation_report.json"
    with open(validation_report_path, "w") as fh:
        json.dump(val_report, fh, indent=2)
    print(f"Validation report saved to: {validation_report_path}")

    df = validation.remove_pii_columns(df_valid)

    if validate_only:
        print("Validation summary:")
        for key in ("missing_columns", "coerced_columns", "shape"):
            print(f"  {key}: {val_report.get(key)}")
        return {"validation_report": val_report, "validation_path": str(validation_report_path)}

    # ── Preprocess & features ────────────────────────────────────
    df_proc = preprocessing.preprocess_data(df, drop_na_columns=["distance_km"])
    df_feat = features.compute_features(df_proc)

    target = "avg_hr"
    numeric_cols = df_feat.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target and not c.startswith("predicted_")]

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = _load_config()
    model_type = cfg.get("modelling", {}).get("model_type", "linear")
    compare_models = cfg.get("modelling", {}).get("compare_models", [model_type])
    cv_folds = cfg.get("evaluation", {}).get("cv_folds", 5)

    # ── Cross-validate: compare all configured model types ───────────────────
    try:
        comparison: dict = {}
        for mtype in compare_models:
            est = modelling.get_estimator(mtype)
            result = evaluation.cross_validate_model_from_df(
                est, df_feat, feature_cols, target, cv=cv_folds
            )
            comparison[mtype] = result

        # Rank by MAE (lower is better) and print summary
        ranked = sorted(comparison.items(), key=lambda kv: kv[1].get("mae_mean", float("inf")))
        print("\nModel comparison (HR — lower MAE is better):")
        for rank, (mtype, res) in enumerate(ranked, 1):
            print(f"  {rank}. {mtype:20s}  MAE={res.get('mae_mean', float('nan')):.4f}"
                  f"  R²={res.get('r2_mean', float('nan')):.4f}")
        print(f"\nSelected model: {model_type}")

        cv_comparison_path = model_dir_path / "cv_comparison.json"
        with open(cv_comparison_path, "w") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"CV comparison saved to: {cv_comparison_path}")

        # Save fold plot for the selected model
        if model_type in comparison:
            _save_plot(
                evaluation_visuals.plot_cv_metrics,
                comparison[model_type],
                out_path=model_dir_path / "cv_fold_errors.png",
                label="CV fold errors plot",
            )
    except Exception as exc:
        print(f"Cross-validation failed: {exc}")

    # ── Train models ─────────────────────────────────────────────────────────
    hr_model_file = str(model_dir_path / f"hr_model_{model_type}.joblib") if save_models else None
    hr_model, df_with_hr = modelling.fit_sustainable_hr_model(
        df_feat, features=feature_cols, target=target,
        model_file=hr_model_file, model_type=model_type,
    )
    metrics = evaluation.evaluate_model(df_with_hr[target], df_with_hr["predicted_sustainable_hr"])
    print("HR model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    pace_model_file = str(model_dir_path / f"pace_model_{model_type}.joblib") if save_models else None
    pace_model, df_with_pace = modelling.fit_pace_from_hr_model(
        df_feat, model_file=pace_model_file, model_type=model_type
    )

    pace_metrics: dict = {}
    if "predicted_pace_min_km" in df_with_pace.columns:
        pace_metrics = evaluation.evaluate_model(
            df_with_pace["avg_pace_min_km"], df_with_pace["predicted_pace_min_km"]
        )
        print("Pace model metrics:")
        for k, v in pace_metrics.items():
            print(f"  {k}: {v:.4f}")

    # ── Save reports & predictions ───────────────────────────────────────────
    pd.DataFrame([{"model": "hr_model", **metrics}]).to_csv(
        model_dir_path / "training_report.csv", index=False
    )
    print(f"Training report saved to: {model_dir_path / 'training_report.csv'}")

    save_df = df_with_pace if "predicted_pace_min_km" in df_with_pace.columns else df_with_hr
    try:
        save_df.to_csv(model_dir_path / "predictions.csv", index=False)
        print(f"Predictions saved to: {model_dir_path / 'predictions.csv'}")
    except Exception as exc:
        print(f"Failed to save predictions: {exc}")

    # ── Diagnostic plots ─────────────────────────────────────────────────────
    _save_plot(
        evaluation_visuals.plot_residuals_distribution,
        df_with_hr[target], df_with_hr["predicted_sustainable_hr"],
        out_path=model_dir_path / "residuals_distribution_hr.png",
        label="Residuals distribution",
    )
    if "distance_km" in df_with_hr.columns:
        _save_plot(
            evaluation_visuals.plot_residuals_vs_feature,
            df_with_hr, "avg_hr", "predicted_sustainable_hr", "distance_km",
            out_path=model_dir_path / "residuals_vs_distance.png",
            label="Residuals vs distance",
        )
    _save_plot(
        evaluation_visuals.plot_predicted_vs_actual,
        df_with_hr[target], df_with_hr["predicted_sustainable_hr"],
        out_path=model_dir_path / "predicted_vs_actual.png",
        label="Predicted vs Actual",
    )
    _save_plot(
        evaluation_visuals.plot_feature_importance,
        hr_model, feature_cols,
        out_path=model_dir_path / "feature_importance.png",
        label="Feature importance",
    )
    corr_cols = [c for c in df_feat.select_dtypes(include=["number"]).columns if c != target]
    if len(corr_cols) > 1:
        _save_plot(
            evaluation_visuals.plot_feature_correlation,
            df_feat, corr_cols,
            out_path=model_dir_path / "feature_correlation.png",
            label="Feature correlation",
        )

    return {
        "hr_model": hr_model,
        "pace_model": pace_model,
        "metrics": {"hr": metrics, "pace": pace_metrics},
    }


def _cli():
    parser = argparse.ArgumentParser(description="Train baseline sustainable-HR and pace models.")
    parser.add_argument("--input", help="Path to activities CSV or folder", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--auto-synthetic", action="store_true", help="Auto-generate synthetic data if no CSVs found")
    parser.add_argument("--model-dir", default="models", help="Output directory for models and reports")
    parser.add_argument("--no-save", action="store_true", help="Do not persist trained models")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only")
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
