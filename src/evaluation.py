"""
This module computes evaluation metrics to understand model performance

"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

def evaluate_model(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def compare_predictions(df: pd.DataFrame, actual_col: str, predicted_col: str) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df[actual_col] - df[predicted_col]
    return df


def cross_validate_regressor(estimator, X, y, cv: int = 5) -> dict:
    """Run cross-validation and return fold metrics and summary.

    Returns a dict with keys: `fold_mae`, `fold_mse`, `fold_r2`, and summary
    values `mae_mean`, `mae_std`, `rmse_mean`, `rmse_std`, `r2_mean`, `r2_std`.
    """
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2",
    }
    cv_res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=False)

    # Extract fold-level results
    fold_mae = -cv_res["test_MAE"]
    fold_mse = -cv_res["test_MSE"]
    fold_r2 = cv_res["test_R2"]

    # Summary statistics
    mae_mean, mae_std = float(np.mean(fold_mae)), float(np.std(fold_mae, ddof=1))
    rmse_mean, rmse_std = float(np.sqrt(np.mean(fold_mse))), float(np.std(np.sqrt(fold_mse), ddof=1))
    r2_mean, r2_std = float(np.mean(fold_r2)), float(np.std(fold_r2, ddof=1))

    return {
        "fold_mae": fold_mae.tolist(),
        "fold_mse": fold_mse.tolist(),
        "fold_r2": fold_r2.tolist(),
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "r2_mean": r2_mean,
        "r2_std": r2_std,
    }


def cross_validate_model_from_df(estimator, df: pd.DataFrame, feature_cols: list, target_col: str, cv: int = 5) -> dict:
    """Helper to run CV directly from a DataFrame and return results."""
    X = df[feature_cols]
    y = df[target_col]
    return cross_validate_regressor(estimator, X, y, cv=cv)
