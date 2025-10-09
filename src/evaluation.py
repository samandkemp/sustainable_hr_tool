"""
This module computes evaluation metrics to understand model performance

"""

from . import np, pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
