"""
This module will manage model training using featured computations

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from typing import Union
from . import features as feat


def get_estimator(model_type: str = "linear"):
    """Return a fresh sklearn estimator for the given model type.

    Supported types: ``"linear"``, ``"ridge"``, ``"random_forest"``,
    ``"gradient_boosting"``.
    """
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    options = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if model_type not in options:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {list(options)}"
        )
    return options[model_type]

# ---------------------------
# Train Model
# ---------------------------

def _default_hr_features(df: pd.DataFrame) -> list:
    candidates = [
        "distance_km",
        "avg_pace_min_km",
        "elevation_gain_m",
        "temperature_c",
        "effort_score",
        "duration_min",
    ]
    return [c for c in candidates if c in df.columns]


def fit_sustainable_hr_model(
    df: pd.DataFrame,
    features: list = None,
    target: str = "avg_hr",
    model_file: Union[str, None] = None,
    model_type: str = "linear",
) -> tuple:
    """Train a baseline linear model to predict sustainable HR.

    The function is intentionally simple (linear regression) to provide an
    interpretable baseline. If `features` is not provided a sensible default
    set is selected from available columns.

    Returns (model, df_with_predictions).
    """
    df = df.copy()

    if features is None:
        features = _default_hr_features(df)

    if not features:
        raise ValueError("No candidate features found for HR model. Provide `features`.")

    X = df[features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_estimator(model_type)
    model.fit(X_train, y_train)

    # Predict for the full dataset and attach predictions
    df["predicted_sustainable_hr"] = model.predict(X)

    # Persist model if requested
    if model_file:
        Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)

    return model, df

# ---------------------------
# Predict New Data
# ---------------------------

def predict_sustainable_hr(df: pd.DataFrame, model_or_path: Union[str, object], features: list):
    """Predict sustainable HR using a model object or a path to a joblib file.

    `model_or_path` may be a fitted estimator with `predict` or a path to a
    persisted joblib file.
    """
    df = df.copy()
    # Load model if a path is provided
    if isinstance(model_or_path, str):
        model_path = Path(model_or_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_or_path}")
        model = joblib.load(model_path)
    else:
        model = model_or_path

    df["predicted_sustainable_hr"] = model.predict(df[features])
    return df["predicted_sustainable_hr"]


def fit_pace_from_hr_model(
    df: pd.DataFrame,
    features: list = None,
    target: str = "avg_pace_min_km",
    model_file: Union[str, None] = None,
    model_type: str = "linear",
) -> tuple:
    """Train a baseline model that predicts pace from HR and other covariates.

    Useful for the inverse-scenario: given HR and distance, what pace is
    expected.
    """
    df = df.copy()
    # sensible defaults: use HR + simple covariates
    if features is None:
        candidates = ["avg_hr", "distance_km", "elevation_gain_m", "effort_score"]
        features = [c for c in candidates if c in df.columns]

    if not features:
        raise ValueError("No candidate features found for pace model. Provide `features`.")

    X = df[features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_estimator(model_type)
    model.fit(X_train, y_train)

    df["predicted_pace_min_km"] = model.predict(X)

    if model_file:
        Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)

    return model, df
