"""targets
---------------
Helpers to create modelling targets and to build scenario inputs for
predicting sustainable HR or predicted pace given user-specified
distance/pace/HR conditions.

These utilities are lightweight and intended to be useful during
iteration while we have synthetic or small Garmin samples.
"""

import pandas as pd
import numpy as np


def ensure_target(df: pd.DataFrame, target_col: str = "avg_hr") -> pd.Series:
    """Return the target series, dropping rows without the target.

    Keeps behaviour explicit for downstream training functions.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    return df[target_col].dropna()


def build_scenario_template(df: pd.DataFrame, pace_min_km: float, distance_km: float) -> pd.DataFrame:
    """Create a one-row scenario DataFrame using column medians as defaults.

    The returned DataFrame has `avg_pace_min_km` and `distance_km` set to
    the requested values and other numeric columns populated with the
    median observed in `df` so models can be asked to predict on a
    plausible input.
    """
    numeric = df.select_dtypes(include=[np.number]).median()
    scenario = numeric.to_frame().T
    scenario["avg_pace_min_km"] = pace_min_km
    scenario["distance_km"] = distance_km
    return scenario.reset_index(drop=True)


def predict_hr_for_scenario(model, scenario_df: pd.DataFrame, feature_cols: list):
    """Return predicted HR (array-like) for the provided scenario(s).

    `scenario_df` should include all `feature_cols` used by the model.
    """
    # Safety: ensure columns present
    missing = [c for c in feature_cols if c not in scenario_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in scenario: {missing}")
    return model.predict(scenario_df[feature_cols])


def build_inverse_scenario(df: pd.DataFrame, hr: float, distance_km: float) -> pd.DataFrame:
    """Create a scenario row to ask 'what pace for HR=X and distance=Y'.

    Uses medians for other numeric fields and sets `avg_hr` and
    `distance_km` to the provided values. Useful when training a model
    that predicts pace from HR (inverse model).
    """
    numeric = df.select_dtypes(include=[np.number]).median()
    scenario = numeric.to_frame().T
    scenario["avg_hr"] = hr
    scenario["distance_km"] = distance_km
    return scenario.reset_index(drop=True)
