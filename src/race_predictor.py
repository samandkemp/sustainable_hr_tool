"""race_predictor
-----------------
Predict sustainable heart rate, pace, and finish time for standard race distances.

Two complementary prediction directions are supported:

  Forward  — given a pace, predict the HR the model expects at that effort level
             for a given distance.
  Inverse  — given a target HR (or target finish time), predict the sustainable
             pace and resulting finish time.

Typical usage
-------------
After training::

    from src import race_predictor
    report = race_predictor.race_report(hr_model, pace_model, df_feat, feature_cols_hr, feature_cols_pace)
    print(report.to_string(index=False))

To predict for a single race::

    result = race_predictor.predict_for_target_time(race="marathon", target_time_min=210)
"""

import pandas as pd
import numpy as np
from typing import Optional

from . import preprocessing, features as feat, targets


RACE_DISTANCES_KM: dict = {
    "5k": 5.0,
    "10k": 10.0,
    "half_marathon": 21.097,
    "marathon": 42.195,
}

RACE_LABELS: dict = {
    "5k": "5K",
    "10k": "10K",
    "half_marathon": "Half Marathon",
    "marathon": "Marathon",
}


def format_finish_time(total_minutes: float) -> str:
    """Convert a float number of minutes to a human-readable HH:MM:SS string."""
    total_minutes = max(0.0, total_minutes)
    h = int(total_minutes // 60)
    m = int(total_minutes % 60)
    s = int(round((total_minutes - int(total_minutes)) * 60))
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _run_scenario_pipeline(scenario: pd.DataFrame) -> pd.DataFrame:
    """Run a pre-built scenario row through the preprocessing and feature pipeline."""
    scenario = preprocessing.preprocess_data(scenario, compute_features=True)
    return feat.compute_features(scenario)


def _build_and_run_scenario(df: pd.DataFrame, pace_min_km: float, distance_km: float) -> pd.DataFrame:
    return _run_scenario_pipeline(
        targets.build_scenario_template(df, pace_min_km=pace_min_km, distance_km=distance_km)
    )


def _build_and_run_inverse_scenario(df: pd.DataFrame, hr: float, distance_km: float) -> pd.DataFrame:
    return _run_scenario_pipeline(
        targets.build_inverse_scenario(df, hr=hr, distance_km=distance_km)
    )


def _align_features(scenario: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Keep only the model's feature columns; fill any missing ones with 0."""
    missing = [c for c in feature_cols if c not in scenario.columns]
    for c in missing:
        scenario[c] = 0.0
    return scenario[feature_cols]


def predict_hr_for_race(
    hr_model,
    df: pd.DataFrame,
    race: str,
    pace_min_km: float,
    feature_cols: list,
) -> dict:
    """Predict sustainable HR for a given race distance and target pace.

    Parameters
    ----------
    hr_model : sklearn estimator
        Trained model that predicts ``avg_hr``.
    df : pd.DataFrame
        Training / historical data used to derive column medians.
    race : str
        One of ``RACE_DISTANCES_KM`` keys (e.g. ``"marathon"``).
    pace_min_km : float
        Target pace in minutes per km.
    feature_cols : list
        Feature column names the model was trained on.

    Returns
    -------
    dict
        Keys: race, distance_km, pace_min_km, predicted_hr_bpm,
        estimated_finish_time, finish_minutes.
    """
    if race not in RACE_DISTANCES_KM:
        raise KeyError(f"Unknown race '{race}'. Choose from: {list(RACE_DISTANCES_KM)}")

    dist = RACE_DISTANCES_KM[race]
    scenario = _build_and_run_scenario(df, pace_min_km=pace_min_km, distance_km=dist)
    X = _align_features(scenario, feature_cols)
    pred_hr = float(hr_model.predict(X)[0])
    finish_min = pace_min_km * dist

    return {
        "race": RACE_LABELS.get(race, race),
        "distance_km": dist,
        "pace_min_km": round(pace_min_km, 2),
        "predicted_hr_bpm": round(pred_hr, 1),
        "estimated_finish_time": format_finish_time(finish_min),
        "finish_minutes": round(finish_min, 1),
    }


def predict_pace_for_target_hr(
    pace_model,
    df: pd.DataFrame,
    race: str,
    target_hr: float,
    feature_cols: list,
) -> dict:
    """Predict pace and finish time given a target sustainable HR and race distance.

    Parameters
    ----------
    pace_model : sklearn estimator
        Trained model that predicts ``avg_pace_min_km``.
    df : pd.DataFrame
        Training / historical data used to derive column medians.
    race : str
        One of ``RACE_DISTANCES_KM`` keys.
    target_hr : float
        Desired average heart rate (bpm) for the race.
    feature_cols : list
        Feature column names the pace model was trained on.

    Returns
    -------
    dict
        Keys: race, distance_km, target_hr_bpm, predicted_pace_min_km,
        estimated_finish_time, finish_minutes.
    """
    if race not in RACE_DISTANCES_KM:
        raise KeyError(f"Unknown race '{race}'. Choose from: {list(RACE_DISTANCES_KM)}")

    dist = RACE_DISTANCES_KM[race]
    scenario = _build_and_run_inverse_scenario(df, hr=target_hr, distance_km=dist)
    X = _align_features(scenario, feature_cols)
    pred_pace = float(pace_model.predict(X)[0])
    finish_min = pred_pace * dist

    return {
        "race": RACE_LABELS.get(race, race),
        "distance_km": dist,
        "target_hr_bpm": target_hr,
        "predicted_pace_min_km": round(pred_pace, 2),
        "estimated_finish_time": format_finish_time(finish_min),
        "finish_minutes": round(finish_min, 1),
    }


def predict_for_target_time(
    race: str,
    target_time_min: float,
) -> dict:
    """Given a target finish time, compute the required pace for a race.

    Parameters
    ----------
    race : str
        One of ``RACE_DISTANCES_KM`` keys.
    target_time_min : float
        Desired finish time in minutes (e.g. 210 for a 3:30 marathon).

    Returns
    -------
    dict
        Keys: race, distance_km, target_finish_time, required_pace_min_km,
        finish_minutes.
    """
    if race not in RACE_DISTANCES_KM:
        raise KeyError(f"Unknown race '{race}'. Choose from: {list(RACE_DISTANCES_KM)}")

    dist = RACE_DISTANCES_KM[race]
    required_pace = target_time_min / dist

    return {
        "race": RACE_LABELS.get(race, race),
        "distance_km": dist,
        "target_finish_time": format_finish_time(target_time_min),
        "required_pace_min_km": round(required_pace, 2),
        "finish_minutes": round(target_time_min, 1),
    }


def race_report(
    hr_model,
    pace_model,
    df: pd.DataFrame,
    feature_cols_hr: list,
    feature_cols_pace: list,
    target_hr: Optional[float] = None,
) -> pd.DataFrame:
    """Generate a full race prediction table for all standard distances.

    For each race the function predicts:
    - HR expected at the runner's median training pace.
    - If ``target_hr`` is provided: pace and finish time achievable at that HR.

    Parameters
    ----------
    hr_model, pace_model : sklearn estimators
        Trained HR and pace models.
    df : pd.DataFrame
        Feature-engineered training data.
    feature_cols_hr, feature_cols_pace : list
        Feature columns for each model.
    target_hr : float, optional
        If given, also predict pace and finish time at this HR for each distance.

    Returns
    -------
    pd.DataFrame
        One row per race distance.
    """
    baseline_pace = (
        df["avg_pace_min_km"].median() if "avg_pace_min_km" in df.columns else 5.5
    )

    rows = []
    for race in RACE_DISTANCES_KM:
        row = predict_hr_for_race(hr_model, df, race, baseline_pace, feature_cols_hr)

        if target_hr is not None:
            inv = predict_pace_for_target_hr(pace_model, df, race, target_hr, feature_cols_pace)
            row["sustainable_pace_at_target_hr"] = inv["predicted_pace_min_km"]
            row["finish_time_at_target_hr"] = inv["estimated_finish_time"]

        rows.append(row)

    return pd.DataFrame(rows)
