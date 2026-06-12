"""
This module contains different computations for feature parameters.
To-Do: 

"""
from . import pd, np


def add_effort_score(df: pd.DataFrame, effort_col: str = "effort_type") -> pd.DataFrame:
    """Map human-labelled effort types to an integer effort score.

    Keeps existing column if present; adds `effort_score` otherwise.
    """
    df = df.copy()
    if effort_col in df.columns:
        mapping = {"easy": 1, "tempo": 2, "race": 3}
        df["effort_score"] = df[effort_col].map(mapping).fillna(0).astype(int)
    return df


def elevation_adjusted_pace(df: pd.DataFrame, elev_col: str = "elevation_gain_m") -> pd.DataFrame:
    """Compute a simple elevation-adjusted pace proxy.

    This adds `grade` (elevation per metre) and `adj_pace_min_km` which
    nudges `avg_pace_min_km` up for positive elevation gain.
    The adjustment is intentionally small and interpretable.
    """
    df = df.copy()
    if "distance_km" in df.columns and elev_col in df.columns and "avg_pace_min_km" in df.columns:
        # avoid division by zero
        df["grade"] = (df[elev_col] / (df["distance_km"] * 1000)).fillna(0)
        # simple linear penalty: 1s per 1% grade per km (approx). Convert to minutes.
        df["adj_pace_min_km"] = (
            df["avg_pace_min_km"] + (df["grade"] * 0.06)
        ).round(2)
    return df


def fatigue_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple proxies for fatigue/stress based on duration and effort.

    - `duration_hr`: duration in hours
    - `stress_score`: effort_score * duration_hr (simple multiplicative proxy)
    """
    df = df.copy()
    if "duration_min" in df.columns:
        df["duration_hr"] = (df["duration_min"] / 60).round(2)
    if "effort_score" in df.columns and "duration_hr" in df.columns:
        df["stress_score"] = (df["effort_score"] * df["duration_hr"]).round(2)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run a sequence of feature engineering steps for modelling.

    This function is safe to run on the synthetic data generator output
    and will be expanded once real Garmin samples are available.
    """
    df = df.copy()
    df = add_effort_score(df)
    df = elevation_adjusted_pace(df)
    df = fatigue_proxies(df)
    return df
