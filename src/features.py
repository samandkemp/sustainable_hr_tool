"""
This module contains different computations for feature parameters.
"""
import pandas as pd
import numpy as np


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


def rolling_training_load(
    df: pd.DataFrame,
    window: int = 7,
    load_col: str = "distance_km",
) -> pd.DataFrame:
    """Add a rolling sum of training load over the previous `window` runs.

    Requires the DataFrame to be in chronological order (oldest first).
    The resulting column is named `rolling_<window>run_load`.
    """
    df = df.copy()
    if load_col in df.columns:
        df[f"rolling_{window}run_load"] = (
            df[load_col].rolling(window=window, min_periods=1).sum().round(1)
        )
    return df


def compute_atl_ctl_tsb(
    df: pd.DataFrame,
    load_col: str = "stress_score",
    atl_halflife: int = 7,
    ctl_halflife: int = 42,
) -> pd.DataFrame:
    """Add Acute Training Load (ATL), Chronic Training Load (CTL), and Training Stress Balance (TSB).

    ATL (fatigue) uses a short exponential decay; CTL (fitness) uses a long one.
    TSB = CTL - ATL: positive means fresh/tapered, negative means fatigued.

    If a date column is present the decay is calendar-aware (days). Otherwise
    it falls back to run-indexed decay (number of runs as the time axis).
    The load column defaults to ``stress_score`` if present, else ``distance_km``.
    """
    df = df.copy()

    if load_col not in df.columns:
        load_col = "distance_km"
    if load_col not in df.columns:
        return df

    date_col = next((c for c in ("date", "activity_date") if c in df.columns), None)

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        load = df[load_col].fillna(0)
        # adjust=True required when times= is supplied (pandas constraint)
        df["atl"] = load.ewm(halflife=f"{atl_halflife}D", times=dates).mean().round(2)
        df["ctl"] = load.ewm(halflife=f"{ctl_halflife}D", times=dates).mean().round(2)
    else:
        load = df[load_col].fillna(0)
        df["atl"] = load.ewm(halflife=atl_halflife, adjust=False).mean().round(2)
        df["ctl"] = load.ewm(halflife=ctl_halflife, adjust=False).mean().round(2)

    df["tsb"] = (df["ctl"] - df["atl"]).round(2)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run a sequence of feature engineering steps for modelling."""
    df = df.copy()
    df = add_effort_score(df)
    df = elevation_adjusted_pace(df)
    df = fatigue_proxies(df)
    df = rolling_training_load(df)
    df = compute_atl_ctl_tsb(df)
    return df
