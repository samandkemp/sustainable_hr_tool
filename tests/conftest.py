import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def minimal_run_df():
    """Minimal processed-format DataFrame with 20 synthetic runs."""
    np.random.seed(0)
    n = 20
    effort_types = ["easy", "tempo", "race"] * 7 + ["easy"]
    return pd.DataFrame({
        "distance_km": np.random.choice([5.0, 10.0, 15.0, 21.1], n),
        "duration_min": np.random.uniform(30, 180, n),
        "avg_pace_min_km": np.random.uniform(4.5, 7.0, n),
        "avg_hr": np.random.uniform(130, 175, n),
        "elevation_gain_m": np.random.uniform(0, 400, n),
        "effort_type": effort_types[:n],
        "temperature_c": np.random.uniform(5, 25, n),
    })


@pytest.fixture
def dated_run_df(minimal_run_df):
    """Run DataFrame with a date column added (chronological, daily)."""
    df = minimal_run_df.copy()
    df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df


@pytest.fixture
def featured_df(minimal_run_df):
    """Run DataFrame that has passed through compute_features()."""
    from src import features
    return features.compute_features(minimal_run_df)
