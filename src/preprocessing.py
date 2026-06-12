"""preprocessing
-----------------
Convenience preprocessing pipeline used by the modelling workflow.

The pipeline is intentionally small and explicit so it is easy to
extend during iterative development when real Garmin data is available.

Functions
- `preprocess_data` : orchestrates basic cleaning, type coercion, and
  optional normalization and duration/pacing computations.
"""

import pandas as pd
import numpy as np


def compute_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure pace-related columns exist: `avg_pace_min_km` and `duration_min`.

    If one is missing it will be computed from the other when possible.
    """
    df = df.copy()
    if "duration_min" in df.columns and "distance_km" in df.columns:
        df["avg_pace_min_km"] = (df["duration_min"] / df["distance_km"]).round(2)
    elif "avg_pace_min_km" in df.columns and "distance_km" in df.columns:
        df["duration_min"] = (df["avg_pace_min_km"] * df["distance_km"]).round(1)
    return df


def normalize_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Add z-score normalized versions for requested numeric columns.

    New columns are suffixed with `_norm`.
    """
    df = df.copy()
    for col in columns or []:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std(ddof=0) or 1.0
            df[f"{col}_norm"] = (df[col] - mean) / std
    return df


def preprocess_data(
    df: pd.DataFrame,
    drop_na_columns: list = None,
    normalize: list = None,
    compute_features: bool = True,
) -> pd.DataFrame:
    """Run a small sequence of preprocessing steps.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw/run-level dataframe.
    drop_na_columns : list, optional
        Columns to require (rows missing these are dropped).
    normalize : list, optional
        Numeric columns to z-normalize.
    compute_features : bool, default True
        Whether to compute basic derived features like pace/duration.

    Returns
    -------
    pd.DataFrame
        Processed dataframe ready for feature engineering.
    """
    df = df.copy()

    # Basic cleaning
    if drop_na_columns:
        df.dropna(subset=drop_na_columns, inplace=True)

    # Feature computations
    if compute_features:
        df = compute_pace(df)

    # Normalization
    if normalize:
        df = normalize_columns(df, normalize)

    return df
