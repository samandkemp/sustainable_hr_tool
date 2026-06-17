"""validation
----------------
Data validation and minimal redundant column removal utilities 
(locations/privacy).

The functions are intentionally lightweight: they check for required
columns, basic type coercion, and produce a small disk-written report. 
PII removal drops common descriptive columns but does not attempt aggressive 
anonymisation since the data is expected to be used by data owners.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def _is_pii_column(name: str) -> bool:
    s = name.lower()
    patterns = ["name", "title", "email", "id", "username", "phone", "address", "favorite"]
    return any(p in s for p in patterns)


def remove_pii_columns(df: pd.DataFrame, keep: List[str] = None) -> pd.DataFrame:
    """Drop obvious PII columns by name while preserving optional `keep` list."""
    keep = keep or []
    cols_to_drop = [c for c in df.columns if _is_pii_column(c) and c not in keep]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def validate_and_coerce(df: pd.DataFrame, required: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """Validate presence of required columns and coerce numeric-like columns.

    Returns (df_clean, report) where report contains keys `missing_columns`,
    `coerced_columns`, and basic dataframe shape information.
    """
    report = {}
    df = df.copy()
    required = required or ["distance_km", "duration_min", "avg_pace_min_km", "avg_hr"]

    missing = [c for c in required if c not in df.columns]
    report["missing_columns"] = missing

    # Attempt simple coercion for numeric-like columns
    coerced = []
    for col in df.columns:
        if df[col].dtype == object:
            # try to coerce to numeric where a large fraction parse
            sample = df[col].dropna().astype(str).head(50)
            if sample.str.replace(",", "").str.match(r"^-?\d+(\.\d+)?$").sum() / max(1, len(sample)) > 0.5:
                df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
                coerced.append(col)
    report["coerced_columns"] = coerced
    report["shape"] = df.shape
    return df, report
