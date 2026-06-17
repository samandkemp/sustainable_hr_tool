"""
This module loads and transforms Garmin CSV data into a pandas DataFrame.

"""

from pathlib import Path
import pandas as pd
import re
import numpy as np
from typing import Optional

def load_data(file_path: str) -> pd.DataFrame:
    # Single CSV load
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(path)
    # If this looks like the Garmin Activities export, parse into canonical schema
    if "Activity Type" in df.columns or "Activity" in path.name:
        return parse_activities_csv(df)
    return df

def load_all_data(folder_path: str) -> pd.DataFrame:
    # Complete CSV load
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_files = list(folder.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    df_list = []
    for f in all_files:
        tmp = pd.read_csv(f)
        if "Activity Type" in tmp.columns or "Activity" in f.name:
            tmp = parse_activities_csv(tmp)
        df_list.append(tmp)
    return pd.concat(df_list, ignore_index=True)


def _snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\s/\-]+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "", name)
    return name.lower()


def _time_to_minutes(t: str) -> Optional[float]:
    """Convert HH:MM:SS or MM:SS to minutes (float)."""
    if pd.isna(t):
        return None
    t = str(t)
    parts = t.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0.0
        m, s = parts
    else:
        return None
    return h * 60.0 + m + s / 60.0


def _pace_to_min_per_km(pace: str) -> Optional[float]:
    """Convert pace strings like '12:51' (min:sec) to minutes per km float."""
    if pd.isna(pace):
        return None
    s = str(pace).strip()
    if s == "":
        return None
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:
        m, sec = parts
        return m + sec / 60.0
    if len(parts) == 3:
        h, m, sec = parts
        return h * 60.0 + m + sec / 60.0
    return None


def _clean_numeric(x):
    if pd.isna(x):
        return None
    s = str(x).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_activities_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a Garmin Activities CSV (one row per activity) into a canonical table.

    The returned DataFrame uses snake_case columns and ensures numeric
    `distance_km`, `duration_min`, `avg_pace_min_km`, `avg_hr`, and
    `elevation_gain_m` when possible.

    TO-DO: Check against other garmin CSV formats/Apple Watch data. Currently based on Vivoactive 3 data format.
    """
    df = df.copy()
    # Normalise column names
    col_map = {c: _snake_case(c) for c in df.columns}
    df.columns = [col_map[c] for c in df.columns]

    # Common column aliases
    # distance may be 'distance' with km units
    if "distance" in df.columns:
        df["distance_km"] = df["distance"].apply(_clean_numeric)

    # Duration: try columns 'time', 'moving_time', 'elapsed_time'
    for cand in ("time", "moving_time", "elapsed_time"):
        if cand in df.columns:
            df["duration_min"] = df[cand].apply(_time_to_minutes)
            break

    # Average pace may be 'avg_pace' or 'avg_pace_min_km' or 'avg_pace' string
    for cand in ("avg_pace", "avg_pace_min_km", "avg pace", "avg pace (min/km)"):
        if cand in df.columns:
            df["avg_pace_min_km"] = df[cand].apply(_pace_to_min_per_km)
            break

    # Heart rate
    for cand in ("avg_hr", "avg_heart_rate", "avgheart" ):
        if cand in df.columns:
            df["avg_hr"] = df[cand].apply(_clean_numeric)
            break

    # Max heart rate
    for cand in ("max_hr", "max_heart_rate", "maxheart"):
        if cand in df.columns:
            df["max_hr"] = df[cand].apply(_clean_numeric)
            break

    # Cadence
    for cand in ("avg_cadence", "avg cadence"):
        if cand in df.columns:
            df["avg_cadence"] = df[cand].apply(_clean_numeric)
            break
    for cand in ("max_cadence", "max cadence"):
        if cand in df.columns:
            df["max_cadence"] = df[cand].apply(_clean_numeric)
            break

    # Elevation / ascent
    for cand in ("total_ascent", "total_ascent_m"):
        if cand in df.columns:
            df["elevation_gain_m"] = df[cand].apply(_clean_numeric)
            break

    # Total descent (if present)
    for cand in ("total_descent", "total_descent_m", "descent"):
        if cand in df.columns:
            df["total_descent_m"] = df[cand].apply(_clean_numeric)
            break

    # Calories
    for cand in ("calories", "kcal"):
        if cand in df.columns:
            df["calories"] = df[cand].apply(_clean_numeric)
            break

    # Avg / best pace as additional columns
    for cand in ("best_pace", "best pace"):
        if cand in df.columns:
            df["best_pace_min_km"] = df[cand].apply(_pace_to_min_per_km)
            break

    # Stride length
    for cand in ("avg_stride_length", "avg stride length"):
        if cand in df.columns:
            df["avg_stride_length"] = df[cand].apply(_clean_numeric)
            break

    # Training stress score
    for cand in ("training_stress_score", "training stress score®", "tss"):
        if cand in df.columns:
            df["tss"] = df[cand].apply(_clean_numeric)
            break

    # Lap, moving and elapsed times
    if "best_lap_time" in df.columns:
        df["best_lap_min"] = df["best_lap_time"].apply(_time_to_minutes)
    if "moving_time" in df.columns:
        df["moving_time_min"] = df["moving_time"].apply(_time_to_minutes)
    if "elapsed_time" in df.columns:
        df["elapsed_time_min"] = df["elapsed_time"].apply(_time_to_minutes)

    # Min/Max elevation
    for cand in ("min_elevation", "min elevation"):
        if cand in df.columns:
            df["min_elevation"] = df[cand].apply(_clean_numeric)
            break
    for cand in ("max_elevation", "max elevation"):
        if cand in df.columns:
            df["max_elevation"] = df[cand].apply(_clean_numeric)
            break

    # Steps with commas
    if "steps" in df.columns:
        df["steps"] = df["steps"].apply(lambda x: int(str(x).replace(",", "")) if pd.notna(x) and str(x).strip() != "" else None)

    # Keep a conservative set of columns; preserve others too
    return df