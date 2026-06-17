"""
Utility functions: 
- synthetic data generation
- DataFrame inspection.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import os

def check_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    print(f"\n{name}: {df.shape[0]} rows × {df.shape[1]} cols")
    print(df.head())

def generate_dummy_data(
    save_path: str = "data/dummy_running_data.csv",
    n_runs: int = 120,
    random_state: int = 42
    ) -> pd.DataFrame:
    
    
    np.random.seed(random_state)

    # Main run metrics
    distances = np.random.choice([5, 10, 15, 21.1, 30, 42.2], n_runs, p=[0.2,0.3,0.2,0.15,0.1,0.05])
    effort_types = np.random.choice(["easy","tempo","race"], n_runs, p=[0.5,0.35,0.15])
    avg_pace_min_km = np.random.normal(5.5, 0.7, n_runs)
    duration_min = distances * avg_pace_min_km
    avg_hr = np.random.normal(150, 10, n_runs)
    for i, e in enumerate(effort_types):
        if e=="easy": avg_hr[i]-=np.random.randint(10,20)
        if e=="race": avg_hr[i]+=np.random.randint(5,15)
    elevation_gain = np.random.randint(0,600,n_runs)
    temperature_c = np.random.normal(15,8,n_runs).clip(-5,35)

    # Create df
    df = pd.DataFrame({
        "run_id": np.arange(1,n_runs+1),
        "distance_km": distances,
        "duration_min": duration_min.round(1),
        "avg_pace_min_km": avg_pace_min_km.round(2),
        "avg_hr": avg_hr.round(0),
        "effort_type": effort_types,
        "elevation_gain_m": elevation_gain,
        "temperature_c": temperature_c.round(1),
    })

    # Derived features (to adjust in future)
    df["effort_score"] = df["effort_type"].map({"easy":1,"tempo":2,"race":3})
    df["distance_category"] = pd.cut(df["distance_km"], bins=[0,10,21.1,42.3], labels=["short","medium","long"])

    # Save processed CSV (canonical schema)
    proc_path = Path(save_path)
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_path, index=False)
    print(f"Dummy processed data saved to: {proc_path.resolve()}")

    return df


def generate_synthetic_and_raw(
    processed_path: str = "data/processed/processed_synthetic.csv",
    raw_path: str = "data/raw/Activities_synthetic.csv",
    n_runs: int = 120,
    random_state: int = 42,
):
    """Generate synthetic processed data and a Garmin-style raw CSV template.

    The processed CSV follows the columns used by the pipeline
    (distance_km, duration_min, avg_pace_min_km, avg_hr, etc.). A parallel
    raw-format CSV is written using common Garmin headers so the project can
    accept a dropped CSV in `data/raw/`.
    """
    proc_df = generate_dummy_data(save_path=processed_path, n_runs=n_runs, random_state=random_state)

    # Probe for an Activities.csv header to reproduce the real export schema
    raw_path_p = Path(raw_path)
    raw_path_p.parent.mkdir(parents=True, exist_ok=True)

    # Prefer an existing Activities.csv in data/raw as the header template
    template_file = Path("data/raw/Activities.csv")
    if not template_file.exists():
        # fallback to provided template path if it exists
        tpl = Path("data/raw/template_Activities.csv")
        if tpl.exists():
            template_file = tpl

    # Helper conversions
    def minutes_to_hms(m):
        if pd.isna(m):
            return ""
        total_seconds = int(round(m * 60))
        h = total_seconds // 3600
        rem = total_seconds % 3600
        mm = rem // 60
        ss = rem % 60
        if h > 0:
            return f"{h}:{mm:02d}:{ss:02d}"
        return f"{mm}:{ss:02d}"

    def minperkm_to_mmss(m):
        if pd.isna(m):
            return ""
        total_seconds = int(round(m * 60))
        mm = total_seconds // 60
        ss = total_seconds % 60
        return f"{mm}:{ss:02d}"

    # Build a comprehensive raw-style DataFrame matching common Garmin export columns
    raw_df = pd.DataFrame()
    raw_df["Activity Type"] = proc_df.get("effort_type", "Run")
    raw_df["Date"] = pd.Timestamp.now().normalize() - pd.to_timedelta(np.arange(len(proc_df)), unit="d")
    raw_df["Favorite"] = False
    raw_df["Title"] = proc_df.get("distance_km").apply(lambda d: f"Synthetic run {d}km")
    raw_df["Distance"] = proc_df.get("distance_km")
    # Calories: simple heuristic
    raw_df["Calories"] = (proc_df.get("distance_km", 0) * 60).round(0).astype(int)
    raw_df["Time"] = proc_df.get("duration_min").apply(minutes_to_hms)
    raw_df["Avg HR"] = proc_df.get("avg_hr")
    raw_df["Max HR"] = (proc_df.get("avg_hr").fillna(140) + np.random.randint(5, 30, size=len(proc_df))).round(0)
    raw_df["Avg Cadence"] = np.random.randint(80, 120, size=len(proc_df))
    raw_df["Max Cadence"] = raw_df["Avg Cadence"] + np.random.randint(5, 40, size=len(proc_df))
    raw_df["Avg Pace"] = proc_df.get("avg_pace_min_km").apply(minperkm_to_mmss)
    raw_df["Best Pace"] = proc_df.get("avg_pace_min_km").apply(lambda m: minperkm_to_mmss(m * 0.9 if pd.notna(m) else m))
    raw_df["Total Ascent"] = proc_df.get("elevation_gain_m").fillna(0).astype(int)
    raw_df["Total Descent"] = (raw_df["Total Ascent"] * 0.9).astype(int)
    raw_df["Avg Stride Length"] = proc_df.get("avg_stride_length") if "avg_stride_length" in proc_df.columns else proc_df.get("avg_pace_min_km").apply(lambda p: 0.75)
    raw_df["Training Stress Score®"] = (proc_df.get("distance_km", 0) * 3).round(0)
    # Steps with comma thousands as in real export
    steps = proc_df.get("steps")
    if steps is None or steps.isnull().all():
        steps = (proc_df.get("distance_km", 0) * 1000).astype(int)
    raw_df["Steps"] = steps.apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    raw_df["Decompression"] = "No"
    raw_df["Best Lap Time"] = proc_df.get("duration_min").apply(lambda m: minutes_to_hms(m/ (1 if m is None else 1)))
    raw_df["Number of Laps"] = 1
    raw_df["Moving Time"] = proc_df.get("duration_min").apply(minutes_to_hms)
    raw_df["Elapsed Time"] = proc_df.get("duration_min").apply(minutes_to_hms)
    raw_df["Min Elevation"] = (
        proc_df["elevation_min"].fillna(0).astype(int)
        if "elevation_min" in proc_df.columns
        else 0
    )
    raw_df["Max Elevation"] = (
        proc_df["elevation_max"].fillna(0).astype(int)
        if "elevation_max" in proc_df.columns
        else raw_df["Total Ascent"].fillna(0).astype(int)
    )

    # Reorder columns to match an Activities export if possible
    if template_file.exists():
        try:
            tpl = pd.read_csv(template_file, nrows=0)
            cols = list(tpl.columns)
            # keep only columns we built, in template order, and append any extras
            out_cols = [c for c in cols if c in raw_df.columns]
            extras = [c for c in raw_df.columns if c not in out_cols]
            raw_df = raw_df[out_cols + extras]
        except Exception:
            pass

    raw_df.to_csv(raw_path_p, index=False)
    print(f"Synthetic raw-format data saved to: {raw_path_p.resolve()}")

    return proc_df, raw_df


def _cli_generate():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic processed and raw-format activity CSVs")
    parser.add_argument("--processed-path", default="data/processed/processed_synthetic.csv", help="Path to write processed CSV")
    parser.add_argument("--raw-path", default="data/raw/Activities_synthetic.csv", help="Path to write raw-format CSV")
    parser.add_argument("--n", type=int, default=120, help="Number of synthetic runs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    proc_df, raw_df = generate_synthetic_and_raw(processed_path=args.processed_path, raw_path=args.raw_path, n_runs=args.n, random_state=args.seed)
    print(f"Wrote processed data to: {args.processed_path}")
    print(f"Wrote raw-format template to: {args.raw_path}")


if __name__ == "__main__":
    _cli_generate()
