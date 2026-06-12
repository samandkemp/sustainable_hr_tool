"""
This module contains utility functions for the tool.
Utils: 
- dummy CSV data generator
- check df structure

To-do: 
- Error checking/validation
- Adjust synthesised data distributions

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

    # Save CSV
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dummy data saved to: {path.resolve()}")
    return df
