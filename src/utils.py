"""
This module contains utility functions for the tool.
Contained Utils: 
- dummy CSV data generator

"""
# src/utils.py
import datetime as dt
import os
from . import pd, np

def create_dummy_garmin_csv(filepath: str, n: int = 600):
    # Create a dummy CSV dataset
    start = dt.datetime.now()
    timestamps = pd.date_range(start=start, periods=n, freq="S")

    base_hr = np.random.choice([135, 145, 160], p=[0.4, 0.4, 0.2])
    heart_rate = base_hr + 5 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)
    distance = np.linspace(0, n * 3, n)
    cadence = np.random.normal(170, 3, n)
    elevation = np.sin(np.linspace(0, 2*np.pi, n)) * 10 + 100
    pace = 1000 / (distance / np.arange(1, n+1))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": heart_rate,
        "distance": distance,
        "cadence": cadence,
        "elevation": elevation,
        "pace": pace,
        "activity_type": ["running"] * n
    })

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Dummy CSV saved to: {filepath}")
    return df
