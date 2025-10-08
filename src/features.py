"""
This module handles different required features.

"""
# src/features.py
from . import pd

def compute_features(df):
    df['hr_rolling'] = df['heart_rate'].rolling(5, min_periods=1).mean()
    return df