"""
This module contains different computations for feature parameters.
To-Do: 

"""

from . import pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'heart_rate' in df.columns and 'time' in df.columns:
        df['hr_delta'] = df['heart_rate'].diff().fillna(0)
        df['hr_rolling_avg'] = df['heart_rate'].rolling(window=30, min_periods=1).mean()
    return df
