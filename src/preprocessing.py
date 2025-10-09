"""
This module will preprocess the data for modelling

"""

from . import pd, np

def preprocess_data(df: pd.DataFrame, drop_na_columns=None, normalize_columns=None) -> pd.DataFrame:
    df = df.copy()
    if drop_na_columns:
        df.dropna(subset=drop_na_columns, inplace=True)
    if normalize_columns:
        for col in normalize_columns:
            df[f"{col}_norm"] = (df[col] - df[col].mean()) / df[col].std()
    return df
