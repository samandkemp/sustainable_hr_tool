"""
This module loads and transforms Garmin CSV data into a pandas DataFrame.

"""
# src/data_loader.py
from pathlib import Path
from . import pd

def load_data(file_path: str) -> pd.DataFrame:
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file)
