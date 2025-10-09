"""
This module loads and transforms Garmin CSV data into a pandas DataFrame.

"""

from pathlib import Path
from . import pd

def load_data(file_path: str) -> pd.DataFrame:
    # Single CSV load
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(path)

def load_all_data(folder_path: str) -> pd.DataFrame:
    # Complete CSV load
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_files = list(folder.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    df_list = [pd.read_csv(f) for f in all_files]
    return pd.concat(df_list, ignore_index=True)