"""
This module will manage model training using featured computations

"""
from . import pd, np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

# ---------------------------
# Train Model
# ---------------------------

def fit_sustainable_hr_model(df: pd.DataFrame, features: list, target: str, model_file: str = None):
    """
    Train a linear regression model to predict sustainable HR.

    Inputs
    ----------
    df : pd.DataFrame (Preprocessed dataset)
    features : list (Feature column names)
    target : str Target (column name (e.g., 'avg_hr'))
    model_file : str, optional (Path to save the trained model, joblib)
    
    Returns
    -------
    model : trained sklearn model
    df_pred : pd.DataFrame (with pred. HR)
    """
    df = df.copy()
    
    X = df[features]
    y = df[target]

    # Train-test split (optional for full dataset you could skip)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict for the full dataset
    df["predicted_sustainable_hr"] = model.predict(X)

    # Save model if path provided
    if model_file:
        Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)

    return model, df

# ---------------------------
# Predict New Data
# ---------------------------

def predict_sustainable_hr(df: pd.DataFrame, model_file: str, features: list):
    """
    Load a trained model and predict sustainable HR for new data.

    Inputs
    ----------
    df : pd.DataFrame (Preprocessed dataset)
    model_file : str (Path to trained model, joblib)
    features : list (Feature columns for prediction)

    Returns
    -------
    pd.Series (Predicted sustainable HR)

    """
    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = joblib.load(model_path)
    df = df.copy()
    df["predicted_sustainable_hr"] = model.predict(df[features])
    return df["predicted_sustainable_hr"]

    
    