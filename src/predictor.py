"""
This modulke will use featured computations for model predictions

"""
# src/predictor.py
from . import pd

def predict_sustainable_hr(model, df, feature_cols=None):
    """
    Use trained model to predict sustainable heart rate
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'heart_rate']]
    
    df['predicted_sustainable_hr'] = model.predict(df[feature_cols])
    return df

