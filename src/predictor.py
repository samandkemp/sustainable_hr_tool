"""
This module manages the model predictions.
To-Do: 

"""

def predict_sustainable_hr(model, df, feature_cols):
    return model.predict(df[feature_cols])