"""
This module will manage model training using featured computations

"""
# src/modelling.py
from . import LinearRegression, RandomForestRegressor, train_test_split, mean_squared_error, r2_score, pd, np

def train_hr_model(df, target_col='heart_rate', feature_cols=None):
    """
    Train a simple regression model to return a trained model object
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ['timestamp', target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model trained. MSE={mse:.2f}, R2={r2:.2f}")

    return model
    
    