"""
This module handles the regression modelling.
To-Do: Update the linear regression to one more appropriate after the workflow is finished

"""

from sklearn.linear_model import LinearRegression

def train_hr_model(df, feature_cols, target_col):
    """ Starting with basic linear regression"""
    x = df[feature_cols]
    y = df[target_col]
    model = LinearRegression()
    model.fit(x, y)
    return model