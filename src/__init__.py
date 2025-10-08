"""
This module imports common packages across modules

"""
# src/__init__.py

# Core data libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine learning / modelling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set visualization defaults
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)
