"""
This module handles the visulisation of data and model results

"""
# src/visualise.py
from . import pd, np, plt, sns

def plot_hr(df, rolling_window: int = 5):
    """
    Plot heart rate and rolling average over time
    """
    if 'hr_rolling' not in df.columns:
        df['hr_rolling'] = df['heart_rate'].rolling(rolling_window, min_periods=1).mean()

    plt.figure(figsize=(10,5))
    plt.plot(df['timestamp'], df['heart_rate'], label="HR")
    plt.plot(df['timestamp'], df['hr_rolling'], linestyle='--', label=f"Rolling Avg ({rolling_window})")
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Heart Rate over Time")
    plt.legend()
    plt.show()

def plot_distance_vs_hr(df):
    """
    Scatter plot of distance vs heart rate
    """
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='distance', y='heart_rate', hue='activity_type', data=df)
    plt.xlabel("Distance (m)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Distance vs Heart Rate")
    plt.show()
