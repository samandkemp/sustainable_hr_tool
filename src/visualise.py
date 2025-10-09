"""
This module handles the visulisation of data and model results

"""

from . import pd, plt, sns

def plot_hr_vs_distance(df: pd.DataFrame):
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="distance_km", y="avg_hr", alpha=0.8)
    plt.title("Average HR vs Distance")
    plt.xlabel("Distance (km)")
    plt.ylabel("Average HR (bpm)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_predicted_vs_actual(df: pd.DataFrame, actual_col: str, predicted_col: str):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x=actual_col, y=predicted_col, alpha=0.8)
    plt.plot([df[actual_col].min(), df[actual_col].max()],
             [df[actual_col].min(), df[actual_col].max()],
             color="red", linestyle="--")
    plt.title("Predicted vs Actual HR")
    plt.xlabel("Actual HR (bpm)")
    plt.ylabel("Predicted HR (bpm)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
