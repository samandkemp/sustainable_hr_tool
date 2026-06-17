"""
This module computes visualisations to understand model performance

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_hr_vs_distance(df: pd.DataFrame, out_path: str = None):
    """Scatter plot of average HR versus distance for exploratory inspection."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x="distance_km", y="avg_hr", alpha=0.8, ax=ax)
    ax.set_title("Average HR vs Distance")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Average HR (bpm)")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_hr_vs_pace(df: pd.DataFrame, out_path: str = None):
    """Scatter plot of average HR versus pace for exploratory inspection."""
    if "avg_pace_min_km" not in df.columns:
        raise KeyError("avg_pace_min_km not found in dataframe")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x="avg_pace_min_km", y="avg_hr", alpha=0.8, ax=ax)
    ax.set_title("Average HR vs Pace")
    ax.set_xlabel("Pace (min/km)")
    ax.set_ylabel("Average HR (bpm)")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_training_history(df: pd.DataFrame, metric: str = "distance_km", out_path: str = None):
    """Line chart of a metric over runs, useful for viewing training load over time."""
    if metric not in df.columns:
        raise KeyError(f"Column '{metric}' not found in dataframe")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[metric].values, marker="o", markersize=3, linewidth=1)
    ax.set_title(f"Training history: {metric}")
    ax.set_xlabel("Run index")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_cv_metrics(cv_results: dict, out_path=None):
    """Plot cross-validation fold metrics (MAE and RMSE) from cv_results dict.

    Expects the dict format produced by `cross_validate_regressor`.
    If `out_path` is provided, the plot is saved to that path.
    """
    fold_mae = cv_results.get("fold_mae", [])
    fold_mse = cv_results.get("fold_mse", [])
    if not fold_mae:
        raise ValueError("cv_results does not contain fold_mae")

    fold_rmse = np.sqrt(np.array(fold_mse))
    folds = np.arange(1, len(fold_mae) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(folds, fold_mae, marker="o", label="MAE")
    ax.plot(folds, fold_rmse, marker="o", label="RMSE")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Error")
    ax.set_title("Cross-validation fold errors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        return fig

def plot_residuals_distribution(y_true, y_pred, out_path: str = None):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals, kde=True, color="teal", bins=20, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Residual Distribution (Actual − Predicted HR)")
    ax.set_xlabel("Residual (bpm)")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()

def plot_residuals_vs_feature(df, actual_col, predicted_col, feature, out_path: str = None):
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found.")
    df = df.copy()
    df["residual"] = df[actual_col] - df[predicted_col]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x=feature, y="residual", alpha=0.8, ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"Residuals vs {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Residual (bpm)")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()

def plot_learning_curve(train_sizes, train_scores, test_scores):
    plt.figure(figsize=(7, 4))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), "o-", label="Validation score")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_predicted_vs_actual(y_true, y_pred, out_path: str = None):
    """Scatter plot of predicted vs actual with identity line and density."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, ax=ax)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_feature_importance(model, feature_names, out_path: str = None, top_n: int = 20):
    """Plot feature importance for linear or tree-based models.

    Uses ``coef_`` for linear models (signed) and ``feature_importances_``
    for ensemble models (unsigned, normalised).
    """
    if hasattr(model, "coef_"):
        df = pd.DataFrame({"feature": feature_names, "importance": model.coef_})
        df = df.reindex(df["importance"].abs().sort_values(ascending=False).index).head(top_n)
        fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(df))))
        sns.barplot(data=df, x="importance", y="feature", hue="feature",
                    palette="vlag", legend=False, ax=ax)
        ax.set_xlabel("Coefficient")
    elif hasattr(model, "feature_importances_"):
        df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        df = df.sort_values("importance", ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(df))))
        sns.barplot(data=df, x="importance", y="feature", hue="feature",
                    palette="Blues_d", legend=False, ax=ax)
        ax.set_xlabel("Importance")
    else:
        raise ValueError("Model exposes neither `coef_` nor `feature_importances_`")

    ax.set_title(f"Feature importance ({type(model).__name__})")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_feature_correlation(df, feature_cols, out_path: str = None):
    """Plot a correlation heatmap for selected feature columns."""
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()
