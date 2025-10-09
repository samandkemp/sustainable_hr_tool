"""
This module computes visualisations to understand model performance

"""

from . import pd, np, sns, plt

def plot_residuals_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, kde=True, color="teal", bins=20)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Residual Distribution (Actual − Predicted HR)")
    plt.xlabel("Residual (bpm)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_residuals_vs_feature(df, actual_col, predicted_col, feature):
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found.")
    df = df.copy()
    df["residual"] = df[actual_col] - df[predicted_col]
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x=feature, y="residual", alpha=0.8)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Residuals vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Residual (bpm)")
    plt.grid(True, linestyle="--", alpha=0.6)
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
