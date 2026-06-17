"""Backwards-compatible re-exports from evaluation_visuals.

All plot functions live in evaluation_visuals; import from there directly.
"""

from .evaluation_visuals import (  # noqa: F401
    plot_hr_vs_distance,
    plot_hr_vs_pace,
    plot_training_history,
    plot_predicted_vs_actual,
    plot_residuals_distribution,
    plot_residuals_vs_feature,
    plot_feature_importance,
    plot_feature_correlation,
    plot_cv_metrics,
)
