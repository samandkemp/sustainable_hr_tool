import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.evaluation import evaluate_model, cross_validate_regressor, cross_validate_model_from_df


class TestEvaluateModel:
    def test_perfect_predictions_zero_mae(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = evaluate_model(y, y)
        assert metrics["MAE"] == pytest.approx(0.0)
        assert metrics["RMSE"] == pytest.approx(0.0)
        assert metrics["R2"] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        y = pd.Series([1.0, 2.0, 3.0])
        metrics = evaluate_model(y, y)
        assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}

    def test_known_mae(self):
        y_true = pd.Series([0.0, 0.0, 0.0])
        y_pred = pd.Series([1.0, 2.0, 3.0])
        metrics = evaluate_model(y_true, y_pred)
        assert metrics["MAE"] == pytest.approx(2.0)

    def test_known_rmse(self):
        y_true = pd.Series([0.0, 0.0])
        y_pred = pd.Series([3.0, 4.0])
        metrics = evaluate_model(y_true, y_pred)
        expected_rmse = np.sqrt((9.0 + 16.0) / 2)
        assert metrics["RMSE"] == pytest.approx(expected_rmse)

    def test_values_are_floats(self):
        y = pd.Series([1.0, 2.0, 3.0])
        metrics = evaluate_model(y, y)
        for v in metrics.values():
            assert isinstance(v, float)


class TestCrossValidateRegressor:
    def _simple_data(self, n=50):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3))
        y = X[:, 0] * 2.0 + rng.standard_normal(n) * 0.1
        return X, y

    def test_returns_summary_keys(self):
        X, y = self._simple_data()
        result = cross_validate_regressor(LinearRegression(), X, y, cv=3)
        for key in ("mae_mean", "mae_std", "rmse_mean", "r2_mean", "r2_std"):
            assert key in result, f"Missing key: {key}"

    def test_mae_mean_positive(self):
        X, y = self._simple_data()
        result = cross_validate_regressor(LinearRegression(), X, y, cv=3)
        assert result["mae_mean"] >= 0.0

    def test_r2_reasonable(self):
        X, y = self._simple_data()
        result = cross_validate_regressor(LinearRegression(), X, y, cv=3)
        assert result["r2_mean"] > 0.5

    def test_fold_arrays_length(self):
        X, y = self._simple_data()
        result = cross_validate_regressor(LinearRegression(), X, y, cv=4)
        assert len(result["fold_mae"]) == 4
        assert len(result["fold_r2"]) == 4


class TestCrossValidateModelFromDf:
    def test_basic_run(self, featured_df):
        feature_cols = [c for c in featured_df.select_dtypes(include="number").columns
                        if c != "avg_hr" and not c.startswith("predicted_")]
        result = cross_validate_model_from_df(
            LinearRegression(), featured_df, feature_cols, "avg_hr", cv=3
        )
        assert "mae_mean" in result
        assert result["mae_mean"] >= 0.0

    def test_raises_on_missing_target(self, featured_df):
        with pytest.raises(KeyError):
            cross_validate_model_from_df(
                LinearRegression(), featured_df, ["distance_km"], "nonexistent_col", cv=3
            )
