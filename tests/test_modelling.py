import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.modelling import get_estimator, fit_sustainable_hr_model, fit_pace_from_hr_model


class TestGetEstimator:
    def test_linear_returns_linear_regression(self):
        assert isinstance(get_estimator("linear"), LinearRegression)

    def test_ridge_returns_ridge(self):
        assert isinstance(get_estimator("ridge"), Ridge)

    def test_random_forest_returns_rf(self):
        assert isinstance(get_estimator("random_forest"), RandomForestRegressor)

    def test_gradient_boosting_returns_gb(self):
        assert isinstance(get_estimator("gradient_boosting"), GradientBoostingRegressor)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_estimator("neural_network")

    def test_returns_fresh_instance_each_call(self):
        m1 = get_estimator("linear")
        m2 = get_estimator("linear")
        assert m1 is not m2

    def test_default_is_linear(self):
        assert isinstance(get_estimator(), LinearRegression)


class TestFitSustainableHrModel:
    def test_returns_tuple(self, featured_df):
        feature_cols = [c for c in featured_df.select_dtypes(include="number").columns
                        if c != "avg_hr" and not c.startswith("predicted_")]
        result = fit_sustainable_hr_model(featured_df, features=feature_cols)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prediction_column_added(self, featured_df):
        feature_cols = [c for c in featured_df.select_dtypes(include="number").columns
                        if c != "avg_hr" and not c.startswith("predicted_")]
        _, df_out = fit_sustainable_hr_model(featured_df, features=feature_cols)
        assert "predicted_sustainable_hr" in df_out.columns

    def test_prediction_length_matches_input(self, featured_df):
        feature_cols = [c for c in featured_df.select_dtypes(include="number").columns
                        if c != "avg_hr" and not c.startswith("predicted_")]
        _, df_out = fit_sustainable_hr_model(featured_df, features=feature_cols)
        assert len(df_out) == len(featured_df)

    def test_model_has_predict(self, featured_df):
        feature_cols = [c for c in featured_df.select_dtypes(include="number").columns
                        if c != "avg_hr" and not c.startswith("predicted_")]
        model, _ = fit_sustainable_hr_model(featured_df, features=feature_cols)
        assert hasattr(model, "predict")

    def test_gradient_boosting_model_type(self, featured_df):
        feature_cols = [c for c in featured_df.select_dtypes(include="number").columns
                        if c != "avg_hr" and not c.startswith("predicted_")]
        model, _ = fit_sustainable_hr_model(
            featured_df, features=feature_cols, model_type="gradient_boosting"
        )
        assert isinstance(model, GradientBoostingRegressor)

    def test_no_features_raises(self, featured_df):
        with pytest.raises(ValueError):
            fit_sustainable_hr_model(featured_df, features=[])

    def test_does_not_mutate_input(self, featured_df):
        cols_before = set(featured_df.columns)
        feature_cols = [c for c in featured_df.select_dtypes(include="number").columns
                        if c != "avg_hr" and not c.startswith("predicted_")]
        fit_sustainable_hr_model(featured_df, features=feature_cols)
        assert set(featured_df.columns) == cols_before


class TestFitPaceFromHrModel:
    def test_prediction_column_added(self, featured_df):
        _, df_out = fit_pace_from_hr_model(featured_df)
        assert "predicted_pace_min_km" in df_out.columns

    def test_prediction_length_matches(self, featured_df):
        _, df_out = fit_pace_from_hr_model(featured_df)
        assert len(df_out) == len(featured_df)

    def test_ridge_model_type(self, featured_df):
        model, _ = fit_pace_from_hr_model(featured_df, model_type="ridge")
        assert isinstance(model, Ridge)
