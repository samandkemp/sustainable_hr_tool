import pandas as pd
import numpy as np
import pytest

from src.preprocessing import compute_pace, normalize_columns, preprocess_data


class TestComputePace:
    def test_pace_derived_from_duration_and_distance(self):
        df = pd.DataFrame({"duration_min": [60.0], "distance_km": [10.0]})
        result = compute_pace(df)
        assert result["avg_pace_min_km"].iloc[0] == pytest.approx(6.0)

    def test_duration_derived_from_pace_and_distance(self):
        df = pd.DataFrame({"avg_pace_min_km": [6.0], "distance_km": [10.0]})
        result = compute_pace(df)
        assert result["duration_min"].iloc[0] == pytest.approx(60.0)

    def test_existing_pace_not_overwritten(self):
        df = pd.DataFrame({
            "duration_min": [60.0],
            "distance_km": [10.0],
            "avg_pace_min_km": [5.0],
        })
        result = compute_pace(df)
        assert result["avg_pace_min_km"].iloc[0] == pytest.approx(6.0)

    def test_missing_distance_no_crash(self):
        df = pd.DataFrame({"duration_min": [60.0]})
        result = compute_pace(df)
        assert "avg_pace_min_km" not in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"duration_min": [60.0], "distance_km": [10.0]})
        compute_pace(df)
        assert "avg_pace_min_km" not in df.columns


class TestNormalizeColumns:
    def test_norm_column_added(self):
        df = pd.DataFrame({"distance_km": [5.0, 10.0, 15.0]})
        result = normalize_columns(df, ["distance_km"])
        assert "distance_km_norm" in result.columns

    def test_mean_approx_zero(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = normalize_columns(df, ["x"])
        assert result["x_norm"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_std_approx_one(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = normalize_columns(df, ["x"])
        assert result["x_norm"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_constant_column_no_crash(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        result = normalize_columns(df, ["x"])
        assert result["x_norm"].notna().all()

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = normalize_columns(df, ["b"])
        assert "b_norm" not in result.columns

    def test_none_columns_no_op(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = normalize_columns(df, None)
        assert "x_norm" not in result.columns


class TestPreprocessData:
    def test_drops_na_rows(self, minimal_run_df):
        df = minimal_run_df.copy()
        df.loc[0, "distance_km"] = float("nan")
        result = preprocess_data(df, drop_na_columns=["distance_km"])
        assert len(result) == len(minimal_run_df) - 1

    def test_no_na_drop_when_not_specified(self, minimal_run_df):
        df = minimal_run_df.copy()
        df.loc[0, "distance_km"] = float("nan")
        result = preprocess_data(df)
        assert len(result) == len(minimal_run_df)

    def test_compute_features_adds_pace(self):
        df = pd.DataFrame({"duration_min": [60.0], "distance_km": [10.0]})
        result = preprocess_data(df, compute_features=True)
        assert "avg_pace_min_km" in result.columns

    def test_compute_features_false_skips_pace(self):
        df = pd.DataFrame({"duration_min": [60.0], "distance_km": [10.0]})
        result = preprocess_data(df, compute_features=False)
        assert "avg_pace_min_km" not in result.columns

    def test_normalize_adds_norm_columns(self, minimal_run_df):
        result = preprocess_data(minimal_run_df, normalize=["distance_km"])
        assert "distance_km_norm" in result.columns

    def test_returns_copy(self, minimal_run_df):
        result = preprocess_data(minimal_run_df, drop_na_columns=["distance_km"])
        assert result is not minimal_run_df
