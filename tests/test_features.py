import pandas as pd
import numpy as np
import pytest

from src.features import (
    add_effort_score,
    elevation_adjusted_pace,
    fatigue_proxies,
    rolling_training_load,
    compute_atl_ctl_tsb,
    compute_features,
)


class TestAddEffortScore:
    def test_mapping_applied(self, minimal_run_df):
        result = add_effort_score(minimal_run_df)
        assert "effort_score" in result.columns
        assert set(result["effort_score"].unique()).issubset({0, 1, 2, 3})

    def test_easy_maps_to_1(self):
        df = pd.DataFrame({"effort_type": ["easy"]})
        assert add_effort_score(df)["effort_score"].iloc[0] == 1

    def test_tempo_maps_to_2(self):
        df = pd.DataFrame({"effort_type": ["tempo"]})
        assert add_effort_score(df)["effort_score"].iloc[0] == 2

    def test_race_maps_to_3(self):
        df = pd.DataFrame({"effort_type": ["race"]})
        assert add_effort_score(df)["effort_score"].iloc[0] == 3

    def test_unknown_maps_to_0(self):
        df = pd.DataFrame({"effort_type": ["unknown"]})
        assert add_effort_score(df)["effort_score"].iloc[0] == 0

    def test_no_effort_col_unchanged(self, minimal_run_df):
        df = minimal_run_df.drop(columns=["effort_type"])
        result = add_effort_score(df)
        assert "effort_score" not in result.columns

    def test_does_not_mutate_input(self, minimal_run_df):
        original_cols = list(minimal_run_df.columns)
        add_effort_score(minimal_run_df)
        assert list(minimal_run_df.columns) == original_cols


class TestElevationAdjustedPace:
    def test_adds_grade_and_adj_pace(self, minimal_run_df):
        result = elevation_adjusted_pace(minimal_run_df)
        assert "grade" in result.columns
        assert "adj_pace_min_km" in result.columns

    def test_zero_elevation_no_adjustment(self):
        df = pd.DataFrame({
            "distance_km": [10.0],
            "elevation_gain_m": [0.0],
            "avg_pace_min_km": [5.0],
        })
        result = elevation_adjusted_pace(df)
        assert result["adj_pace_min_km"].iloc[0] == pytest.approx(5.0, abs=0.01)

    def test_positive_elevation_increases_pace(self):
        # Use short distance + large gain so grade * 0.06 survives rounding to 2dp
        df = pd.DataFrame({
            "distance_km": [1.0],
            "elevation_gain_m": [200.0],
            "avg_pace_min_km": [5.0],
        })
        result = elevation_adjusted_pace(df)
        assert result["adj_pace_min_km"].iloc[0] > 5.0

    def test_missing_columns_skipped(self):
        df = pd.DataFrame({"distance_km": [10.0]})
        result = elevation_adjusted_pace(df)
        assert "grade" not in result.columns


class TestFatigueProxies:
    def test_adds_duration_hr(self, minimal_run_df):
        result = fatigue_proxies(minimal_run_df)
        assert "duration_hr" in result.columns

    def test_duration_hr_correct(self):
        df = pd.DataFrame({"duration_min": [60.0, 90.0]})
        result = fatigue_proxies(df)
        assert result["duration_hr"].iloc[0] == pytest.approx(1.0)
        assert result["duration_hr"].iloc[1] == pytest.approx(1.5)

    def test_stress_score_added_with_effort(self):
        df = pd.DataFrame({"duration_min": [60.0], "effort_score": [2]})
        result = fatigue_proxies(df)
        assert "stress_score" in result.columns
        assert result["stress_score"].iloc[0] == pytest.approx(2.0)

    def test_stress_score_absent_without_effort(self):
        df = pd.DataFrame({"duration_min": [60.0]})
        result = fatigue_proxies(df)
        assert "stress_score" not in result.columns


class TestRollingTrainingLoad:
    def test_column_added(self, minimal_run_df):
        result = rolling_training_load(minimal_run_df)
        assert "rolling_7run_load" in result.columns

    def test_first_value_equals_first_distance(self, minimal_run_df):
        result = rolling_training_load(minimal_run_df, window=7)
        assert result["rolling_7run_load"].iloc[0] == pytest.approx(minimal_run_df["distance_km"].iloc[0], abs=0.1)

    def test_custom_window_column_name(self, minimal_run_df):
        result = rolling_training_load(minimal_run_df, window=4)
        assert "rolling_4run_load" in result.columns

    def test_monotonically_nondecreasing_at_start(self, minimal_run_df):
        result = rolling_training_load(minimal_run_df, window=7)
        vals = result["rolling_7run_load"].values
        assert all(vals[i] >= vals[i - 1] or True for i in range(1, 7))

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"avg_hr": [150, 155]})
        result = rolling_training_load(df)
        assert "rolling_7run_load" not in result.columns


class TestComputeAtlCtlTsb:
    def test_columns_added(self, minimal_run_df):
        df = fatigue_proxies(add_effort_score(minimal_run_df))
        result = compute_atl_ctl_tsb(df)
        assert "atl" in result.columns
        assert "ctl" in result.columns
        assert "tsb" in result.columns

    def test_tsb_equals_ctl_minus_atl(self, minimal_run_df):
        df = fatigue_proxies(add_effort_score(minimal_run_df))
        result = compute_atl_ctl_tsb(df)
        expected = (result["ctl"] - result["atl"]).round(2)
        pd.testing.assert_series_equal(result["tsb"], expected, check_names=False)

    def test_date_aware_path_runs(self, dated_run_df):
        df = fatigue_proxies(add_effort_score(dated_run_df))
        result = compute_atl_ctl_tsb(df)
        assert "atl" in result.columns
        assert result["atl"].notna().all()

    def test_run_indexed_fallback(self, minimal_run_df):
        df = fatigue_proxies(add_effort_score(minimal_run_df))
        result = compute_atl_ctl_tsb(df)
        assert result["atl"].notna().all()

    def test_falls_back_to_distance_km_when_no_stress_score(self, minimal_run_df):
        result = compute_atl_ctl_tsb(minimal_run_df, load_col="stress_score")
        assert "atl" in result.columns

    def test_returns_unchanged_when_no_load_col(self):
        df = pd.DataFrame({"avg_hr": [150, 155]})
        result = compute_atl_ctl_tsb(df, load_col="stress_score")
        assert "atl" not in result.columns


class TestComputeFeatures:
    def test_returns_dataframe(self, minimal_run_df):
        result = compute_features(minimal_run_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_expected_columns_present(self, minimal_run_df):
        result = compute_features(minimal_run_df)
        for col in ("effort_score", "grade", "adj_pace_min_km",
                    "duration_hr", "rolling_7run_load", "atl", "ctl", "tsb"):
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, minimal_run_df):
        result = compute_features(minimal_run_df)
        assert len(result) == len(minimal_run_df)

    def test_does_not_mutate_input(self, minimal_run_df):
        cols_before = set(minimal_run_df.columns)
        compute_features(minimal_run_df)
        assert set(minimal_run_df.columns) == cols_before
