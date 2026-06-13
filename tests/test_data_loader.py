import pandas as pd
import pytest

from src.data_loader import (
    _snake_case,
    _time_to_minutes,
    _pace_to_min_per_km,
    _clean_numeric,
    parse_activities_csv,
)


class TestSnakeCase:
    def test_spaces_replaced(self):
        assert _snake_case("Avg HR") == "avg_hr"

    def test_slashes_replaced(self):
        assert _snake_case("Total Ascent/m") == "total_ascent_m"

    def test_already_snake(self):
        assert _snake_case("distance_km") == "distance_km"

    def test_leading_trailing_whitespace(self):
        assert _snake_case("  Title  ") == "title"

    def test_special_chars_stripped(self):
        assert _snake_case("Training Stress Score®") == "training_stress_score"

    def test_mixed_separators(self):
        assert _snake_case("Avg-Pace Min/Km") == "avg_pace_min_km"


class TestTimeToMinutes:
    def test_hms(self):
        assert _time_to_minutes("1:30:00") == pytest.approx(90.0)

    def test_ms(self):
        assert _time_to_minutes("45:30") == pytest.approx(45.5)

    def test_zero(self):
        assert _time_to_minutes("0:00") == pytest.approx(0.0)

    def test_nan_returns_none(self):
        assert _time_to_minutes(float("nan")) is None

    def test_invalid_string_returns_none(self):
        assert _time_to_minutes("not_a_time") is None

    def test_single_part_returns_none(self):
        assert _time_to_minutes("30") is None

    def test_seconds_fractional(self):
        result = _time_to_minutes("0:01:30")
        assert result == pytest.approx(1.5)


class TestPaceToMinPerKm:
    def test_mm_ss(self):
        assert _pace_to_min_per_km("5:30") == pytest.approx(5.5)

    def test_exactly_five_mins(self):
        assert _pace_to_min_per_km("5:00") == pytest.approx(5.0)

    def test_nan_returns_none(self):
        assert _pace_to_min_per_km(float("nan")) is None

    def test_empty_string_returns_none(self):
        assert _pace_to_min_per_km("") is None

    def test_three_parts_hms(self):
        assert _pace_to_min_per_km("0:05:30") == pytest.approx(5.5)

    def test_invalid_returns_none(self):
        assert _pace_to_min_per_km("fast") is None


class TestCleanNumeric:
    def test_integer_string(self):
        assert _clean_numeric("1234") == pytest.approx(1234.0)

    def test_comma_thousands(self):
        assert _clean_numeric("1,234") == pytest.approx(1234.0)

    def test_float_string(self):
        assert _clean_numeric("5.75") == pytest.approx(5.75)

    def test_nan_returns_none(self):
        assert _clean_numeric(float("nan")) is None

    def test_non_numeric_returns_none(self):
        assert _clean_numeric("abc") is None

    def test_numeric_passthrough(self):
        assert _clean_numeric(42.0) == pytest.approx(42.0)


class TestParseActivitiesCsv:
    def _garmin_df(self):
        return pd.DataFrame({
            "Activity Type": ["Running", "Running"],
            "Date": ["2024-01-01", "2024-01-02"],
            "Distance": [10.0, 5.0],
            "Time": ["1:00:00", "0:25:00"],
            "Avg HR": [145, 155],
            "Avg Pace": ["6:00", "5:00"],
            "Total Ascent": [100, 50],
            "Calories": [600, 300],
        })

    def test_returns_dataframe(self):
        result = parse_activities_csv(self._garmin_df())
        assert isinstance(result, pd.DataFrame)

    def test_distance_km_parsed(self):
        result = parse_activities_csv(self._garmin_df())
        assert "distance_km" in result.columns
        assert result["distance_km"].iloc[0] == pytest.approx(10.0)

    def test_duration_min_parsed(self):
        result = parse_activities_csv(self._garmin_df())
        assert "duration_min" in result.columns
        assert result["duration_min"].iloc[0] == pytest.approx(60.0)

    def test_avg_hr_parsed(self):
        result = parse_activities_csv(self._garmin_df())
        assert "avg_hr" in result.columns
        assert result["avg_hr"].iloc[0] == pytest.approx(145.0)

    def test_avg_pace_parsed(self):
        result = parse_activities_csv(self._garmin_df())
        assert "avg_pace_min_km" in result.columns
        assert result["avg_pace_min_km"].iloc[0] == pytest.approx(6.0)

    def test_elevation_parsed(self):
        result = parse_activities_csv(self._garmin_df())
        assert "elevation_gain_m" in result.columns
        assert result["elevation_gain_m"].iloc[0] == pytest.approx(100.0)

    def test_column_names_snake_case(self):
        result = parse_activities_csv(self._garmin_df())
        for col in result.columns:
            assert col == col.lower(), f"Column not lower-case: {col}"
            assert " " not in col, f"Column has spaces: {col}"

    def test_does_not_mutate_input(self):
        df = self._garmin_df()
        original_cols = list(df.columns)
        parse_activities_csv(df)
        assert list(df.columns) == original_cols

    def test_steps_comma_removal(self):
        df = pd.DataFrame({"steps": ["1,234", "5,678"]})
        result = parse_activities_csv(df)
        assert result["steps"].iloc[0] == 1234
        assert result["steps"].iloc[1] == 5678
