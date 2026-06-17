import pytest

from src.race_predictor import (
    format_finish_time,
    predict_for_target_time,
    RACE_DISTANCES_KM,
    RACE_LABELS,
)


class TestFormatFinishTime:
    def test_sub_hour_mm_ss(self):
        assert format_finish_time(25.5) == "25:30"

    def test_exactly_one_hour(self):
        assert format_finish_time(60.0) == "1:00:00"

    def test_one_hour_thirty(self):
        assert format_finish_time(90.0) == "1:30:00"

    def test_zero_minutes(self):
        assert format_finish_time(0.0) == "0:00"

    def test_negative_clamped_to_zero(self):
        assert format_finish_time(-5.0) == "0:00"

    def test_fractional_seconds(self):
        result = format_finish_time(5.5)
        assert result == "5:30"

    def test_marathon_finish_time(self):
        result = format_finish_time(210.0)
        assert result == "3:30:00"


class TestPredictForTargetTime:
    def test_5k_pace_arithmetic(self):
        result = predict_for_target_time(race="5k", target_time_min=25.0)
        expected_pace = 25.0 / 5.0
        assert result["required_pace_min_km"] == pytest.approx(expected_pace, abs=0.01)

    def test_marathon_pace_arithmetic(self):
        result = predict_for_target_time(race="marathon", target_time_min=210.0)
        expected_pace = 210.0 / 42.195
        assert result["required_pace_min_km"] == pytest.approx(expected_pace, abs=0.01)

    def test_returns_race_label(self):
        result = predict_for_target_time(race="10k", target_time_min=50.0)
        assert result["race"] == RACE_LABELS["10k"]

    def test_returns_distance_km(self):
        result = predict_for_target_time(race="half_marathon", target_time_min=100.0)
        assert result["distance_km"] == pytest.approx(RACE_DISTANCES_KM["half_marathon"])

    def test_finish_minutes_matches_input(self):
        result = predict_for_target_time(race="5k", target_time_min=30.0)
        assert result["finish_minutes"] == pytest.approx(30.0)

    def test_invalid_race_raises(self):
        with pytest.raises(KeyError):
            predict_for_target_time(race="ultra", target_time_min=600.0)


class TestRaceConstants:
    def test_all_distances_positive(self):
        for race, dist in RACE_DISTANCES_KM.items():
            assert dist > 0, f"{race} distance not positive"

    def test_marathon_longer_than_half(self):
        assert RACE_DISTANCES_KM["marathon"] > RACE_DISTANCES_KM["half_marathon"]

    def test_half_longer_than_10k(self):
        assert RACE_DISTANCES_KM["half_marathon"] > RACE_DISTANCES_KM["10k"]

    def test_labels_cover_all_distances(self):
        assert set(RACE_LABELS.keys()) == set(RACE_DISTANCES_KM.keys())
