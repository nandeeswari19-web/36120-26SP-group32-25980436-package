import numpy as np
import pandas as pd
import pytest

from nba_draft_features.aggregation import aggregate_daily, DAILY_AGG_RULES


def _hourly_two_days():
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=48, freq="h"),
        "temperature_2m": np.arange(48, dtype=float),
        "relative_humidity_2m": [50.0] * 48,
        "wind_speed_10m": [5.0] * 48,
        "cloud_cover": [10.0] * 48,
        "precipitation": [1.0] * 48,
        "wind_gusts_10m": list(range(48)),
        "snowfall": [0.5] * 48,
    })


def test_aggregate_daily_row_count():
    out = aggregate_daily(_hourly_two_days())
    assert len(out) == 2


def test_aggregate_daily_mean_columns():
    out = aggregate_daily(_hourly_two_days())
    # day 1 temperature_2m hours 0..23 -> mean 11.5
    assert out["temperature_2m"].iloc[0] == pytest.approx(11.5)
    assert out["relative_humidity_2m"].iloc[0] == pytest.approx(50.0)


def test_aggregate_daily_sum_columns():
    out = aggregate_daily(_hourly_two_days())
    assert out["precipitation"].iloc[0] == pytest.approx(24.0)
    assert out["snowfall"].iloc[0] == pytest.approx(12.0)


def test_aggregate_daily_max_column():
    out = aggregate_daily(_hourly_two_days())
    assert out["wind_gusts_10m"].iloc[0] == 23
    assert out["wind_gusts_10m"].iloc[1] == 47


def test_aggregate_daily_missing_datetime_col_raises():
    df = _hourly_two_days().drop(columns=["time"])
    with pytest.raises(ValueError):
        aggregate_daily(df)


def test_aggregate_daily_partial_columns_only_aggregates_whats_present():
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=24, freq="h"),
        "temperature_2m": [20.0] * 24,
    })
    out = aggregate_daily(df)
    assert "temperature_2m" in out.columns
    assert "precipitation" not in out.columns


def test_aggregate_daily_no_known_columns_raises():
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=24, freq="h"),
        "some_unrelated_col": [1.0] * 24,
    })
    with pytest.raises(ValueError):
        aggregate_daily(df)


def test_aggregate_daily_sorted_by_date():
    df = _hourly_two_days().iloc[::-1].reset_index(drop=True)  # shuffle order
    out = aggregate_daily(df)
    assert out["date"].is_monotonic_increasing
