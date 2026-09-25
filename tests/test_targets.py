import numpy as np
import pandas as pd
import pytest

from nba_draft_features.targets import (
    add_cci,
    add_whi,
    add_whc,
    whi_to_category,
    whi_to_class,
    shift_target,
)


# ---------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------

def test_add_cci_perfect_day_is_100():
    df = pd.DataFrame({
        "temperature_2m": [22.0],
        "relative_humidity_2m": [50.0],
        "wind_speed_10m": [10.0],
        "cloud_cover": [0.0],
        "precipitation": [0.0],
    })
    out = add_cci(df)
    assert out["cci"].iloc[0] == pytest.approx(100.0)


def test_add_cci_extreme_day_is_low():
    df = pd.DataFrame({
        "temperature_2m": [45.0],   # far from 22
        "relative_humidity_2m": [100.0],
        "wind_speed_10m": [60.0],
        "cloud_cover": [100.0],
        "precipitation": [30.0],
    })
    out = add_cci(df)
    assert out["cci"].iloc[0] < 20


def test_add_cci_scores_never_go_negative():
    # extreme values that would push a raw score below 0 without clipping
    df = pd.DataFrame({
        "temperature_2m": [200.0],
        "relative_humidity_2m": [50.0],
        "wind_speed_10m": [10.0],
        "cloud_cover": [0.0],
        "precipitation": [0.0],
    })
    out = add_cci(df)
    assert out["temp_score"].iloc[0] == 0.0
    assert out["cci"].iloc[0] >= 0.0


def test_add_cci_missing_column_raises():
    df = pd.DataFrame({"temperature_2m": [22.0]})
    with pytest.raises(ValueError):
        add_cci(df)


# ---------------------------------------------------------------------
# WHI / WHC
# ---------------------------------------------------------------------

def test_add_whi_calm_day_is_zero():
    df = pd.DataFrame({
        "precipitation": [0.0],
        "wind_gusts_10m": [0.0],
        "cloud_cover": [0.0],
        "snowfall": [0.0],
        "temperature_2m": [22.0],
    })
    out = add_whi(df)
    assert out["whi"].iloc[0] == pytest.approx(0.0)


def test_add_whi_hazards_capped_at_one():
    df = pd.DataFrame({
        "precipitation": [1000.0],
        "wind_gusts_10m": [1000.0],
        "cloud_cover": [100.0],
        "snowfall": [1000.0],
        "temperature_2m": [200.0],
    })
    out = add_whi(df)
    assert out["rain_hazard"].iloc[0] == 1.0
    assert out["wind_hazard"].iloc[0] == 1.0
    assert out["whi"].iloc[0] == pytest.approx(100.0)


@pytest.mark.parametrize(
    "whi_value,expected_label,expected_class",
    [
        (0.0, "Low Risk", 0),
        (24.9, "Low Risk", 0),
        (25.0, "Moderate Risk", 1),
        (49.9, "Moderate Risk", 1),
        (50.0, "High Risk", 2),
        (74.9, "High Risk", 2),
        (75.0, "Extreme Risk", 3),
        (100.0, "Extreme Risk", 3),
    ],
)
def test_whc_band_boundaries(whi_value, expected_label, expected_class):
    s = pd.Series([whi_value])
    assert whi_to_category(s).iloc[0] == expected_label
    assert whi_to_class(s).iloc[0] == expected_class


def test_add_whc_end_to_end():
    df = pd.DataFrame({
        "precipitation": [0.0],
        "wind_gusts_10m": [0.0],
        "cloud_cover": [0.0],
        "snowfall": [0.0],
        "temperature_2m": [22.0],
    })
    out = add_whc(df)
    assert out["whc_label"].iloc[0] == "Low Risk"
    assert out["whc_class"].iloc[0] == 0


def test_add_whi_missing_column_raises():
    df = pd.DataFrame({"precipitation": [0.0]})
    with pytest.raises(ValueError):
        add_whi(df)


# ---------------------------------------------------------------------
# shift_target
# ---------------------------------------------------------------------

def test_shift_target_basic():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "cci": [10, 20, 30, 40, 50],
    })
    shifted = shift_target(df, "cci", horizon_days=3)
    assert shifted.tolist()[:2] == [40, 50]
    assert shifted.iloc[2:].isna().all()


def test_shift_target_seven_day_horizon():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10),
        "whc_class": list(range(10)),
    })
    shifted = shift_target(df, "whc_class", horizon_days=7)
    assert shifted.iloc[0] == 7
    assert shifted.iloc[2] == 9
    assert shifted.iloc[3:].isna().all()


def test_shift_target_handles_unsorted_input():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5)[::-1],
        "cci": [50, 40, 30, 20, 10],
    })
    shifted = shift_target(df, "cci", horizon_days=1)
    # regardless of input order, result aligns back to df's original index
    assert len(shifted) == len(df)
