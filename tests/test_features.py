import pandas as pd
import pytest

from nba_draft_features.features import add_age_at_season, add_per40_rates, add_shot_profile_ratios, add_interaction_features


def test_add_age_at_season_computes_reasonable_age():
    df = pd.DataFrame({"dob": ["2000-01-01"], "year": [2020]})
    out = add_age_at_season(df)
    # Season starts Nov 1, 2019 -> ~19.83 years old
    assert out["age_season"].iloc[0] == pytest.approx(19.83, abs=0.05)


def test_add_age_at_season_drops_dob_by_default():
    df = pd.DataFrame({"dob": ["2000-01-01"], "year": [2020]})
    out = add_age_at_season(df)
    assert "dob" not in out.columns


def test_add_per40_rates_handles_zero_minutes():
    df = pd.DataFrame({"mp": [0, 20], "pts": [0, 10], "treb": [0, 5], "ast": [0, 2]})
    out = add_per40_rates(df)
    assert pd.isna(out["pts_per40"].iloc[0])
    assert out["pts_per40"].iloc[1] == pytest.approx(20.0)


def test_add_shot_profile_ratios_renames_columns():
    df = pd.DataFrame({
        "rimmade/(rimmade+rimmiss)": [0.6],
        "dunksmade/(dunksmade+dunksmiss)": [0.8],
    })
    out = add_shot_profile_ratios(df)
    assert out["rim_rate"].iloc[0] == 0.6
    assert out["dunk_rate"].iloc[0] == 0.8


def test_add_interaction_features_computes_products():
    df = pd.DataFrame({
        "usg": [20.0], "TS_per": [0.55], "stl_per": [2.0], "blk_per": [3.0],
    })
    out = add_interaction_features(df)
    assert out["usage_efficiency"].iloc[0] == pytest.approx(11.0)
    assert out["defensive_versatility"].iloc[0] == pytest.approx(6.0)


def test_add_interaction_features_handles_missing_columns():
    df = pd.DataFrame({"usg": [20.0], "TS_per": [0.55]})
    out = add_interaction_features(df)
    assert "usage_efficiency" in out.columns
    assert "defensive_versatility" not in out.columns
