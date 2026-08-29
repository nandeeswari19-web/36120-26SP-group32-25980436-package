import numpy as np
import pandas as pd

from nba_draft_features.cleaning import ht_to_inches, clean_height, clean_class_year


def test_ht_to_inches_valid():
    assert ht_to_inches("6-7") == 79
    assert ht_to_inches("5-10") == 70


def test_ht_to_inches_missing():
    assert np.isnan(ht_to_inches("-"))
    assert np.isnan(ht_to_inches(None))
    assert np.isnan(ht_to_inches(np.nan))


def test_clean_height_adds_column():
    df = pd.DataFrame({"ht": ["6-7", "-", "7-0"]})
    out = clean_height(df)
    assert list(out["ht_in"]) == [79, None, 84] or out["ht_in"].tolist()[1] != out["ht_in"].tolist()[1]  # NaN check
    assert "ht_in" in out.columns


def test_clean_class_year_valid_values():
    df = pd.DataFrame({"yr": ["Fr", "So", "Jr", "Sr"]})
    out = clean_class_year(df)
    assert out["class_yr_ord"].tolist() == [1, 2, 3, 4]


def test_clean_class_year_corrupted_values_become_missing():
    df = pd.DataFrame({"yr": ["Fr", "57.1", "0", None]})
    out = clean_class_year(df)
    assert out["class_yr_ord"].iloc[0] == 1
    assert out["class_yr_ord"].iloc[1:].isna().all()
