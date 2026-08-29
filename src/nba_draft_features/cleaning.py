"""Data-cleaning utilities for raw college basketball player-season data."""

from __future__ import annotations

import numpy as np
import pandas as pd

_VALID_CLASS_YEARS = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}


def ht_to_inches(ht: str | float) -> float:
    """Convert a "feet-inches" height string (e.g. "6-7") to total inches.

    Returns NaN for missing values or the "-" sentinel used in the raw data.
    """
    if pd.isna(ht) or ht == "-":
        return np.nan
    try:
        feet, inches = str(ht).split("-")
        return int(feet) * 12 + int(inches)
    except (ValueError, AttributeError):
        return np.nan


def clean_height(df: pd.DataFrame, ht_col: str = "ht", out_col: str = "ht_in") -> pd.DataFrame:
    """Add a numeric height-in-inches column parsed from a "feet-inches" string column."""
    df = df.copy()
    df[out_col] = df[ht_col].apply(ht_to_inches)
    return df


def clean_class_year(df: pd.DataFrame, yr_col: str = "yr", out_col: str = "class_yr_ord") -> pd.DataFrame:
    """Map class-year strings (Fr/So/Jr/Sr) to an ordinal 1-4 scale.

    Any value outside {Fr, So, Jr, Sr} (e.g. corrupted export values such as
    "57.1") is treated as missing rather than guessed at.
    """
    df = df.copy()
    df[out_col] = df[yr_col].map(_VALID_CLASS_YEARS)
    return df
