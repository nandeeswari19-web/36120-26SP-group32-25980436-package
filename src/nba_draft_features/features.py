"""Feature-engineering utilities for the NBA sustained-career-prediction model."""

from __future__ import annotations

import pandas as pd


def add_age_at_season(
    df: pd.DataFrame,
    dob_col: str = "dob",
    year_col: str = "year",
    out_col: str = "age_season",
    drop_dob: bool = True,
) -> pd.DataFrame:
    """Compute a player's age (in years) at the start of their college season.

    The NCAA season is approximated to start on Nov 1 of ``year - 1`` for a
    season labelled ``year``.
    """
    df = df.copy()
    dob = pd.to_datetime(df[dob_col], errors="coerce")
    season_start = pd.to_datetime(
        (df[year_col] - 1).astype("Int64").astype(str) + "-11-01", errors="coerce"
    )
    df[out_col] = (season_start - dob).dt.days / 365.25
    if drop_dob:
        df = df.drop(columns=[dob_col])
    return df


def add_per40_rates(
    df: pd.DataFrame,
    minutes_col: str = "mp",
    counting_cols: tuple[str, ...] = ("pts", "treb", "ast"),
) -> pd.DataFrame:
    """Rescale raw counting stats to a per-40-minutes rate, removing playing-time bias."""
    df = df.copy()
    minutes = df[minutes_col].replace(0, pd.NA)
    for col in counting_cols:
        df[f"{col}_per40"] = df[col] / minutes * 40
    return df


def add_shot_profile_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw rim/dunk finishing-efficiency ratio columns for clarity."""
    df = df.copy()
    if "rimmade/(rimmade+rimmiss)" in df.columns:
        df["rim_rate"] = df["rimmade/(rimmade+rimmiss)"]
    if "dunksmade/(dunksmade+dunksmiss)" in df.columns:
        df["dunk_rate"] = df["dunksmade/(dunksmade+dunksmiss)"]
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit multiplicative interaction terms between related stats.

    ``usage_efficiency`` (usg * TS_per) separates high-usage-but-inefficient players from
    high-usage-and-efficient ones, a distinction a linear model can't represent with the two
    terms alone. ``defensive_versatility`` (stl_per * blk_per) rewards two-way defensive
    contribution over a player who is strong in only one of the two rates.
    """
    df = df.copy()
    if "usg" in df.columns and "TS_per" in df.columns:
        df["usage_efficiency"] = df["usg"] * df["TS_per"]
    if "stl_per" in df.columns and "blk_per" in df.columns:
        df["defensive_versatility"] = df["stl_per"] * df["blk_per"]
    return df
