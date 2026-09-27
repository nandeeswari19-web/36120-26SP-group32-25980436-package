"""
Aggregate hourly Open-Meteo weather observations into daily rows.

AT2 requires specific daily aggregations per variable before any target
variable (CCI, WHI) can be calculated:

    Variable            Daily aggregation
    ------------------  -----------------
    temperature_2m      mean
    relative_humidity_2m mean
    wind_speed_10m       mean
    cloud_cover           mean
    precipitation         sum
    wind_gusts_10m        max
    snowfall               sum

Only columns actually present in the input are aggregated — same
"skip what's missing, don't crash" pattern used in
nba_draft_features.features.add_interaction_features.
"""

from __future__ import annotations

import pandas as pd

# variable -> aggregation function name (as understood by pandas .agg)
DAILY_AGG_RULES: dict[str, str] = {
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "wind_speed_10m": "mean",
    "cloud_cover": "mean",
    "precipitation": "sum",
    "wind_gusts_10m": "max",
    "snowfall": "sum",
}


def aggregate_daily(
    df: pd.DataFrame,
    datetime_col: str = "time",
    agg_rules: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resample hourly weather data to one row per calendar day.

    Parameters
    ----------
    df : DataFrame with an hourly datetime column and one or more of the
        weather variables listed in DAILY_AGG_RULES.
    datetime_col : name of the hourly timestamp column. Parsed with
        pd.to_datetime if not already datetime64.
    agg_rules : optional override of {column: agg_func}. Defaults to
        DAILY_AGG_RULES. Only columns present in `df` are used — an
        agg_rules entry for a column that doesn't exist is silently
        skipped rather than raising.

    Returns
    -------
    DataFrame indexed by a new `date` column (date, not datetime), one
    row per day, containing only the aggregated columns that were
    actually available in the input.

    Raises
    ------
    ValueError if `datetime_col` is not in `df`, or if none of the
    configured variables are present (nothing to aggregate).
    """
    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' not found in dataframe columns")

    rules = agg_rules if agg_rules is not None else DAILY_AGG_RULES
    available_rules = {col: fn for col, fn in rules.items() if col in df.columns}

    if not available_rules:
        raise ValueError(
            "None of the expected weather variables "
            f"({list(rules)}) were found in the dataframe"
        )

    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col])
    out["date"] = out[datetime_col].dt.date

    daily = out.groupby("date", as_index=False).agg(available_rules)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)
