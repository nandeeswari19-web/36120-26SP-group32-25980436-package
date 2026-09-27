"""
AT2 target variable construction: Climate Comfort Index (CCI, regression)
and Weather Hazard Category (WHC, multiclass classification).

Formulas are fixed by the assignment spec — see the AT2 brief. Unlike
nba_draft_features.features.add_interaction_features (which skips a
derived feature if its source columns are missing, since those features
are optional extras), the component scores here are *mandatory*: a
missing input column means the target can't be trusted, so these
functions raise rather than silently dropping a weighted term.

Expects a daily-aggregated dataframe — run aggregation.aggregate_daily
first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Climate Comfort Index (CCI) — regression target, 3-day horizon
# ---------------------------------------------------------------------

CCI_REQUIRED_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
]

CCI_WEIGHTS = {
    "temp": 0.35,
    "humidity": 0.20,
    "wind": 0.15,
    "cloud": 0.15,
    "rain": 0.15,
}


def _require_cols(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required column(s) {missing}")


def add_cci(df: pd.DataFrame) -> pd.DataFrame:
    """Add component scores + Climate Comfort Index (0-100) to `df`.

    Adds columns: temp_score, humidity_score, wind_score, cloud_score,
    rain_score, cci.
    """
    _require_cols(df, CCI_REQUIRED_COLS, "add_cci")
    out = df.copy()

    out["temp_score"] = (1 - (out["temperature_2m"] - 22).abs() / 20).clip(lower=0)
    out["humidity_score"] = (1 - (out["relative_humidity_2m"] - 50).abs() / 50).clip(lower=0)
    out["wind_score"] = (1 - (out["wind_speed_10m"] - 10).abs() / 40).clip(lower=0)
    out["cloud_score"] = (1 - out["cloud_cover"] / 100).clip(lower=0)
    out["rain_score"] = (1 - out["precipitation"] / 20).clip(lower=0)

    out["cci"] = 100 * (
        CCI_WEIGHTS["temp"] * out["temp_score"]
        + CCI_WEIGHTS["humidity"] * out["humidity_score"]
        + CCI_WEIGHTS["wind"] * out["wind_score"]
        + CCI_WEIGHTS["cloud"] * out["cloud_score"]
        + CCI_WEIGHTS["rain"] * out["rain_score"]
    )
    return out


# ---------------------------------------------------------------------
# Weather Hazard Category (WHC) — multiclass target, exactly +7 days
# ---------------------------------------------------------------------

WHI_REQUIRED_COLS = [
    "precipitation",
    "wind_gusts_10m",
    "cloud_cover",
    "snowfall",
    "temperature_2m",
]

WHI_WEIGHTS = {
    "rain": 0.30,
    "wind": 0.30,
    "cloud": 0.20,
    "snow": 0.10,
    "temp": 0.10,
}

# (upper_bound_exclusive, label, class_id) — checked in order
WHC_BANDS = [
    (25, "Low Risk", 0),
    (50, "Moderate Risk", 1),
    (75, "High Risk", 2),
    (np.inf, "Extreme Risk", 3),
]


def add_whi(df: pd.DataFrame) -> pd.DataFrame:
    """Add hazard component scores + Weather Hazard Index (0-100) to `df`.

    Adds columns: rain_hazard, wind_hazard, cloud_hazard, snow_hazard,
    temp_hazard, whi.
    """
    _require_cols(df, WHI_REQUIRED_COLS, "add_whi")
    out = df.copy()

    out["rain_hazard"] = (out["precipitation"] / 30).clip(upper=1)
    out["wind_hazard"] = (out["wind_gusts_10m"] / 100).clip(upper=1)
    out["cloud_hazard"] = (out["cloud_cover"] / 100).clip(upper=1)
    out["snow_hazard"] = (out["snowfall"] / 15).clip(upper=1)
    out["temp_hazard"] = ((out["temperature_2m"] - 22).abs() / 25).clip(upper=1)

    out["whi"] = 100 * (
        WHI_WEIGHTS["rain"] * out["rain_hazard"]
        + WHI_WEIGHTS["wind"] * out["wind_hazard"]
        + WHI_WEIGHTS["cloud"] * out["cloud_hazard"]
        + WHI_WEIGHTS["snow"] * out["snow_hazard"]
        + WHI_WEIGHTS["temp"] * out["temp_hazard"]
    )
    return out


def whi_to_category(whi: pd.Series) -> pd.Series:
    """Map a WHI (0-100) Series to WHC labels: Low/Moderate/High/Extreme Risk."""
    labels = pd.Series(pd.NA, index=whi.index, dtype="object")
    lower = 0
    for upper, label, _class_id in WHC_BANDS:
        mask = (whi >= lower) & (whi < upper)
        labels[mask] = label
        lower = upper
    return labels


def whi_to_class(whi: pd.Series) -> pd.Series:
    """Map a WHI (0-100) Series to WHC integer classes 0-3 (Low..Extreme)."""
    classes = pd.Series(pd.NA, index=whi.index, dtype="Int64")
    lower = 0
    for upper, _label, class_id in WHC_BANDS:
        mask = (whi >= lower) & (whi < upper)
        classes[mask] = class_id
        lower = upper
    return classes


def add_whc(df: pd.DataFrame) -> pd.DataFrame:
    """Add whi, whc_label, whc_class columns to `df` (calls add_whi first)."""
    out = add_whi(df)
    out["whc_label"] = whi_to_category(out["whi"])
    out["whc_class"] = whi_to_class(out["whi"])
    return out


# ---------------------------------------------------------------------
# Horizon shifting — turn "today's index" into "the label to predict"
# ---------------------------------------------------------------------


def shift_target(
    df: pd.DataFrame,
    target_col: str,
    horizon_days: int,
    date_col: str = "date",
) -> pd.Series:
    """Shift `target_col` back by `horizon_days` so each row's label is
    the target value `horizon_days` in the future.

    Requires `df` to have one row per calendar day with no gaps, sorted
    by `date_col` ascending — reindex to a full date range first if your
    data has missing days, otherwise this will silently shift across a
    gap.

    CCI: horizon_days=3 gives a "next 3 days" value for day+3.
    WHC: horizon_days=7 gives the "exactly +7 days" value.

    Returns a Series aligned to `df`'s index; the last `horizon_days`
    rows will be NaN/NA (no future value exists yet).
    """
    out = df.sort_values(date_col).reset_index(drop=True)
    shifted = out[target_col].shift(-horizon_days)
    # re-align back to original df's index order
    shifted.index = out.index
    return shifted.reindex(df.index)
