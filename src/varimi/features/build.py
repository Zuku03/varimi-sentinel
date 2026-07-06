"""Feature engineering for VaRimi Sentinel.

Turns the raw dataset-02 frame into a model-ready feature matrix plus the three
prediction targets (risk band, yield, farmgate price direction), and provides a
time-aware train/test split that prevents leakage across reporting months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from varimi import config

# Zimbabwe cropping-calendar season buckets (month number -> season label).
_SEASON_BY_MONTH = {
    11: "wet", 12: "wet", 1: "wet", 2: "wet", 3: "wet",   # rainfed summer crop
    4: "cool_dry", 5: "cool_dry", 6: "cool_dry", 7: "cool_dry",  # post-harvest
    8: "hot_dry", 9: "hot_dry", 10: "hot_dry",            # land prep / pre-season
}

# Month-on-month farmgate price change thresholds for the direction target.
_PRICE_UP = 2.0    # >= +2%  -> "up"
_PRICE_DOWN = -2.0  # <= -2% -> "down"; otherwise "flat"


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical ``season`` derived from the ``month`` (YYYY-MM) field."""
    df = df.copy()
    month_num = pd.to_datetime(df["month"], format="%Y-%m").dt.month
    df["season"] = month_num.map(_SEASON_BY_MONTH).astype("object")
    return df


def add_price_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``price_direction`` = sign of month-on-month farmgate price change.

    Computed within each (province, crop) series ordered by month. The first
    observation in each series has no prior month and is labelled ``flat``.
    """
    df = df.copy()
    df = df.sort_values(["province", "crop", "month"]).reset_index(drop=True)
    pct = (
        df.groupby(["province", "crop"])["avg_farmgate_price_usd_per_tonne"]
        .pct_change()
        .mul(100.0)
    )

    def _label(x: float) -> str:
        if np.isnan(x):
            return "flat"
        if x >= _PRICE_UP:
            return "up"
        if x <= _PRICE_DOWN:
            return "down"
        return "flat"

    df["price_direction"] = pct.map(_label).astype("object")
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all engineering steps and return an enriched frame."""
    return add_price_direction(add_season(df))


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model input columns (numeric + categorical features)."""
    cols = config.NUM_FEATURES + config.CAT_FEATURES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Enriched frame missing feature columns: {missing}. Call build() first.")
    return df[cols].copy()


def time_aware_split(df: pd.DataFrame, test_months: int = 2):
    """Split by reporting month, holding out the last ``test_months`` months.

    Prevents temporal leakage: the model is always evaluated on months strictly
    later than any it trained on.
    """
    months = sorted(df["month"].unique())
    if len(months) <= test_months:
        raise ValueError(
            f"Need > {test_months} distinct months to split; found {len(months)}"
        )
    cutoff = set(months[-test_months:])
    test_mask = df["month"].isin(cutoff)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
