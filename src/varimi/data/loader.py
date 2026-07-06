"""Load and validate dataset 02 (agriculture climate & market signals).

The dataset is a *synthetic aggregate* sample from the POTRAZ AI4I Design-Track
pack. It is not official statistics. See docs/DATASET_STATEMENT.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from varimi import config

EXPECTED_COLUMNS = [
    "month",
    "province",
    "district",
    "latitude",
    "longitude",
    "crop",
    "rainfall_mm",
    "ndvi_proxy_0_1",
    "pest_incidents_reported",
    "irrigation_coverage_pct",
    "input_availability_score_0_100",
    "avg_farmgate_price_usd_per_tonne",
    "estimated_yield_t_per_ha",
    "climate_crop_risk_score_0_100",
    "risk_level",
]

# Inclusive sanity ranges used for a lightweight data-contract check.
RANGE_CHECKS = {
    "ndvi_proxy_0_1": (0.0, 1.0),
    "irrigation_coverage_pct": (0.0, 100.0),
    "input_availability_score_0_100": (0.0, 100.0),
    "climate_crop_risk_score_0_100": (0.0, 100.0),
    "rainfall_mm": (0.0, 2000.0),
    "estimated_yield_t_per_ha": (0.0, 50.0),
    "pest_incidents_reported": (0.0, 10000.0),
}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read dataset 02 and enforce the schema + range contract.

    Raises ValueError if columns are missing or numeric values fall outside
    their documented ranges.
    """
    path = Path(path) if path is not None else config.DATA_02
    if not path.exists():
        raise FileNotFoundError(f"Dataset 02 not found at {path}")
    df = pd.read_csv(path)
    _validate_schema(df)
    _validate_ranges(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset 02 is missing expected columns: {missing}")


def _validate_ranges(df: pd.DataFrame) -> None:
    for col, (lo, hi) in RANGE_CHECKS.items():
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            raise ValueError(f"Column {col!r} contains non-numeric values")
        out_of_range = series[(series < lo) | (series > hi)]
        if not out_of_range.empty:
            raise ValueError(
                f"Column {col!r} has {len(out_of_range)} values outside [{lo}, {hi}]"
            )
