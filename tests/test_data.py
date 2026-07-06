"""Tests for the dataset-02 loader and its data contract."""

import pandas as pd
import pytest

from varimi.data import loader


def test_load_raw_shape_and_columns():
    df = loader.load_raw()
    assert len(df) == 360
    for col in loader.EXPECTED_COLUMNS:
        assert col in df.columns


def test_missing_column_raises():
    bad = pd.DataFrame({"month": ["2026-01"]})
    with pytest.raises(ValueError, match="missing expected columns"):
        loader._validate_schema(bad)


def test_range_violation_raises():
    df = loader.load_raw()
    df.loc[0, "ndvi_proxy_0_1"] = 5.0  # out of [0, 1]
    with pytest.raises(ValueError, match="outside"):
        loader._validate_ranges(df)
