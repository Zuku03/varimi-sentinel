"""Tests for feature engineering and the time-aware split."""

from varimi import config
from varimi.data import loader
from varimi.features import build


def test_season_values_valid():
    df = build.add_season(loader.load_raw())
    assert set(df["season"].unique()) <= {"wet", "cool_dry", "hot_dry"}
    assert df["season"].notna().all()


def test_price_direction_values_valid():
    df = build.build(loader.load_raw())
    assert set(df["price_direction"].unique()) <= set(config.PRICE_DIR_CLASSES)


def test_feature_matrix_columns():
    df = build.build(loader.load_raw())
    X = build.feature_matrix(df)
    assert list(X.columns) == config.NUM_FEATURES + config.CAT_FEATURES


def test_time_aware_split_no_month_overlap():
    df = build.build(loader.load_raw())
    train, test = build.time_aware_split(df, test_months=2)
    assert set(train["month"]).isdisjoint(set(test["month"]))
    assert len(train) + len(test) == len(df)
