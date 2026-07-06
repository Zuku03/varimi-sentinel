"""Smoke tests: package imports and configuration resolve."""

import varimi
from varimi import config


def test_package_version():
    assert varimi.__version__ == "0.1.0"


def test_dataset_path_configured():
    assert config.DATA_02.name == "02_agriculture_climate_market_signals.csv"


def test_feature_target_disjoint():
    # A feature must never also be a declared target / leakage column.
    features = set(config.NUM_FEATURES) | set(config.CAT_FEATURES)
    assert features.isdisjoint(set(config.LEAKAGE_COLS))
