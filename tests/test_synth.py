"""Tests for the Gaussian-copula synthesizer and augmentation."""

import numpy as np

from varimi.data import loader
from varimi.features import build
from varimi.synth import augment
from varimi.synth.copula import GaussianCopulaSynthesizer


def _real():
    return build.add_season(loader.load_raw())


def test_sample_preserves_marginal_range():
    real = _real()
    synth = GaussianCopulaSynthesizer(
        augment.CONTINUOUS, augment.CATEGORICAL, augment.INTEGER, seed=1
    )
    synth.fit(real)
    s = synth.sample(200)
    # Empirical-quantile inversion keeps samples within the observed support.
    assert s["ndvi_proxy_0_1"].min() >= real["ndvi_proxy_0_1"].min() - 1e-9
    assert s["ndvi_proxy_0_1"].max() <= real["ndvi_proxy_0_1"].max() + 1e-9


def test_integer_column_is_integer():
    real = _real()
    s = augment.make_synthetic(real, n=100, seed=2)
    vals = s["pest_incidents_reported"].to_numpy()
    assert np.allclose(vals, np.round(vals))


def test_risk_labels_valid():
    real = _real()
    s = augment.make_synthetic(real, n=100, seed=3)
    assert set(s["risk_level"].unique()) <= {"Low", "Medium", "High"}


def test_augmented_frame_grows():
    train, _ = build.time_aware_split(build.build(loader.load_raw()), test_months=2)
    aug = augment.augment_training_frame(train, ratio=1.0, seed=4)
    assert len(aug) > len(train)
