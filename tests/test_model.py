"""Tests for the multi-head model mechanics."""

from varimi import config
from varimi.data import loader
from varimi.features import build
from varimi.model.pipeline import FEATURE_ORDER, VarimiModel


def test_model_fit_predict_contract():
    enriched = build.build(loader.load_raw())
    train, test = build.time_aware_split(enriched, test_months=2)
    model = VarimiModel()
    model.fit_encoder(enriched)
    model.fit(train, price_df=train)
    preds = model.predict(test)

    assert set(preds) == {config.TARGET_RISK, config.TARGET_YIELD, config.TARGET_PRICE_DIR}
    assert len(preds[config.TARGET_RISK]) == len(test)
    assert set(preds[config.TARGET_RISK]) <= set(config.RISK_CLASSES)
    assert set(preds[config.TARGET_PRICE_DIR]) <= set(config.PRICE_DIR_CLASSES)


def test_encode_width_matches_feature_order():
    enriched = build.build(loader.load_raw())
    model = VarimiModel()
    model.fit_encoder(enriched)
    x = model.encode(enriched.head(5))
    assert x.shape == (5, len(FEATURE_ORDER))