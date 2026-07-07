"""VaRimi multi-head model: risk band, yield outlook, farmgate price direction.

Design choices tuned for the AI4I Track-3 edge use-case:
  * A single shared feature matrix (5 numeric + 3 categorical columns) feeds three
    HistGradientBoosting heads. HGB is chosen over deep nets because the data is
    small, tabular and heterogeneous - gradient-boosted trees are the right-sized
    method (no "sledgehammer"), they train in milliseconds and export to a tiny
    ONNX graph for edge inference.
  * Categoricals (province / crop / season) are ordinal-encoded and treated by the
    tree ensemble as ordered numeric codes. We deliberately do NOT use HGB's native
    categorical split support: it does not round-trip through skl2onnx (the ONNX
    TreeEnsemble would mis-predict), and on this data the ordinal-numeric treatment
    scores at least as well. This keeps the trained model and its ONNX edge export
    bit-for-bit faithful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from varimi import config

FEATURE_ORDER = config.NUM_FEATURES + config.CAT_FEATURES

_HGB_KWARGS = dict(
    learning_rate=0.08,
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=15,
    l2_regularization=1.0,
)


def _new_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, **_HGB_KWARGS)


def _new_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed, **_HGB_KWARGS)


class VarimiModel:
    """Bundle of the encoder plus the three prediction heads."""

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.seed = seed
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.risk_clf: HistGradientBoostingClassifier | None = None
        self.yield_reg: HistGradientBoostingRegressor | None = None
        self.price_clf: HistGradientBoostingClassifier | None = None

    def fit_encoder(self, vocab_df: pd.DataFrame) -> None:
        self.encoder.fit(vocab_df[config.CAT_FEATURES])

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Return the float feature matrix in FEATURE_ORDER (ONNX input layout)."""
        num = df[config.NUM_FEATURES].to_numpy(dtype=float)
        cat = self.encoder.transform(df[config.CAT_FEATURES]).astype(float)
        return np.hstack([num, cat])

    def fit(self, risk_yield_df: pd.DataFrame, price_df: pd.DataFrame) -> VarimiModel:
        x_ry = self.encode(risk_yield_df)
        self.risk_clf = _new_classifier(self.seed).fit(x_ry, risk_yield_df[config.TARGET_RISK])
        self.yield_reg = _new_regressor(self.seed).fit(x_ry, risk_yield_df[config.TARGET_YIELD])
        x_p = self.encode(price_df)
        self.price_clf = _new_classifier(self.seed).fit(x_p, price_df[config.TARGET_PRICE_DIR])
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x = self.encode(df)
        return {
            config.TARGET_RISK: self.risk_clf.predict(x),
            config.TARGET_YIELD: self.yield_reg.predict(x),
            config.TARGET_PRICE_DIR: self.price_clf.predict(x),
        }