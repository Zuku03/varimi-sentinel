"""Gaussian-copula synthesizer (fully disclosed, dependency-light).

Method (standard Gaussian copula over continuous columns):
  1. Rank-transform each continuous column to uniform scores u = (rank-0.5)/n.
  2. Map to standard-normal scores z = Phi^-1(u).
  3. Estimate the correlation matrix R of the z-scores (with a tiny ridge for
     positive-definiteness).
  4. To sample: draw z* ~ N(0, R), map back to uniforms via Phi(z*), then invert
     each column through its empirical quantile function. This reproduces each
     marginal distribution *exactly* and preserves rank correlations.

Categorical columns are sampled independently from their empirical marginals.
This simplification is disclosed in reports/synthetic_validation.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class GaussianCopulaSynthesizer:
    def __init__(
        self,
        continuous_cols: list[str],
        categorical_cols: list[str],
        integer_cols: tuple[str, ...] = (),
        seed: int = 42,
    ):
        self.continuous_cols = list(continuous_cols)
        self.categorical_cols = list(categorical_cols)
        self.integer_cols = set(integer_cols)
        self.seed = seed
        self._quantile_data: dict[str, np.ndarray] = {}
        self._corr: np.ndarray | None = None
        self._cat_dist: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, df: pd.DataFrame) -> GaussianCopulaSynthesizer:
        cont = df[self.continuous_cols].to_numpy(dtype=float)
        n, d = cont.shape
        z = np.empty_like(cont)
        for j, col in enumerate(self.continuous_cols):
            values = cont[:, j]
            self._quantile_data[col] = np.sort(values)
            ranks = stats.rankdata(values, method="average")
            u = (ranks - 0.5) / n
            z[:, j] = stats.norm.ppf(u)
        corr = np.corrcoef(z, rowvar=False)
        corr = corr + np.eye(d) * 1e-6  # ridge for numerical PD
        self._corr = corr
        for col in self.categorical_cols:
            vc = df[col].value_counts(normalize=True)
            self._cat_dist[col] = (vc.index.to_numpy(), vc.to_numpy())
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._corr is None:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(self.seed)
        d = len(self.continuous_cols)
        z = rng.multivariate_normal(np.zeros(d), self._corr, size=n)
        u = stats.norm.cdf(z)
        out: dict[str, np.ndarray] = {}
        for j, col in enumerate(self.continuous_cols):
            samples = np.quantile(self._quantile_data[col], u[:, j])
            if col in self.integer_cols:
                samples = np.round(samples)
            out[col] = samples
        for col in self.categorical_cols:
            cats, probs = self._cat_dist[col]
            out[col] = rng.choice(cats, size=n, p=probs)
        return pd.DataFrame(out)
