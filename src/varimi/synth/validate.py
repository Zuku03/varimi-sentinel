"""Validate synthetic data against the real sample (C3 evidence generator).

Produces reports/synthetic_validation.md containing:
  - per-continuous-column two-sample Kolmogorov-Smirnov tests (marginal fidelity),
  - the real-vs-synthetic correlation-matrix delta (dependence fidelity),
and saves a correlation-delta heatmap PNG.

Run: python -m varimi.synth.validate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from varimi import config
from varimi.data import loader
from varimi.features import build
from varimi.synth.augment import CONTINUOUS, make_synthetic


def _corr(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.corrcoef(df[cols].to_numpy(dtype=float), rowvar=False)


def run(seed: int = 42) -> dict:
    real = build.add_season(loader.load_raw())
    synth = make_synthetic(real, n=len(real), seed=seed)

    ks_rows = []
    for col in CONTINUOUS:
        stat, pval = stats.ks_2samp(real[col].to_numpy(float), synth[col].to_numpy(float))
        ks_rows.append((col, stat, pval))

    real_corr = _corr(real, CONTINUOUS)
    synth_corr = _corr(synth, CONTINUOUS)
    delta = real_corr - synth_corr
    frob = float(np.linalg.norm(delta))
    max_abs = float(np.max(np.abs(delta)))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(ks_rows, frob, max_abs, len(real))
    _save_heatmap(delta, CONTINUOUS)
    return {"ks": ks_rows, "frobenius": frob, "max_abs_corr_delta": max_abs}


def _write_report(ks_rows, frob: float, max_abs: float, n: int) -> None:
    lines = [
        "# Synthetic Data Validation Report",
        "",
        "Generator: **Gaussian copula** over the continuous block "
        "(`src/varimi/synth/copula.py`); categoricals sampled from empirical",
        "marginals. Method fully disclosed in `docs/DATASET_STATEMENT.md`.",
        "",
        f"Real rows: {n}. Synthetic rows compared: {n}.",
        "",
        "## 1. Marginal fidelity - two-sample KS tests",
        "",
        "A high p-value / low statistic means the synthetic marginal is "
        "statistically indistinguishable from the real one.",
        "",
        "| Continuous column | KS statistic | p-value | Pass (p>0.05) |",
        "|---|---|---|---|",
    ]
    for col, stat, pval in ks_rows:
        ok = "yes" if pval > 0.05 else "no"
        lines.append(f"| {col} | {stat:.4f} | {pval:.4f} | {ok} |")

    lines += [
        "",
        "## 2. Dependence fidelity - correlation-matrix delta",
        "",
        f"- Frobenius norm of (real_corr - synth_corr): **{frob:.4f}**",
        f"- Max absolute pairwise correlation delta: **{max_abs:.4f}**",
        "",
        "Interpretation: values near 0 indicate the copula preserved the "
        "real inter-feature correlation structure. See "
        "`correlation_delta_heatmap.png`.",
        "",
        "## 3. Downstream check",
        "",
        "A train-on-synthetic / test-on-real (TSTR) comparison is reported in "
        "the model evaluation (`reports/model_report.md`, Phase 3): the risk "
        "classifier trained on real+synthetic must not underperform the "
        "real-only baseline on the held-out real test months.",
    ]
    (config.REPORTS_DIR / "synthetic_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _save_heatmap(delta: np.ndarray, cols: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(delta, cmap="coolwarm", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Real - Synthetic correlation delta")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "correlation_delta_heatmap.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    result = run()
    print(f"KS tests: {len(result['ks'])} columns")
    print(f"Frobenius corr delta: {result['frobenius']:.4f}")
    print(f"Max abs corr delta:   {result['max_abs_corr_delta']:.4f}")
