# Synthetic Data Validation Report

Generator: **Gaussian copula** over the continuous block (`src/varimi/synth/copula.py`); categoricals sampled from empirical
marginals. Method fully disclosed in `docs/DATASET_STATEMENT.md`.

Real rows: 360. Synthetic rows compared: 360.

## 1. Marginal fidelity - two-sample KS tests

A high p-value / low statistic means the synthetic marginal is statistically indistinguishable from the real one.

| Continuous column | KS statistic | p-value | Pass (p>0.05) |
|---|---|---|---|
| rainfall_mm | 0.0306 | 0.9961 | yes |
| ndvi_proxy_0_1 | 0.0333 | 0.9885 | yes |
| pest_incidents_reported | 0.0194 | 1.0000 | yes |
| irrigation_coverage_pct | 0.0611 | 0.5127 | yes |
| input_availability_score_0_100 | 0.0500 | 0.7599 | yes |
| climate_crop_risk_score_0_100 | 0.0417 | 0.9141 | yes |
| estimated_yield_t_per_ha | 0.0500 | 0.7599 | yes |

## 2. Dependence fidelity - correlation-matrix delta

- Frobenius norm of (real_corr - synth_corr): **0.4704**
- Max absolute pairwise correlation delta: **0.1704**

Interpretation: values near 0 indicate the copula preserved the real inter-feature correlation structure. See `correlation_delta_heatmap.png`.

## 3. Downstream check

A train-on-synthetic / test-on-real (TSTR) comparison is reported in the model evaluation (`reports/model_report.md`, Phase 3): the risk classifier trained on real+synthetic must not underperform the real-only baseline on the held-out real test months.
