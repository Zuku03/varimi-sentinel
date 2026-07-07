# Model Card - VaRimi Sentinel

## Overview
Three gradient-boosted (HistGradientBoosting) heads sharing one 8-feature tabular
input, predicting per (district, crop, month):
- **Risk band** (Low / Medium / High) - classification
- **Yield outlook** (t/ha) - regression
- **Farmgate price direction** (down / flat / up) - classification

## Intended use
Decision support for agricultural extension officers and food-security planners,
surfaced through a low-end Android app or USSD/SMS advisory. **Not** a substitute
for agronomic ground-truth; outputs are advisory signals.

## Training data
`02_agriculture_climate_market_signals.csv` - synthetic aggregate (360 rows), not
official statistics. Risk/yield heads trained on real training months plus
disclosed Gaussian-copula synthetic rows; price head on real months only
(temporal target). See `docs/DATASET_STATEMENT.md` and
`reports/synthetic_validation.md`.

## Features / leakage controls
Inputs: rainfall_mm, ndvi_proxy_0_1, pest_incidents_reported,
irrigation_coverage_pct, input_availability_score_0_100 (numeric) + province,
crop, season (categorical). Raw targets and the continuous risk score are
excluded from inputs; evaluation uses a time-aware split (hold out latest months).

## Performance (held-out real months 2026-05, 2026-06; n=120)
| Head | Metric | Model | Baseline |
|---|---|---|---|
| Risk | macro-F1 / accuracy | 0.589 / 0.742 | 0.126 (most-frequent) |
| Yield | MAE / R2 | 2.875 / 0.230 | 3.544 / 0.0 (mean) |
| Price | macro-F1 | 0.285 | 0.074 (most-frequent) |

Top risk drivers (permutation importance): pest incidents, NDVI, irrigation.

## Limitations
- Synthetic data cannot capture true agronomic causality; not field-calibrated.
- The price-direction head is a weak directional signal (small month coverage),
  deliberately reported rather than hidden.
- Province/crop/season vocabularies are fixed; genuinely novel categories map to
  an "unknown" code.

## Ethical considerations
No personal data. Aggregate-only. Advisory framing avoids over-reliance; the app
shows the driver explanation and a confidence caveat. Per-province performance parity is measured on the held-out months (see "Performance parity by province" in `reports/model_report.md`); large gaps trigger review and retraining before any pilot.