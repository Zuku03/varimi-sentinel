# Dataset Statement

## Source & provenance
- **File:** `02_agriculture_climate_market_signals.csv`
- **Origin:** POTRAZ AI4I Grand Challenge official Design-Track dataset pack.
- **Status:** **Synthetic aggregate** sample data created for challenge use.
  **Not official statistics.** Must not be used for operational policy reporting
  outside the challenge environment.
- **Rows:** 360. **Grain:** one row per (month, province, district, crop).
- **Privacy:** contains no names, national IDs, phone numbers, addresses, or
  individual-level records. All values are aggregate and synthetic.

## Fields used
| Role | Field | Type | Notes |
|------|-------|------|-------|
| Key | `month` | YYYY-MM | Reporting month (e.g. 2026-03) |
| Key | `province`, `district` | text | Zimbabwe admin areas |
| Feature | `rainfall_mm` | decimal | Monthly rainfall proxy |
| Feature | `ndvi_proxy_0_1` | decimal | Vegetation index proxy (0-1) |
| Feature | `pest_incidents_reported` | integer | Reported pest issues |
| Feature | `irrigation_coverage_pct` | percent | Share of crop area irrigated |
| Feature | `input_availability_score_0_100` | decimal | Seed/fertilizer/advisory access |
| Feature (engineered) | `season` | category | wet / cool_dry / hot_dry, from `month` |
| Target | `risk_level` | category | Low / Medium / High (band of the risk score) |
| Target | `estimated_yield_t_per_ha` | decimal | Yield proxy (t/ha) |
| Target (engineered) | `price_direction` | category | down/flat/up, MoM change of farmgate price |

## Leakage controls
The following columns are **excluded from model inputs** because they are raw
targets or direct derivations of a target (using them would leak the answer):
`climate_crop_risk_score_0_100` (the continuous score that `risk_level` bands),
`risk_level`, `estimated_yield_t_per_ha`, `avg_farmgate_price_usd_per_tonne`.
`latitude`/`longitude` are retained only for map display, not as model features.
Evaluation uses a **time-aware split** (hold out the latest reporting months) so
the model is never tested on a month it trained on.

## Synthetic augmentation
Because the sample is small (360 rows) and already synthetic, training data is
expanded with a **fully-disclosed Gaussian-copula synthesizer** (see
`src/varimi/synth/` and `reports/synthetic_validation.md`). Synthetic rows are
validated against the real sample with per-feature KS tests and a
correlation-matrix delta, and a train-on-synthetic / test-on-real check.

## Limitations
- Synthetic data cannot capture true agronomic causality; the models demonstrate
  the product mechanism and are **not** field-calibrated for live advisories.
- Small month coverage limits the reliability of the price-direction target;
  treated as a directional signal, not a point forecast.
