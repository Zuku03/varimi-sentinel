<!-- COVER -->
# VaRimi Sentinel
## Offline Edge-AI Crop-Climate Risk & Market Advisory for Zimbabwe

**Track:** Track 3 (Development)
**Team Name:** VaRimi Sentinel Team
**Lead Innovator:** [Lead Innovator Name]
**Date:** 6 July 2026

*Prototype built on the official AI4I synthetic dataset pack (dataset 02). All data is synthetic aggregate and not official statistics.*

<!-- PAGEBREAK -->

## Section 1 - Problem Definition & Strategic Alignment

### The local problem
Zimbabwe's smallholder farmers carry the country's food security but make planting, input and selling decisions with little timely, localized, quantitative guidance. Climate variability (erratic rainfall, pest outbreaks) and volatile farmgate prices routinely turn a promising season into a loss. Agricultural extension officers - the state's main advisory channel - are stretched across many wards and lack a fast, evidence-based way to tell **which district-crop combinations are at risk this month and what to do about it**. Rural connectivity is poor, so any solution that assumes a smartphone and a reliable data connection excludes the very users who need it most.

### Target users
1. **Agricultural extension officers** (primary) - need a per district-crop risk, yield and price read plus a concrete recommended action.
2. **Smallholder farmers** - reached through the officer or directly via **USSD/SMS on feature phones**, in **Shona, Ndebele or English**.
3. **Food-security planners** - need an aggregate view of risk hotspots to pre-position inputs.

### Alignment with the National AI Strategy 2026-2030
- **Sectoral adoption (agriculture):** agriculture is a named priority sector; VaRimi targets it directly with a food-security use-case.
- **Infrastructure sovereignty:** inference runs **fully offline on a low-end device** (no cloud dependency); training runs inside the **ZCHPC Controlled Compute Environment (CCE)**. Data stays in-country.
- **Ubuntu ethics & inclusion:** advisories are delivered in **local languages** and over **feature-phone channels**, extending benefit to rural, low-bandwidth communities rather than only smartphone owners.
- **Justified AI:** the tool is deployed only where AI adds measurable value (see Section 2), respecting the Strategy's caution against "sledgehammer" AI.

## Section 2 - Technical Design & Product Logic

### System architecture
```
Dataset 02 (CSV, synthetic)
  |  ETL + feature pipeline (season, ordinal-encoded categoricals)
  v
Gaussian-copula synthesizer --> correlation-validated synthetic rows
  |
  v
Multi-head model (HistGradientBoosting):
   risk-band classifier | yield regressor | price-direction classifier
  |  skl2onnx export (parity 1.000)
  v
ONNX heads (1.82 MB total) --> on-device inference (ONNX Runtime, CPU, offline)
  |                                   |
  |                                   +-- USSD/SMS advisory (officer-mediated)
  v                                   +-- Android app (bundled models)
ZCHPC CCE: training + model registry <-- periodic sync when online
```

### Models and product logic
For a given **(district, crop, month)** the product returns three predictions plus an explained action:
- **Risk band** (Low/Medium/High) - classification
- **Yield outlook** (t/ha) - regression
- **Farmgate price direction** (down/flat/up) - classification

The method is **gradient-boosted decision trees** (scikit-learn HistGradientBoosting). This is deliberately *right-sized*: the data is small, tabular and heterogeneous, so trees outperform deep nets, train in milliseconds, need no GPU, and export to a sub-megabyte ONNX graph for the edge. Five numeric signals (rainfall, NDVI proxy, pest incidents, irrigation coverage, input availability) and three categoricals (province, crop, season) feed all three heads. Raw targets and the continuous risk score are excluded from inputs to prevent leakage, and evaluation uses a **time-aware split** (hold out the latest months).

### Why AI, not a rule table (C2)
On the held-out months the model clearly beats a trivial baseline, proving it extracts multivariate signal a `SELECT ... WHERE` cannot reproduce:

| Head | Model | Trivial baseline | Lift |
|---|---|---|---|
| Risk band (macro-F1) | 0.623 | 0.126 | +0.497 |
| Yield (MAE t/ha, lower better) | 3.22 | 3.54 | -0.32 |
| Price direction (macro-F1) | 0.295 | 0.074 | +0.221 |

The risk head nearly **5x** the baseline macro-F1; yield reaches positive R2 where a mean predictor is zero. The price head is a weaker directional signal on this small synthetic sample and is surfaced with an explicit caveat rather than hidden.

### Edge feasibility (C4)
The three ONNX heads total **1.82 MB** (vs the 256 MB device budget) and a full three-head advisory runs at **~9 ms p95 latency** (vs the 100 ms budget), CPU-only, verified by an automated benchmark. Because the models are tree ensembles, int8 quantization (a neural-net technique) is unnecessary - the fp32 tree graph already fits the device by more than two orders of magnitude.

### Explainability
Each advisory shows its **drivers** (e.g. "pest incidents high, NDVI low") computed from the row's position in the feature distribution, backed by the model's global permutation importances (pest incidents, NDVI, irrigation are the top drivers). A small, transparent rule layer phrases the **recommended action** on top of the model's classification.

## Section 3 - Deliverables & CCE Implementation Roadmap

### Deliverables (in the repository)
- Modular, documented Python package (`src/varimi/*`) with a **passing pytest suite (20 tests)**, ruff-clean code, pinned `requirements.lock`, and CI.
- Reproducible **training + evaluation** pipeline (`python -m varimi.model.train`) emitting `reports/model_report.md`.
- **Synthetic data generator + validation** (`reports/synthetic_validation.md`: all 7 KS marginal tests pass).
- **ONNX edge export with parity check** and **edge benchmark** (`reports/edge_benchmark.md`).
- **Advisory service, USSD flow, and Streamlit officer demo** (the demo URL).

### CCE implementation roadmap
- **Compute:** training is CPU-only and completes in seconds; it runs inside the **ZCHPC CCE** with a **dependency-locked manifest** (`requirements.lock`) for exact reproducibility, packaged as a container for sandbox validation runs.
- **Milestones (Grand Challenge window):**
  - *M1* - data contract + feature pipeline hardened; dataset statement signed off.
  - *M2* - model + synthetic augmentation validated against real held-out months.
  - *M3* - ONNX edge export + Android/USSD delivery integration; benchmark on target low-end hardware.
  - *M4* - controlled field pilot preparation with an extension-services partner.
- **Dependencies:** ZCHPC CCE access; a district extension-services partner for pilot data and validation; a USSD short-code from a local aggregator.

## Section 4 - Compliance & Risk Mitigation

- **Data Protection Act [Chapter 12:07]:** the model uses only **aggregate, non-personal** data (no names, IDs, phone numbers or addresses). Any interaction logging in the app is opt-in behind a **data-use consent checkbox**, with data minimization, local-first storage and a defined retention window.
- **Model assurance:** unit tests cover the data contract, feature engineering, synthesizer, model contract and ONNX parity; CI runs them on every change.
- **Cybersecurity:** signed/verified model artifacts, input validation on the encoder, a **rate-limited sync API**, and no PII embedded in models. USSD sessions are stateless and carry no personal identifiers.
- **Key risks & mitigations:**
  - *Synthetic data is not field-calibrated* -> pilot calibration against real district data before any live advisory; product framed as decision-support, not ground truth.
  - *Weak price-direction signal* -> presented as directional only, with a caveat; not used for irreversible decisions.
  - *Fairness across regions* -> monitor risk/yield performance parity across provinces and settlement types; retrain if disparity emerges.

## Section 5 - Sustainability & Future Adoption

- **Operating cost:** minimal - CPU-only training, a light periodic-sync API, and negligible per-device inference cost. No GPU or cloud-inference bill.
- **Licensing register:** all components are permissively licensed (BSD/MIT/Apache) and disclosed in `ASSET_LICENCE_REGISTER.md`; models are the team's own work (MIT).
- **Adoption & revenue pathway:** licence the advisory to **extension services** (Ministry of Lands, Agriculture, Fisheries, Water and Rural Development), **agro-dealers** and **farmer unions**; a freemium USSD tier keeps farmer access free while institutions fund the service.
- **Scaling:** the pipeline generalizes across every district x crop with no schema change, and the same architecture extends to other priority sectors (e.g. livestock, horticulture) as data becomes available.

---
*Appendix material (code, full reports, model card, dataset statement) accompanies this proposal in the project repository and does not count toward the page limit.*