<!-- COVER -->
# VaRimi Sentinel
## Offline Edge-AI Crop-Climate Risk & Market Advisory for Zimbabwe

**Track:** Track 3 (Development)
**Team Name:** VaRimi Sentinel Team
**Lead Innovator:** Chrispen Nyanhete
**Date:** 7 July 2026

**Repository:** https://github.com/Zuku03/varimi-sentinel &nbsp;|&nbsp; **Live demo:** https://varimisentinel.streamlit.app

*Prototype built on the official AI4I synthetic dataset pack (dataset 02). All data is synthetic aggregate and not official statistics.*

<!-- PAGEBREAK -->

## Section 1 - Problem Definition & Strategic Alignment

### The local problem
Zimbabwe's smallholder farmers carry the country's food security but make planting, input and selling decisions with little timely, localized, quantitative guidance. Climate variability (erratic rainfall, pest outbreaks) and volatile farmgate prices routinely turn a promising season into a loss. Agricultural extension officers - the state's main advisory channel - are stretched across many wards and lack a fast, evidence-based way to tell **which district-crop combinations are at risk this month and what to do about it**. Rural connectivity is poor, so any solution that assumes a smartphone and a reliable data connection excludes the very users who need it most.

### Target users
1. **Agricultural extension officers** (primary) - need a per district-crop risk, yield and price read plus a concrete recommended action they can defend to a farmer.
2. **Smallholder farmers** - reached through the officer or directly via **USSD/SMS on feature phones**, in **Shona, Ndebele or English**.
3. **Food-security planners** - need an aggregate view of risk hotspots to pre-position inputs and target irrigation support.

### User evidence: a day in the workflow
The prototype was designed around the extension-officer workflow implied by the challenge dataset pack (which names the *agricultural extension officer, farmer association and food security planner* as the intended personas for dataset 02):

1. **Monday planning:** the officer opens the cockpit (or dials the USSD code where there is no data signal), selects her district and the two dominant crops, and reads the month's risk bands with their drivers ("high risk - pest incidents high, NDVI low").
2. **Field visits:** she walks affected wards with a concrete, localized action ("ongororai minda uye rapai zvipembenene..."), not a generic leaflet.
3. **Reporting up:** the food-security planner sees the same hotspots aggregated by province and pre-positions inputs before the situation worsens.
4. **Farmer self-service:** a farmer with a feature phone repeats the same query by USSD in her own language and receives the same advisory the officer would give.

### Stakeholder engagement plan
We have designed the pilot around **AGRITEX** (the Department of Agricultural, Technical and Extension Services) as the natural institutional home for the advisory, with the district agricultural extension office as the pilot unit. Engagement is **planned, not yet secured**: our first milestone after selection (M1) includes a structured briefing to AGRITEX district officers and the collection of officer feedback on the advisory wording, thresholds and workflow fit. We deliberately make no claim of existing endorsement.

### Alignment with the National AI Strategy 2026-2030
- **Sectoral adoption (agriculture):** agriculture is a named priority sector; VaRimi targets it directly with a food-security use-case.
- **Infrastructure sovereignty:** inference runs **fully offline on a low-end device** (no cloud dependency); training runs inside the **ZCHPC Controlled Compute Environment (CCE)**. Data stays in-country.
- **Ubuntu ethics & inclusion:** advisories are fully catalogued in **Shona and Ndebele** (drafts pending native-speaker sign-off, with a formal validation sheet in the repository) and delivered over **feature-phone channels**, extending benefit to rural, low-bandwidth communities rather than only smartphone owners.
- **Justified AI:** the tool uses AI only where it adds measured value over a rule baseline (Section 2), respecting the Strategy's caution against "sledgehammer" AI.


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
  |  skl2onnx export (prediction parity 1.000, verified in tests)
  v
ONNX heads (1.82 MB total) --> on-device inference (ONNX Runtime, CPU, offline)
  |                                   |
  |                                   +-- USSD/SMS advisory (officer-mediated)
  |                                   +-- Android harness (in repo; device
  |                                       validation at milestone M3)
  v
ZCHPC CCE: training + model registry <-- periodic sync when online
```

### Models and product logic
For a given **(district, crop, month)** the product returns three predictions plus an explained action:
- **Risk band** (Low/Medium/High) - classification
- **Yield outlook** (t/ha) - regression
- **Farmgate price direction** (down/flat/up) - classification

The method is **gradient-boosted decision trees** (scikit-learn HistGradientBoosting). This is deliberately *right-sized*: the data is small, tabular and heterogeneous, so trees outperform deep nets, train in milliseconds, need no GPU, and export to a sub-megabyte ONNX graph for the edge. Five numeric signals (rainfall, NDVI proxy, pest incidents, irrigation coverage, input availability) and three categoricals (province, crop, season) feed all three heads. Raw targets and the continuous risk score are excluded from inputs to prevent leakage, and evaluation uses a **time-aware split** (the model is only ever tested on months later than any it trained on).

### Why AI, not a rule table (C2)
On the held-out months the model clearly beats a trivial baseline, demonstrating it extracts multivariate signal a `SELECT ... WHERE` cannot reproduce:

| Head | Model | Trivial baseline | Lift |
|---|---|---|---|
| Risk band (macro-F1) | 0.623 | 0.126 | +0.497 |
| Yield (MAE t/ha, lower better) | 3.22 | 3.54 | -0.32 |
| Price direction (macro-F1) | 0.295 | 0.074 | +0.221 |

The risk head is ~5x the baseline macro-F1; yield reaches positive R2 (0.230) where a mean predictor scores zero. The price head is a weaker directional signal on this small sample and is surfaced with an explicit caveat rather than hidden.

### Validity of the evidence - what these metrics do and do not prove
We state this plainly because it is the right question to ask of any model trained on challenge data: **dataset 02 is synthetic**, so its outcome columns were authored by a data generator, and the statistical relationships our model learns are the generator's, not nature's. The metrics above therefore prove three things - that the **pipeline is sound** (leakage-controlled features, honest time-aware evaluation, reproducible training), that the **method extracts real multivariate structure** where a rule table cannot, and that the **product mechanism works end-to-end** from CSV to localized advisory on a device. They do **not** prove field predictive validity. That is a property no team can honestly claim from this dataset. Our roadmap treats it as the first real deliverable: milestones M1-M2 replace the synthetic targets with observed district data (AGRITEX crop assessments, AMA price series, rainfall stations) and re-run the *same* evaluation harness, which was built so that swapping the data source requires no methodological change.

### Edge feasibility (C4)
The three ONNX heads total **1.82 MB** on disk against the 256 MB device budget, and a full three-head advisory completes in **well under 10 ms p95** (sub-millisecond on an unloaded machine; two orders of magnitude inside the 100 ms budget), CPU-only, measured by an automated benchmark committed to the repository. Because the models are tree ensembles, int8 quantization (a neural-network technique) is unnecessary - the fp32 tree graph already fits the device with two orders of magnitude to spare. These numbers are from x86 CPU; a **buildable Android harness ships in the repository** (ONNX Runtime Mobile, bundled models, per-advisory latency logging) so the true hardware-in-the-loop number is recorded on a low-end device at milestone M3.

### Explainability and localization
Each advisory shows its **drivers** (e.g. "pest incidents high, NDVI low") computed from the row's position in the feature distribution, backed by the model's global permutation importances (pest incidents, NDVI and irrigation are the top risk drivers). A small, transparent rule layer phrases the **recommended action** on top of the model's classification - the classification is always the model's. Every user-facing string is catalogued in English, Shona and Ndebele; the local-language drafts ship with a formal **native-speaker validation sheet** (`docs/TRANSLATION_VALIDATION.md`) and are marked as pending sign-off until validated. Automated tests forbid English fragments in Shona/Ndebele output.


## Section 3 - Deliverables & CCE Implementation Roadmap

### Deliverables (in the repository today)
- Modular, documented Python package (`src/varimi/*`) with a **passing test suite (23 tests)**, ruff-clean code, pinned `requirements.lock`, and CI.
- Reproducible **training + evaluation** pipeline (`python -m varimi.model.train`) emitting `reports/model_report.md`, including a **per-province performance-parity (fairness) table**.
- **Synthetic data generator + validation** (`reports/synthetic_validation.md`: all 7 KS marginal tests pass; correlation-delta heatmap).
- **ONNX edge export with a parity test** and an **automated edge benchmark** (`reports/edge_benchmark.md`).
- **Advisory service (en/sn/nd), USSD flow, Streamlit officer cockpit** (https://varimisentinel.streamlit.app), and a **buildable Android device harness** with an asset-sync script.
- Governance documents: dataset statement, model card, AI-justification note, translation validation sheet, asset & licence register.

### Milestone roadmap (Grand Challenge window)

| Milestone | Target date | Work | Exit evidence |
|---|---|---|---|
| M1 - Data & institutional grounding | Aug 2026 | AGRITEX district briefing; officer feedback on advisory wording/thresholds; begin real-data acquisition (AGRITEX assessments, AMA prices, rainfall stations); native-speaker translation sign-off | Engagement minutes; signed translation sheet; data-access memo |
| M2 - Field calibration | Sep 2026 | Retrain on observed district data with the same harness; re-run baseline, fairness and TSTR evaluations; calibrate thresholds | Updated model report on real data; go/no-go on price head |
| M3 - Device & channel validation | Oct 2026 | Build Android harness on target low-end device; record hardware-in-the-loop latency; integrate USSD short-code with a local aggregator (sandbox) | Device benchmark; USSD sandbox transcript |
| M4 - Pilot readiness | Nov 2026 | Controlled pilot in one district with AGRITEX officers; feedback loop; incident/rollback procedure | Pilot protocol; officer feedback report |

### CCE implementation plan
Training is CPU-only and completes in seconds, so our ZCHPC CCE footprint is deliberately small: a containerised training job pinned by `requirements.lock` for bit-reproducible sandbox validation runs, a model registry entry per release, and scheduled retraining as new monthly data lands. No GPU allocation is requested.

### Dependencies
ZCHPC CCE access; AGRITEX district partnership (M1); AMA/market price data access (M2); USSD short-code via a local aggregator (M3).

### Team & delivery capability
| Member | Role | Relevant credentials |
|---|---|---|
| Chrispen Nyanhete | Lead Innovator - product, ML engineering, on-device validation | BSc Information Technology; MSc Big Data Technologies |
| Tendai Nyebera | Co-innovator - engineering & delivery | BSc Electronic Engineering |

Specialist capacity the core team does not carry in-house - agronomy/extension
liaison and native-language validation - is deliberately budgeted as contracted
expertise in Section 5 rather than claimed as team skills.

The repository itself is offered as the primary evidence of delivery capability: locked dependencies, CI, 23 automated tests, parity-checked model export and honest reporting of weak results demonstrate the engineering discipline the team will apply through M1-M4.


## Section 4 - Compliance & Risk Mitigation

### Data Protection Act [Chapter 12:07]
The model uses only **aggregate, non-personal** data (no names, IDs, phone numbers or addresses). Any interaction logging in the app or USSD flow is opt-in behind a **data-use consent checkbox**, with data minimization, local-first storage and a defined retention window. USSD sessions are stateless and carry no personal identifiers.

### Model assurance & fairness
Unit tests cover the data contract, feature engineering, synthesizer, model contract, localization completeness and ONNX parity; CI runs them on every change. **Fairness is measured, not asserted:** the evaluation emits a per-province performance-parity table (risk macro-F1 and yield MAE by province) on every training run, and large gaps trigger review and retraining before any pilot exposure.

### Cybersecurity
Signed and hash-verified model artifacts; input validation at the encoder boundary; a **rate-limited sync API**; no PII embedded in models or logs; dependency manifests pinned and auditable.

### Risk register

| Risk | Type | Likelihood | Mitigation |
|---|---|---|---|
| Synthetic training data is not field-calibrated | Data | Certain (by design) | M1-M2 recalibration on observed data using the same harness; advisory framed as decision-support with visible caveats until then |
| Price-direction head remains weak on real data | Technical | Medium | Reported honestly with caveat; M2 go/no-go gate - drop or replace the head if it fails on real prices |
| Machine-drafted local-language strings mistranslate | Ethical | Medium | Formal native-speaker validation sheet; strings blocked from live use until signed off; tests forbid mixed-language output |
| Officers distrust or ignore the advisory | Adoption | Medium | Driver explanations with every prediction; officer feedback loop at M1; advisory never contradicts officer judgement, it informs it |
| AGRITEX partnership slips | Institutional | Medium | Pilot can proceed with a farmer association or NGO extension partner; engagement started at M1, not at pilot time |
| Regional performance disparity emerges | Ethical | Low-Medium | Per-province parity table on every run; retrain trigger defined before pilot |
| Device fragmentation breaks the Android build | Technical | Low | Harness targets minSdk 24 with plain views; USSD channel is the fallback path for any device the app cannot serve |


## Section 5 - Sustainability & Future Adoption


### Grant allocation (USD 80,000)

| Line | Amount | Notes |
|---|---|---|
| Personnel (2 core members, part-time, 12 months) | 38,000 | ML, mobile and delivery; includes contracted agronomy/extension specialist time |
| Field calibration pilot (2 districts) | 14,000 | Travel, enumerators, officer stipends, data collection |
| USSD short-code + aggregator fees (12 months) | 6,000 | Setup + session fees |
| Training & documentation for extension officers | 5,000 | Workshops, printed quick-reference cards |
| Device procurement (pilot) | 4,000 | ~20 low-end Android + feature phones |
| Translation validation & accessibility review | 3,000 | Native-speaker linguists, accessibility audit |
| Hosting & CCE compute | 3,000 | Sync API VPS, registry, CI |
| Contingency (~9%) | 7,000 | Price/exchange-rate movement, re-work |
| **Total** | **80,000** | |

### Steady-state operating cost (post-grant, per year)

| Item | Est. cost / yr | Basis |
|---|---|---|
| VPS hosting (sync API + demo) | 360 | ~$30/month |
| USSD aggregator fees | 1,800 | ~$0.03/session x ~5,000 sessions/month |
| Maintenance & monthly retraining (0.2 FTE) | 4,800 | Includes parity-table review |
| **Total** | **~7,000** | |

### Revenue and adoption pathway
- **Institutional licences:** the advisory is licensed to institutions while farmer USSD access stays free. Indicative tiers: district extension office **$3,000/yr**, provincial planning unit **$5,000/yr**, agro-industry (input suppliers, contract farming schemes) **$5,000-10,000/yr** for the aggregated hotspot feed.
- **3-year projection:** Year 1 - grant-funded pilot, first AGRITEX licence negotiated. Year 2 - 3 licences (~$12,000) against ~$7,000 operating cost: break-even. Year 3 - 6 licences plus one agro-industry feed (~$24,000+): surplus funds new crops and provinces.
- **Scaling:** the pipeline generalizes across every district x crop with no schema change; the same architecture extends to livestock and horticulture as data becomes available, and every artifact (training, validation, benchmark, localization catalog) is already built to be re-run rather than re-built.

---
*Appendix material (code, full reports, model card, dataset statement, translation validation sheet) accompanies this proposal in the project repository and does not count toward the page limit.*