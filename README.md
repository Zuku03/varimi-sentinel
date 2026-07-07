# VaRimi Sentinel

Offline-first **edge-AI crop-climate risk & market advisory** for Zimbabwe.
POTRAZ AI4I Grand Challenge — **Track 3 (Development)**.

For a given **district + crop + month**, VaRimi Sentinel predicts:

1. **Climate-crop risk band** (Low / Medium / High) with driver explanation
2. **Yield outlook** (tonnes per hectare)
3. **Farmgate price direction** (down / flat / up)

…and renders a plain-language recommended action for an agricultural extension
officer, delivered on a low-end Android device or via USSD/SMS in Shona, Ndebele
or English.

## Why AI (not a rule table)
Risk, yield and price are joint non-linear functions of rainfall, vegetation
(NDVI), pest pressure, irrigation coverage and input availability, conditioned on
season and agro-ecology. A static rule/SQL table cannot generalise across unseen
district × crop × month combinations. See `docs/AI_JUSTIFICATION.md` (Phase 3).

## Data
Trained on `Datasets/02_agriculture_climate_market_signals.csv` — a **synthetic
aggregate** sample from the official AI4I Design-Track pack (not official
statistics). Augmented with a fully-disclosed Gaussian-copula synthesizer,
validated with statistical correlation tests. See `docs/DATASET_STATEMENT.md`.

## Layout
```
src/varimi/{data,features,synth,model,edge,serving}   # library code
tests/                                                # pytest suite
reports/                                              # validation + benchmark reports
docs/                                                 # dataset statement, model card, AI justification
proposal/                                             # 10-page Track-3 proposal (Phase 5)
```

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,demo]"
pytest
```

## Status
Prototype / MVP under active build for the AI4I Grand Challenge. Synthetic data
only; not for operational policy decisions.

## Reproduce end-to-end
```bash
python -m varimi.model.train        # train + reports/model_report.md + metrics
python -m varimi.synth.validate     # reports/synthetic_validation.md (C3)
python -m varimi.edge.export        # models/*.onnx (parity-checked)
python -m varimi.edge.benchmark     # reports/edge_benchmark.md (C4 budgets)
streamlit run src/varimi/serving/app.py   # officer demo (needs .[demo])
python scripts/build_proposal.py    # proposal/*_AI4I_Proposal_Development.pdf (needs .[proposal])
```
## Android device harness
`android/` contains a minimal, buildable Kotlin harness (ONNX Runtime Mobile,
bundled models, Logcat latency logging) for hardware-in-the-loop validation on
a low-end device. Status: scaffold, not compile-verified here — see
`android/README.md` for build steps and the honest caveat.