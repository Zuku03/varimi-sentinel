# AI Justification (C2) - why not a rule table?

The Track-3 ToR penalises "forcing" AI where a database query or static rule would
suffice (the "sledgehammer to crack a nut" rule). VaRimi Sentinel's task fails a
rule-based approach for concrete, measured reasons:

## 1. The target is a joint non-linear function of several signals
Climate-crop risk, yield and price direction depend on interactions between
rainfall, vegetation (NDVI), pest pressure, irrigation coverage and input
availability, conditioned on crop, province and season. A static threshold table
would need a hand-tuned rule for every crop x province x season cell and still
miss interactions.

## 2. Measured lift over a trivial baseline (held-out months)
| Head | Model | Trivial baseline | Lift |
|---|---|---|---|
| Risk band (macro-F1) | 0.589 | 0.126 | +0.463 |
| Yield (MAE, lower better) | 2.875 | 3.544 | -0.669 |
| Price direction (macro-F1) | 0.285 | 0.074 | +0.210 |

The risk head more than quadruples macro-F1 over most-frequent-class; the yield
head cuts mean absolute error and reaches positive R2 (0.230) where a mean
predictor is 0. This is signal a `SELECT ... WHERE` cannot reproduce.

## 3. Right-sized, not over-engineered
The method is gradient-boosted decision trees, not a deep network or LLM. It fits
in milliseconds on 480 rows, needs no GPU, and quantizes to a sub-megabyte ONNX
graph for edge inference - appropriate to the data size and the offline device
constraint. No "sledgehammer" in either direction: AI is justified by the lift,
and the model is as small as the job allows.

## 4. Honest scope
The price-direction head is a weak signal on this small synthetic sample and is
presented as directional only. We report it rather than drop it, and the product
surfaces it with an explicit caveat.
## 5. What the metrics do and do not prove (validity statement)

Dataset 02 is synthetic: its outcome columns were authored by a data generator,
so the relationships the model learns are the generator's, not nature's. The
lift table above therefore proves the pipeline is sound (leakage-controlled,
time-aware, reproducible) and that the method extracts multivariate structure a
rule table cannot - it does not prove field predictive validity, which no team
can honestly claim from this dataset. Milestones M1-M2 of the roadmap replace
the synthetic targets with observed district data and re-run this same
evaluation harness unchanged.