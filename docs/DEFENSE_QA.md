# Defense Q&A - anticipated adjudicator questions

Rehearsal notes for the team. Each answer is grounded in a repository artifact;
never claim more than the artifact shows.

## Q1. "Your data is synthetic - isn't your model just reverse-engineering the generator's formula?"
**Yes, partly - and we say so in the proposal (Section 2, "Validity of the evidence").**
The synthetic targets were authored by a generator, so our metrics prove pipeline
soundness (leakage controls, time-aware evaluation, reproducibility) and that the
method extracts multivariate structure a rule table cannot - not field validity.
That is why M1-M2 of our roadmap swap in observed district data (AGRITEX
assessments, AMA prices, rainfall stations) and re-run the *same* harness. The
honest framing is our strength: no team can claim field validity from this
dataset; we are the team that engineered for the moment real data arrives.

## Q2. "Why gradient-boosted trees and not a deep model or an LLM?"
Right-sizing. The data is small (360 rows), tabular and heterogeneous - the
regime where boosted trees are state of the art. Trees train in seconds on CPU
(fits the ZCHPC CCE without GPU), export to a 1.8 MB ONNX graph (fits a $50
phone), and their drivers are explainable to an extension officer. A deep net or
LLM here would be exactly the "sledgehammer" the ToR penalizes.

## Q3. "Explain the Gaussian copula in one minute."
Rank each continuous column, map ranks to a normal distribution, estimate the
correlation matrix of those normal scores, sample new normal vectors with the
same correlations, then map each column back through the original data's own
quantiles. Result: synthetic rows whose marginal distributions match the real
data *exactly* and whose correlations match closely. Evidence: all 7 KS tests
pass (p > 0.05), max correlation delta 0.17 (`reports/synthetic_validation.md`).

## Q4. "What does your TSTR table actually show?"
Train-on-synthetic, test-on-real. A risk classifier trained *only* on our
synthetic rows scores 0.52 macro-F1 on real held-out months - close to the
real-only 0.41 - which validates the generator captured usable structure.
Real+synthetic (0.59) beats real-only (0.41), so augmentation helps rather than
harms. The gain is a regularization/smoothing effect on a small sample, and we
describe it that way.

## Q5. "Your price-direction head is weak (0.295 macro-F1). Why ship it?"
Because hiding weak results is worse than reporting them. It still beats the
trivial baseline (0.074), it is labelled a directional signal with a caveat in
the product, and M2 has an explicit go/no-go gate: if it stays weak on real
price series, we drop or replace it. Judges should read this as evidence of how
we will report results to them during the programme.

## Q6. "All your commits are from two days. Did you really build this?"
Yes - the history is honest and we can walk through any file. The compressed
timeline reflects an intensive prototyping sprint on top of significant prior
design work (rubric analysis, dataset assessment, architecture selection). Ask
us anything: the copula math, the leakage list in `config.py`, why ordinal
encoding replaced native categorical splits (ONNX parity), the protobuf shim.

## Q7. "How do you meet the edge budgets - and on what hardware?"
Measured, not estimated: 1.82 MB total model footprint vs the 256 MB budget,
sub-10 ms p95 advisory latency vs 100 ms - from an automated benchmark in the
repo (`python -m varimi.edge.benchmark`). Caveat we volunteer: that is x86 CPU.
The repo ships a buildable Android harness (ONNX Runtime Mobile) that logs
per-advisory latency, and M3 records the true low-end-device number. Tree
models are microsecond-scale; the budget risk is negligible.

## Q8. "What about the Data Protection Act?"
The training data is aggregate and synthetic - no personal data exists anywhere
in the system today. When interaction logging arrives (pilot), it is opt-in
behind a consent checkbox, minimized, stored locally first, with a defined
retention window. USSD sessions are stateless. See proposal Section 4.

## Q9. "How do you know the model is fair across regions?"
We measure it every training run: `reports/model_report.md` section 4 breaks
risk macro-F1 and yield MAE down by province, and a documented trigger requires
review/retraining if gaps widen before any pilot exposure. Per-province samples
are small, so we treat the table as a tripwire, not a certificate - and say so.

## Q10. "Your Shona/Ndebele - who validated it?"
Nobody yet, and the repo says so: every string is catalogued in
`docs/TRANSLATION_VALIDATION.md` with a native-speaker sign-off sheet, the
module docstring marks the drafts as pending validation, and budget line
"Translation validation" ($3,000) funds professional review at M1. Automated
tests already forbid mixed-language output.

## Q11. "Who is your institutional partner?"
Planned, not secured - the proposal says exactly that. AGRITEX is the designed
pilot home; M1 is a structured district briefing plus officer feedback, and the
risk register carries a fallback (farmer association / NGO extension partner)
if the AGRITEX timeline slips. We chose honesty over a name-drop.

## Q12. "What happens to this after the grant?"
Steady-state cost is ~$7k/yr (hosting, USSD fees, 0.2 FTE maintenance) against
institutional licences at $3-5k each - break-even at ~3 licences in Year 2,
surplus by Year 3 funding new crops/provinces. Farmer USSD access stays free.
Numbers are drafted estimates flagged for validation with real quotes.