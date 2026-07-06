"""CLI entrypoint to train models and write C2/C3 reports.

Run: python -m varimi.model.train
"""

from __future__ import annotations

from varimi.model.evaluate import run

if __name__ == "__main__":
    m = run()
    print("Risk macro-F1:", round(m["risk"]["model_macro_f1"], 3),
          "vs baseline", round(m["risk"]["baseline_macro_f1"], 3))
    print("Yield MAE:", round(m["yield"]["model_mae"], 3),
          "vs baseline", round(m["yield"]["baseline_mae"], 3))
    print("Price macro-F1:", round(m["price"]["model_macro_f1"], 3),
          "vs baseline", round(m["price"]["baseline_macro_f1"], 3))