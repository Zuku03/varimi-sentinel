"""Train the VaRimi models, benchmark against baselines, and emit C2/C3 evidence.

Produces:
  * models/varimi_model.joblib      - the fitted bundle (encoder + 3 heads)
  * models/metrics.json             - all headline metrics
  * reports/model_report.md         - baseline comparison, TSTR lift, importances

Run: python -m varimi.model.train
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

from varimi import config
from varimi.data import loader
from varimi.features import build
from varimi.model.pipeline import FEATURE_ORDER, VarimiModel, _new_classifier
from varimi.synth import augment

TEST_MONTHS = 2
SYNTH_RATIO = 1.0


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def _prepare():
    enriched = build.build(loader.load_raw())
    train, test = build.time_aware_split(enriched, test_months=TEST_MONTHS)
    risk_yield_train = augment.augment_training_frame(
        train, ratio=SYNTH_RATIO, seed=config.RANDOM_SEED
    )
    return enriched, train, test, risk_yield_train


def _risk_f1_for(frame: pd.DataFrame, encoder_vocab: pd.DataFrame, test: pd.DataFrame) -> float:
    """Fit a standalone risk classifier on ``frame`` and score on real ``test``."""
    m = VarimiModel()
    m.fit_encoder(encoder_vocab)
    clf = _new_classifier(config.RANDOM_SEED).fit(m.encode(frame), frame[config.TARGET_RISK])
    return _macro_f1(test[config.TARGET_RISK], clf.predict(m.encode(test)))


def run() -> dict:
    import joblib

    enriched, train, test, risk_yield_train = _prepare()

    model = VarimiModel()
    model.fit_encoder(enriched)  # fixed a-priori category vocabulary
    model.fit(risk_yield_train, price_df=train)
    preds = model.predict(test)

    # --- Model metrics on the held-out real months ---
    risk_f1 = _macro_f1(test[config.TARGET_RISK], preds[config.TARGET_RISK])
    risk_acc = float(accuracy_score(test[config.TARGET_RISK], preds[config.TARGET_RISK]))
    yield_mae = float(mean_absolute_error(test[config.TARGET_YIELD], preds[config.TARGET_YIELD]))
    yield_r2 = float(r2_score(test[config.TARGET_YIELD], preds[config.TARGET_YIELD]))
    price_f1 = _macro_f1(test[config.TARGET_PRICE_DIR], preds[config.TARGET_PRICE_DIR])

    # --- Baselines (the C2 "prove AI beats a trivial baseline" evidence) ---
    dummy_risk = DummyClassifier(strategy="most_frequent").fit(
        train[FEATURE_ORDER], train[config.TARGET_RISK]
    )
    base_risk_f1 = _macro_f1(test[config.TARGET_RISK], dummy_risk.predict(test[FEATURE_ORDER]))
    dummy_yield = DummyRegressor(strategy="mean").fit(
        train[FEATURE_ORDER], train[config.TARGET_YIELD]
    )
    base_yield_mae = float(
        mean_absolute_error(test[config.TARGET_YIELD], dummy_yield.predict(test[FEATURE_ORDER]))
    )
    dummy_price = DummyClassifier(strategy="most_frequent").fit(
        train[FEATURE_ORDER], train[config.TARGET_PRICE_DIR]
    )
    base_price_f1 = _macro_f1(
        test[config.TARGET_PRICE_DIR], dummy_price.predict(test[FEATURE_ORDER])
    )

    # --- TSTR: does synthetic augmentation help / not hurt? ---
    synth_only = augment.make_synthetic(train, n=len(train), seed=config.RANDOM_SEED)
    tstr = {
        "real_only": _risk_f1_for(train, enriched, test),
        "synth_only": _risk_f1_for(synth_only, enriched, test),
        "real_plus_synth": risk_f1,
    }

    # --- Driver explanations (global permutation importance for the risk head) ---
    perm = permutation_importance(
        model.risk_clf,
        model.encode(test),
        test[config.TARGET_RISK],
        n_repeats=10,
        random_state=config.RANDOM_SEED,
        scoring="f1_macro",
    )
    importances = sorted(
        zip(FEATURE_ORDER, perm.importances_mean.tolist(), strict=False),
        key=lambda t: t[1],
        reverse=True,
    )

    metrics = {
        "n_train": int(len(train)),
        "n_train_augmented": int(len(risk_yield_train)),
        "n_test": int(len(test)),
        "test_months": sorted(test["month"].unique().tolist()),
        "risk": {
            "model_macro_f1": risk_f1,
            "model_accuracy": risk_acc,
            "baseline_macro_f1": base_risk_f1,
        },
        "yield": {
            "model_mae": yield_mae,
            "model_r2": yield_r2,
            "baseline_mae": base_yield_mae,
        },
        "price": {"model_macro_f1": price_f1, "baseline_macro_f1": base_price_f1},
        "tstr_risk_macro_f1": tstr,
        "risk_permutation_importance": importances,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "varimi_model.joblib")
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(metrics)
    return metrics


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _write_report(m: dict) -> None:
    r = m["risk"]
    y = m["yield"]
    p = m["price"]
    t = m["tstr_risk_macro_f1"]

    risk_delta = f"{r['model_macro_f1'] - r['baseline_macro_f1']:+.3f}"
    yield_delta = f"{y['model_mae'] - y['baseline_mae']:+.3f}"
    price_delta = f"{p['model_macro_f1'] - p['baseline_macro_f1']:+.3f}"

    lines = [
        "# Model Report - VaRimi Sentinel",
        "",
        f"Held-out real test months: {', '.join(m['test_months'])} "
        f"(n_test={m['n_test']}). Train rows: {m['n_train']} real, "
        f"{m['n_train_augmented']} after synthetic augmentation.",
        "",
        "## 1. Model vs trivial baseline (C2 evidence)",
        "",
        _row(["Head", "Metric", "Model", "Baseline", "Delta"]),
        "|---|---|---|---|---|",
        _row(["Risk band", "macro-F1", f"{r['model_macro_f1']:.3f}",
              f"{r['baseline_macro_f1']:.3f}", risk_delta]),
        _row(["Risk band", "accuracy", f"{r['model_accuracy']:.3f}", "-", "-"]),
        _row(["Yield (t/ha)", "MAE (lower=better)", f"{y['model_mae']:.3f}",
              f"{y['baseline_mae']:.3f}", yield_delta]),
        _row(["Yield (t/ha)", "R2", f"{y['model_r2']:.3f}", "0.000",
              f"{y['model_r2']:+.3f}"]),
        _row(["Price direction", "macro-F1", f"{p['model_macro_f1']:.3f}",
              f"{p['baseline_macro_f1']:.3f}", price_delta]),
        "",
        "The baselines are a most-frequent-class classifier and a mean regressor.",
        "A positive macro-F1 delta and a lower yield MAE show the model extracts",
        "genuine multivariate signal a rule/SQL lookup cannot - the C2 justification",
        "for using AI rather than a static rule table.",
        "",
        "## 2. Synthetic augmentation - train-on-synthetic / test-on-real (C3)",
        "",
        "Risk-band macro-F1 on the real held-out months by training source:",
        "",
        _row(["Training data", "Risk macro-F1"]),
        "|---|---|",
        _row(["Real only", f"{t['real_only']:.3f}"]),
        _row(["Synthetic only", f"{t['synth_only']:.3f}"]),
        _row(["Real + synthetic", f"{t['real_plus_synth']:.3f}"]),
        "",
        "Synthetic-only close to real-only validates the generator; real+synthetic",
        "not underperforming real-only shows augmentation is safe.",
        "",
        "## 3. Risk-head driver importance (permutation, macro-F1 drop)",
        "",
        _row(["Feature", "Importance"]),
        "|---|---|",
    ]
    for feat, imp in m["risk_permutation_importance"]:
        lines.append(_row([feat, f"{imp:.4f}"]))
    lines += [
        "",
        "These global drivers back the plain-language explanation the advisory",
        "shows to extension officers (e.g. 'high risk driven by low NDVI and",
        "rising pest incidents').",
    ]
    (config.REPORTS_DIR / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print("Risk macro-F1:", round(result["risk"]["model_macro_f1"], 3),
          "(baseline", round(result["risk"]["baseline_macro_f1"], 3), ")")