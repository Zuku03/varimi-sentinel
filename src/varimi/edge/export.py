"""Export the VaRimi heads to ONNX for edge inference, with parity validation.

Run: python -m varimi.edge.export
Requires a trained bundle at models/varimi_model.joblib (python -m varimi.model.train).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from varimi import config
from varimi.data import loader
from varimi.edge import compat
from varimi.features import build
from varimi.model.pipeline import FEATURE_ORDER

OPSET = 15


def _convert(estimator, n_features: int, is_classifier: bool):
    compat.apply()
    init = [("input", FloatTensorType([None, n_features]))]
    options = {id(estimator): {"zipmap": False}} if is_classifier else None
    return to_onnx(estimator, initial_types=init, target_opset=OPSET, options=options)


def build_onnx(model) -> dict:
    n = len(FEATURE_ORDER)
    return {
        "risk": _convert(model.risk_clf, n, True),
        "yield": _convert(model.yield_reg, n, False),
        "price": _convert(model.price_clf, n, True),
    }


def _parity(onx, estimator, x: np.ndarray, regression: bool) -> float:
    sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.astype(np.float32)})[0].ravel()
    ref = estimator.predict(x)
    if regression:
        denom = np.maximum(np.abs(ref), 1e-6)
        return float((np.abs(out - ref) / denom < 0.02).mean())  # within 2%
    return float((out.astype(str) == ref.astype(str)).mean())


def run() -> dict:
    import joblib

    model_path = config.MODELS_DIR / "varimi_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train first: python -m varimi.model.train")
    model = joblib.load(model_path)

    enriched = build.build(loader.load_raw())
    x = model.encode(enriched)

    heads = build_onnx(model)
    estimators = {"risk": model.risk_clf, "yield": model.yield_reg, "price": model.price_clf}
    summary = {}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, onx in heads.items():
        path = config.MODELS_DIR / f"{name}.onnx"
        data = onx.SerializeToString()
        path.write_bytes(data)
        parity = _parity(onx, estimators[name], x, regression=(name == "yield"))
        summary[name] = {"path": str(path), "size_bytes": len(data), "parity": parity}
    return summary


if __name__ == "__main__":
    result = run()
    total = sum(v["size_bytes"] for v in result.values())
    for name, v in result.items():
        print(f"{name:6s} size={v['size_bytes']/1024:7.1f} KB  parity={v['parity']:.3f}")
    print(f"total ONNX footprint: {total/1024:.1f} KB")