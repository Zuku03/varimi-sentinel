"""Edge export parity + budget test (self-contained)."""

import numpy as np
import onnxruntime as ort

from varimi.data import loader
from varimi.edge.export import build_onnx
from varimi.features import build
from varimi.model.pipeline import VarimiModel


def test_onnx_risk_parity_and_size():
    enriched = build.build(loader.load_raw())
    train, test = build.time_aware_split(enriched, test_months=2)
    model = VarimiModel()
    model.fit_encoder(enriched)
    model.fit(train, price_df=train)

    heads = build_onnx(model)
    data = heads["risk"].SerializeToString()
    assert len(data) < 256 * 1024 * 1024  # well under the RAM budget

    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    x = model.encode(test).astype(np.float32)
    onnx_pred = sess.run(None, {"input": x})[0].ravel()
    skl_pred = model.risk_clf.predict(x)
    parity = (onnx_pred.astype(str) == skl_pred.astype(str)).mean()
    assert parity == 1.0