"""Compatibility shim for skl2onnx + protobuf>=7.

protobuf 7 tightened type checking and rejects Python/numpy ``bool`` values in
ONNX integer-list attributes (raising "Expected an int, got a boolean"). skl2onnx
still emits booleans for tree-ensemble node attributes such as
``nodes_missing_value_tracks_true``. This shim wraps ``onnx.helper.make_attribute``
to coerce those bools to ints. Call :func:`apply` once before any conversion.
"""

from __future__ import annotations

import numpy as np
import onnx.helper as _oh

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    _orig = _oh.make_attribute

    def _patched(key, value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.dtype == bool:
            value = value.astype(np.int64)
        elif isinstance(value, (list, tuple)) and any(
            isinstance(v, (bool, np.bool_)) for v in value
        ):
            value = [int(v) for v in value]
        return _orig(key, value, *args, **kwargs)

    _oh.make_attribute = _patched
    _applied = True