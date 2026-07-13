"""Test the Android strings.json exporter covers the full catalog."""

import importlib.util
from pathlib import Path

from varimi.serving.advisory import _ACTIONS, LANGUAGES

_SPEC = importlib.util.spec_from_file_location(
    "sync_android_assets",
    Path(__file__).resolve().parents[1] / "scripts" / "sync_android_assets.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_strings_payload_covers_all_languages_and_actions():
    payload = _MOD.strings_payload()
    assert payload["languages"] == list(LANGUAGES)
    for section in ("lead", "advice_prefix"):
        assert set(payload[section]) == set(LANGUAGES)
    for section in ("risk", "price"):
        for lang in LANGUAGES:
            assert lang in payload[section]
    assert set(payload["actions"]) == set(_ACTIONS)
    for key, variants in payload["actions"].items():
        assert set(variants) == set(LANGUAGES), f"action {key} incomplete"
    assert set(payload["driver_thresholds"]) == set(payload["driver_features"])