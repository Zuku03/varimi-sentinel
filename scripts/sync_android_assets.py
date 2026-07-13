"""Sync model + data assets into the Android harness.

Copies the exported ONNX heads, generates the encoder vocabulary JSON from the
fitted OrdinalEncoder (so Kotlin never hardcodes category codes), writes a
lookup CSV snapshot (per district/crop/month feature rows), and exports the
localization catalogs + rule-layer constants as strings.json so the app renders
the exact advisory wording defined in varimi.serving.advisory (single source of
truth - the translation validation sheet flows here too).

Run: python scripts/sync_android_assets.py   (after model.train + edge.export)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib

from varimi import config
from varimi.data import loader
from varimi.features import build
from varimi.serving import advisory

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"


def strings_payload() -> dict:
    """Localization catalogs + driver rule constants for the Android app.

    The unfavourable-driver thresholds mirror advisory._drivers(): pest
    incidents are unfavourable at a high percentile (>= 0.66), NDVI and
    irrigation at a low percentile (<= 0.34).
    """
    return {
        "languages": list(advisory.LANGUAGES),
        "lead": advisory._LEAD,
        "advice_prefix": advisory._ADVICE_PREFIX,
        "risk": advisory._RISK_LOCAL,
        "price": advisory._PRICE_LOCAL,
        "actions": advisory._ACTIONS,
        "driver_features": list(advisory._DRIVER_FEATURES),
        "driver_thresholds": {
            "pest_incidents_reported": {"direction": "high", "cutoff": 0.66},
            "ndvi_proxy_0_1": {"direction": "low", "cutoff": 0.34},
            "irrigation_coverage_pct": {"direction": "low", "cutoff": 0.34},
        },
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for head in ("risk", "yield", "price"):
        src = config.MODELS_DIR / f"{head}.onnx"
        if not src.exists():
            raise FileNotFoundError(f"{src} missing - run: python -m varimi.edge.export")
        shutil.copy2(src, ASSETS / src.name)

    model = joblib.load(config.MODELS_DIR / "varimi_model.joblib")
    vocab = {
        col: [str(v) for v in cats]
        for col, cats in zip(config.CAT_FEATURES, model.encoder.categories_, strict=True)
    }
    (ASSETS / "encoder_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    enriched = build.build(loader.load_raw())
    cols = ["month", "province", "district", "crop", "season"] + config.NUM_FEATURES
    enriched[cols].to_csv(ASSETS / "lookup.csv", index=False)

    (ASSETS / "strings.json").write_text(
        json.dumps(strings_payload(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Synced ONNX heads, encoder_vocab.json, lookup.csv and strings.json to {ASSETS}")


if __name__ == "__main__":
    main()