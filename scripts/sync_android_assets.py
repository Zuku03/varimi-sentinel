"""Sync model + data assets into the Android harness.

Copies the exported ONNX heads, generates the encoder vocabulary JSON from the
fitted OrdinalEncoder (so Kotlin never hardcodes category codes), and writes a
lookup CSV snapshot (per district/crop/month feature rows) so the app can run
fully offline.

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

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"


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
    print(f"Synced ONNX heads, encoder_vocab.json and lookup.csv to {ASSETS}")


if __name__ == "__main__":
    main()