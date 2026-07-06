"""Advisory service: turn a (district, crop, month) into an explained, localized
recommendation. This is the product core shared by the USSD flow and the demo UI.

The AI model predicts risk / yield / price; a small, transparent rule layer maps
those predictions plus the row's driver signals into a recommended action. The
rule layer sits ON TOP of the model (it does not replace it) and exists only to
phrase the action - the classification itself is the model's.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from varimi import config
from varimi.data import loader
from varimi.features import build

LANGUAGES = ("en", "sn", "nd")

# Indicative translations for demonstration. A production deployment must have
# these validated by native Shona/Ndebele speakers before going live.
_RISK_LOCAL = {
    "en": {"Low": "Low", "Medium": "Medium", "High": "High"},
    "sn": {"Low": "Yakaderera", "Medium": "Yepakati", "High": "Yakakwirira"},
    "nd": {"Low": "Ephansi", "Medium": "Ephakathi", "High": "Ephezulu"},
}
_PRICE_LOCAL = {
    "en": {"down": "falling", "flat": "stable", "up": "rising"},
    "sn": {"down": "ari kudzikira", "flat": "akagadzikana", "up": "ari kukwira"},
    "nd": {"down": "ehla", "flat": "amile", "up": "enyuka"},
}
_LEAD = {
    "en": "{crop} in {district} ({month}): risk {risk}, price {price}.",
    "sn": "{crop} muno {district} ({month}): njodzi {risk}, mutengo {price}.",
    "nd": "{crop} e-{district} ({month}): ingozi {risk}, intengo {price}.",
}

# Driver features ranked by the model's global permutation importance.
_DRIVER_FEATURES = ("pest_incidents_reported", "ndvi_proxy_0_1", "irrigation_coverage_pct")


@lru_cache(maxsize=1)
def _context():
    import joblib

    model = joblib.load(config.MODELS_DIR / "varimi_model.joblib")
    enriched = build.build(loader.load_raw())
    return model, enriched


def _lookup_row(
    enriched: pd.DataFrame, district: str, crop: str, month: str | None
) -> pd.DataFrame:
    subset = enriched[(enriched["district"] == district) & (enriched["crop"] == crop)]
    if subset.empty:
        raise ValueError(f"No data for district={district!r}, crop={crop!r}")
    if month is not None:
        picked = subset[subset["month"] == month]
        if not picked.empty:
            return picked.head(1)
        # Requested month has no row for this crop; fall back to the latest month.
    latest = sorted(subset["month"].unique())[-1]
    return subset[subset["month"] == latest].head(1)


def _drivers(row: pd.Series, enriched: pd.DataFrame) -> list[dict]:
    out = []
    for feat in _DRIVER_FEATURES:
        pct = float((enriched[feat] <= row[feat]).mean())
        unfavourable = pct >= 0.66 if feat == "pest_incidents_reported" else pct <= 0.34
        out.append({"feature": feat, "percentile": round(pct, 2), "unfavourable": unfavourable})
    return out


def _recommend(risk: str, price: str, drivers: list[dict], language: str) -> str:
    bad = {d["feature"] for d in drivers if d["unfavourable"]}
    en = []
    if risk == "High":
        if "pest_incidents_reported" in bad:
            en.append("scout and treat pests; consider resistant varieties")
        if "ndvi_proxy_0_1" in bad or "irrigation_coverage_pct" in bad:
            en.append("prioritise irrigation and drought-tolerant inputs")
        if not en:
            en.append("increase monitoring and secure inputs early")
    elif risk == "Medium":
        en.append("monitor conditions and maintain input supply")
    else:
        en.append("conditions favourable; proceed with the normal plan")
    if price == "up":
        en.append("a favourable selling window is forming")
    elif price == "down":
        en.append("consider storage or delaying sale")
    action = "; ".join(en)
    if language == "sn":
        return "Zano: " + action
    if language == "nd":
        return "Iseluleko: " + action
    return "Advice: " + action


def advise(district: str, crop: str, month: str | None = None, language: str = "en") -> dict:
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}")
    model, enriched = _context()
    row_df = _lookup_row(enriched, district, crop, month)
    row = row_df.iloc[0]
    preds = model.predict(row_df)
    risk = str(preds[config.TARGET_RISK][0])
    price = str(preds[config.TARGET_PRICE_DIR][0])
    yield_val = float(preds[config.TARGET_YIELD][0])
    drivers = _drivers(row, enriched)

    lead = _LEAD[language].format(
        crop=crop, district=district, month=row["month"],
        risk=_RISK_LOCAL[language][risk], price=_PRICE_LOCAL[language][price],
    )
    action = _recommend(risk, price, drivers, language)
    return {
        "district": district,
        "crop": crop,
        "month": row["month"],
        "language": language,
        "risk_level": risk,
        "yield_t_per_ha": round(yield_val, 2),
        "price_direction": price,
        "drivers": drivers,
        "recommended_action": action,
        "message": f"{lead} {action}",
    }