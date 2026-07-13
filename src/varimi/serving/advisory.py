"""Advisory service: turn a (district, crop, month) into an explained, localized
recommendation. This is the product core shared by the USSD flow and the demo UI.

The AI model predicts risk / yield / price; a small, transparent rule layer maps
those predictions plus the row's driver signals into a recommended action. The
rule layer sits ON TOP of the model (it does not replace it) and exists only to
phrase the action - the classification itself is the model's.

Localization: every user-facing string is drawn from the catalogs below in
English (en), Shona (sn) and Ndebele (nd). The sn/nd renderings are indicative
drafts pending native-speaker validation - see docs/TRANSLATION_VALIDATION.md
for the sign-off sheet. They must be validated before any live deployment.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from varimi import config
from varimi.data import loader
from varimi.features import build

LANGUAGES = ("en", "sn", "nd")

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
_ADVICE_PREFIX = {"en": "Advice: ", "sn": "Zano: ", "nd": "Iseluleko: "}

# Action catalog. Keys are stable identifiers used by the rule layer below and
# by the translation validation sheet.
_ACTIONS = {
    "pests": {
        "en": "scout and treat pests; plant resistant varieties where possible",
        "sn": "ongororai minda uye rapai zvipembenene; dyarai mbeu dzinodzivirira kana zvichiita",
        "nd": "hlolani amasimu lilaphe izinambuzane; hlanyelani inhlanyelo eqinileyo nxa kusenzeka",
    },
    "irrigation": {
        "en": "prioritise irrigation and drought-tolerant inputs",
        "sn": "koshesai kudiridza uye mbeu dzinotsungirira kushaya mvura",
        "nd": "qakathekisani ukuthelela lenhlanyelo ebekezelela isomiso",
    },
    "monitor_high": {
        "en": "increase field checks and secure inputs early",
        "sn": "wedzerai kuongorora minda uye chengetedzai zvinodiwa nekukurumidza",
        "nd": "andisani ukuhlola amasimu liqoqe okudingekayo masinyane",
    },
    "monitor_medium": {
        "en": "keep watching conditions and maintain input supply",
        "sn": "rambai muchiongorora mamiriro uye chengetedzai zvinodiwa",
        "nd": "qhubekani lihlola isimo ligcine okudingekayo",
    },
    "proceed": {
        "en": "conditions look good; proceed with the normal plan",
        "sn": "mamiriro akanaka; endererai mberi nehurongwa hwenyu",
        "nd": "isimo sihle; qhubekani ngohlelo lwenu",
    },
    "sell_window": {
        "en": "a good market window is forming",
        "sn": "nguva yakanaka yekutengesa iri kuuya",
        "nd": "isikhathi esihle sokuthengisa siyeza",
    },
    "hold_sale": {
        "en": "think about storage or a later sale",
        "sn": "fungai kuchengeta kana kunonoka kutengesa",
        "nd": "cabangani ukugcina kumbe ukuphuzisa ukuthengisa",
    },
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


def _action_keys(risk: str, price: str, drivers: list[dict]) -> list[str]:
    """Pick the action-catalog keys for this prediction (language-independent).

    Mirrored in Kotlin (android/.../MainActivity.kt, actionKeys()) - keep both
    in sync. The string catalogs flow to the app via scripts/sync_android_assets.py.
    """
    bad = {d["feature"] for d in drivers if d["unfavourable"]}
    keys: list[str] = []
    if risk == "High":
        if "pest_incidents_reported" in bad:
            keys.append("pests")
        if "ndvi_proxy_0_1" in bad or "irrigation_coverage_pct" in bad:
            keys.append("irrigation")
        if not keys:
            keys.append("monitor_high")
    elif risk == "Medium":
        keys.append("monitor_medium")
    else:
        keys.append("proceed")
    if price == "up":
        keys.append("sell_window")
    elif price == "down":
        keys.append("hold_sale")
    return keys


def _recommend(risk: str, price: str, drivers: list[dict], language: str) -> str:
    keys = _action_keys(risk, price, drivers)
    joined = "; ".join(_ACTIONS[k][language] for k in keys)
    return _ADVICE_PREFIX[language] + joined


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