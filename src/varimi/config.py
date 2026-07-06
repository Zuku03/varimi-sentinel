"""Central configuration: paths, feature/target definitions and constants.

Kept dependency-free (stdlib only) so it can be imported anywhere, including the
edge inference harness, without pulling in numpy/pandas.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT.parent / "Datasets"
DATA_02 = DATASETS_DIR / "02_agriculture_climate_market_signals.csv"
DATA_DICTIONARY = DATASETS_DIR / "00_data_dictionary.csv"

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
DOCS_DIR = REPO_ROOT / "docs"

RANDOM_SEED = 42

# --- Dataset 02 schema groups -------------------------------------------
# Exogenous predictors only (known before the outcome is realised).
NUM_FEATURES = [
    "rainfall_mm",
    "ndvi_proxy_0_1",
    "pest_incidents_reported",
    "irrigation_coverage_pct",
    "input_availability_score_0_100",
]
CAT_FEATURES = ["province", "crop", "season"]  # season engineered from month

# Targets
TARGET_RISK = "risk_level"                 # 3-class: Low / Medium / High
TARGET_YIELD = "estimated_yield_t_per_ha"  # regression (t/ha)
TARGET_PRICE_DIR = "price_direction"       # 3-class: down / flat / up (engineered)

RISK_CLASSES = ["Low", "Medium", "High"]
PRICE_DIR_CLASSES = ["down", "flat", "up"]

# Columns that must NOT be used as model features (raw targets / leakage / ids).
LEAKAGE_COLS = [
    "climate_crop_risk_score_0_100",  # risk_level is a discretised band of this
    "risk_level",
    "estimated_yield_t_per_ha",
    "avg_farmgate_price_usd_per_tonne",
    "latitude",
    "longitude",
    "district",  # high cardinality vs row count; province used instead
    "month",
]
