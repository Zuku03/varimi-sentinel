# Asset & Licence Register

Full disclosure of every third-party component, model, dataset and asset used,
as required by the Track 3 Technical Delivery Standards.

## Data
| Asset | Source | Licence / Status | Notes |
|-------|--------|------------------|-------|
| `02_agriculture_climate_market_signals.csv` | POTRAZ AI4I Design-Track dataset pack | Challenge-provided, synthetic aggregate | Not official statistics; challenge use |
| `00_data_dictionary.csv` | POTRAZ AI4I dataset pack | Challenge-provided | Field definitions |
| Synthetic augmented rows | Generated in-repo (`src/varimi/synth`) | Own work (MIT) | Gaussian-copula, disclosed in reports |

## Code & libraries (runtime)
| Component | Licence | Purpose |
|-----------|---------|---------|
| numpy | BSD-3 | Numerics |
| pandas | BSD-3 | Dataframes / ETL |
| scikit-learn | BSD-3 | HistGradientBoosting models, metrics |
| scipy | BSD-3 | Stats (KS tests, copula math) |
| onnx | Apache-2.0 | Model interchange format |
| onnxruntime | MIT | Edge inference runtime |
| skl2onnx | Apache-2.0 | sklearn → ONNX export |
| matplotlib | PSF/BSD-compatible | Validation plots |

## Code & libraries (dev / demo)
| Component | Licence | Purpose |
|-----------|---------|---------|
| pytest | MIT | Test runner |
| ruff | MIT | Lint / format |
| streamlit | Apache-2.0 | Officer demo UI |

## Models
| Model | Origin | Licence | Notes |
|-------|--------|---------|-------|
| VaRimi risk/yield/price models | Trained in-repo | MIT (own work) | sklearn HistGradientBoosting → ONNX int8 |

## Fonts / icons
| Asset | Licence | Notes |
|-------|---------|-------|
| (proposal typography) Avenir or Arial | Per POTRAZ format spec | Used in final PDF only |

_No third-party pre-trained weights, proprietary datasets, or copyrighted media are used._
