# Edge Benchmark - VaRimi Sentinel (C4 evidence)

Scenario: one (district, crop, month) advisory = one forward pass through
all three ONNX heads, single-row input, CPU-only (no GPU), 2000 timed iterations after 200 warm-up runs.

## On-disk model footprint

| Head | Size (KB) |
|---|---|
| risk | 785.2 |
| yield | 318.8 |
| price | 759.2 |
| **Total** | **1863.2** |

Total footprint: **1.82 MB**.

## Single-query latency

| Statistic | ms |
|---|---|
| Median | 0.200 |
| p95 | 9.219 |
| Max | 30.583 |

## Budget verdict (Track-3 C4)

| Budget | Limit | Measured | Pass |
|---|---|---|---|
| Model footprint (RAM proxy) | < 256 MB | 1.82 MB | YES |
| Latency (p95) | < 100 ms | 9.219 ms | YES |

## Note on int8 quantization

The models are gradient-boosted tree ensembles exported as ONNX `TreeEnsemble` operators. Dynamic int8 quantization targets neural-net GEMM/MatMul ops and is a no-op for tree nodes, so we ship the compact fp32 tree graph. It already meets the RAM budget by more than two orders of magnitude and runs far under the latency budget on CPU, so no quantization is required to fit the edge device.
