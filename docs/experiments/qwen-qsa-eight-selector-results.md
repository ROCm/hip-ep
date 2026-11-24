<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen eight-selector QSA: performance through 32K

Qwen/Qwen3.8-27B on Radeon 8060S/gfx1151, Windows. The unchanged native W4A16 group128 backbone uses FP16 activation/KV and FP32 recurrent state; selectors are BF16. Zero-based QSA layers are **3, 7, 23, 27, 31, 35, 39, 63**. No training, learned calibration or kernel/compiler rebuild occurred.

**Decode throughput improved over four selectors at every measured length.** Prefill is slower at 4K/16K and slightly faster at 32K. All eight weights, including the overlapping four positions, come from checkpoint 640; this is not a layer-count-only change.

| Context | Phase | Default tok/s (reused) | Four QSA tok/s (reused) | Eight QSA tok/s | Eight vs default | Eight vs four |
|---|---|---:|---:|---:|---:|---:|
| 4K | prefill | 438.06 | 419.97 | 398.08 | -9.13% | -5.21% |
| 4K | decode | 10.68 | 10.74 | 10.81 | +1.21% | +0.71% |
| 16K | prefill | 415.94 | 397.34 | 372.26 | -10.50% | -6.31% |
| 16K | decode | 10.13 | 10.25 | 10.45 | +3.20% | +1.94% |
| 32K | prefill | 384.85 | 350.87 | 354.10 | -7.99% | +0.92% |
| 32K | decode | 9.36 | 9.51 | 10.01 | +6.99% | +5.31% |

Eight uses one warmup and three measured requests at batch one, with 64 common teacher-forced decode inputs. Default/four controls are reused across runs (two warmups, six requests, two resident sessions sharing cache buffers); Eight has one resident session. No paired confidence interval or isolated layer-count causal claim applies. Both prefill and decode use QSA at all eight selected layers.

Each Eight value is phase token count divided by mean synchronized CPU-wall seconds over all three measured requests, not mean per-request TPS. The raw rows include one warmup (`trial=-1`) and three measured requests (`0,1,2`). Allocation, binding, recurrent zeroing and load/compile are outside phase timers. Finite outputs, exact replay, stable cache addresses and unchanged sessions pass at every shape.

The 140W charger rating is user-reported. All sampled AC states stayed connected. The separate Original 4K charger control recovered to 431.71 prefill / 10.79 decode tok/s, within 1.45% / 1.01% of the archive. Only the 4K Original was freshly checked after the charger change. The earlier pre-charger/suspected-contention run remains preserved and excluded from this clean comparison; a specific cause is not proven by the before/after observation.

The locked timing prefixes concatenate known 4K test windows; they are not a natural-long quality holdout. 64K and batched measurements remain excluded.

Eight-selector BFCL/GSM8K accuracy and application-memory comparisons have not been run on hip-ep. The source checkpoint still misses its fixed 32K PPL gate. Historical four-selector accuracy, memory and paired intervals do not transfer to Eight.

See the [paper handoff](../reproducibility/qwen-eight-hip-ep/README.md), [performance JSON](../reproducibility/qwen-eight-hip-ep/eight-140w-performance.json), [CSV](../reproducibility/qwen-eight-hip-ep/eight-140w-performance.csv) and [runtime-estimate JSON](../reproducibility/qwen-eight-hip-ep/evaluation-time-estimate.json). The handoff links all raw rows, saved aggregates, protocols, exact identities and local immutable ZIP receipts. Runtime estimates are planning ranges; no accuracy evaluation is scheduled.

Regenerate the linked performance documentation with `python scripts/qwen_qsa/performance_docs.py` from the repository root. It verifies the saved raw-row arithmetic and hashes and uses no GPU.
