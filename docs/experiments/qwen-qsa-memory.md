<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen native memory-only matrix

<!-- QWEN_EIGHT_PERFORMANCE_BEGIN -->
## Latest eight-selector performance

All memory measurements below belong to four selectors. Host RAM guards in the eight-selector performance run are not a new GPU-memory comparison.

**Decode throughput improved over four selectors at every measured length.** Prefill is slower at 4K/16K and slightly faster at 32K. All eight weights, including the overlapping four positions, come from checkpoint 640; this is not a layer-count-only change.

| Context | Eight vs four: prefill | Eight vs four: decode |
|---|---:|---:|
| 4K | -5.21% | +0.71% |
| 16K | -6.31% | +1.94% |
| 32K | +0.92% | +5.31% |

Eight uses one warmup and three measured requests at batch one, with 64 common teacher-forced decode inputs. Default/four controls are reused across runs (two warmups, six requests, two resident sessions sharing cache buffers); Eight has one resident session. No paired confidence interval or isolated layer-count causal claim applies. Both prefill and decode use QSA at all eight selected layers.

Eight-selector BFCL/GSM8K accuracy and application-memory comparisons have not been run on hip-ep. The source checkpoint still misses its fixed 32K PPL gate. Historical four-selector accuracy, memory and paired intervals do not transfer to Eight. See the [eight-selector report](qwen-qsa-eight-selector-results.md) and [raw-data paper handoff](../reproducibility/qwen-eight-hip-ep/README.md).
<!-- QWEN_EIGHT_PERFORMANCE_END -->

The user selected batch1/4K as the single memory comparison and canceled all remaining probes. No probes are queued. The already completed batch4/4K pair is supplementary. Batch8/4K default was interrupted during prefill; its values are lower bounds only.

One fresh hip-ep session per process, default and frozen QSA measured separately. Each probe runs one full prefill and eight fixed decode inputs, reserving prefix+64 KV slots. No repeated performance benchmark is run. 64K is excluded at the user's request.

Numbers are GiB. GPU columns are peak Windows WDDM process **total commitment**, not physical residency. Process RSS is shown separately and must not be added to GPU commitment. Sampled GPU peaks can miss sub-100ms spikes; OS lifetime process high-water values and all phase traces are retained.

| Batch | Prefix | Default status | QSA status | Default GPU commit | QSA GPU commit | Default process RSS | QSA process RSS |
|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 4K | complete | complete | 19.34 | 19.58 | 2.95 | 2.95 |
| 4 | 4K | complete | complete | 26.21 | 27.09 | 2.95 | 2.95 |
| 8 | 4K | canceled_user | canceled_user | 35.34† | — | 2.95† | — |
| 1 | 16K | canceled_user | canceled_user | — | — | — | — |
| 4 | 16K | canceled_user | canceled_user | — | — | — | — |
| 8 | 16K | canceled_user | canceled_user | — | — | — | — |
| 1 | 32K | canceled_user | canceled_user | — | — | — | — |
| 4 | 32K | canceled_user | canceled_user | — | — | — | — |
| 8 | 32K | canceled_user | canceled_user | — | — | — | — |

A `capacity_limit` case was stopped promptly by the 4 GiB available-system-RAM guard. † marks values observed before an incomplete probe stopped; these are lower bounds, not completed configuration peaks. Raw samples, load/prefill/decode events, worker results, errors and identities are retained under `_local/qwen-qsa/memory-only-v1/`. Per-phase peaks and OS lifetime high-water counters are in `matrix.json`.
