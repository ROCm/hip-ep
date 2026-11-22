<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen native memory-only matrix

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
