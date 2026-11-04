<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Current-branch TurboQuant performance

Revision scope: these tables retain the measured source revision below. The
subsequent merge of main `6a52f055ee0cdf3ffb376163104c371cadba68eb` changes
GQA host-length handling. Its sanity validation is separate; these timings have
not been remeasured on that merge.

All three configurations use source `112c3c5b98d82666a9d75205ed149b1d5e8f53d8`, merged main/Ring PR1366, and the same installed HIP-EP/kernel DLLs. No historical kernel measurements or accuracy-diagnostic timings are mixed into these tables. No warm-up calls; every measured request from the final protocol is retained. First-use costs remain in raw samples. Kernel tables use three-round medians; model tables use the arithmetic mean of two measured requests/conversations.

Windows Radeon 8060S (`gfx1151`), AC power, serial GPU jobs; HIP 7.15.26333, ORT 1.31.0, GenAI 0.17.0 plus the documented append companion, NumPy 2.4.3, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`. The companion was built against ORT 1.30 SDK and runs with ORT 1.31; HIP-EP was built against 1.31. Debug/performance tracing is disabled. Model construction and explicit harness buffer/generator setup are outside timing. Runtime allocations and deferred initialization performed inside inference remain included. No online GQA tuning: no-Ring uses the current DLL's normal lookup policy/configured entry points.

Modes: no-Ring is default GQA with FP16 cache; Ring explicitly enables dense Ring; TurboQuant enables Ring prefill and MSE-only 4-bit packed KV decode. The model uses identical INT4 group-128 weights in every arm (`amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, revision `9b37cf56a001720ba7ce98ff80ea651b02e5931b`). This campaign evaluates latency, not semantic quality.

## 1. Single-turn kernel performance through 256K

B=1, 24 query heads, 4 KV heads, D=256, FP16 Q/output. Fresh causal prefill has query length equal to context. TurboQuant prefill is measured Ring attention plus separately synchronized full-cache encoding; the reported sum includes both stage boundaries. Dense cache preparation/transpose, allocation and transfers are excluded for all arms. Decode has one query and includes attention scan/merge and TurboQuant rotations; KV append/encoding is excluded from decode kernel timing (included in full-model timing). Prefill is three measured single calls; decode is three measured batches of 30 calls with synchronization at each batch boundary. Mode order rotates across rounds. Synchronized host wall time is used, not zero-valued HIP event timestamps. No graph replay or explicit cache displacement. Kernel decode ms/token here means one attention layer, not the entire model.

| Context | No Ring prefill ms | No Ring decode ms/token | Ring prefill ms | Ring decode ms/token | TurboQuant prefill ms | TurboQuant decode ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.328 | 0.0379 | 0.229 | 0.0271 | 0.561 | 0.0382 |
| 256 | 0.252 | 0.0485 | 0.294 | 0.0302 | 0.767 | 0.0460 |
| 512 | 0.447 | 0.0624 | 0.524 | 0.0422 | 1.543 | 0.0611 |
| 1K | 0.904 | 0.0678 | 3.041 | 0.0635 | 3.927 | 0.0890 |
| 2K | 2.462 | 0.0892 | 4.356 | 0.0731 | 6.927 | 0.1115 |
| 4K | 10.677 | 0.1494 | 13.735 | 0.1135 | 19.302 | 0.1744 |
| 8K | 43.908 | 0.2579 | 51.821 | 0.2410 | 62.780 | 0.3225 |
| 16K | 164.256 | 0.4001 | 200.050 | 0.3631 | 219.456 | 0.3844 |
| 32K | 587.200 | 0.7139 | 755.123 | 0.6874 | 778.732 | 0.6348 |
| 64K | 2400.479 | 1.6747 | 2716.720 | 1.2994 | 2907.068 | 1.1622 |
| 128K | 9826.987 | 3.2237 | 10800.230 | 2.6642 | 11003.215 | 2.3799 |
| 256K | 43947.246 | 6.2029 | 43447.455 | 4.9955 | 43799.537 | 4.8205 |

## 2. Single-turn model performance through 64K

Batch 1, unchunked deterministic random token prompts shared across modes, 128 greedy output tokens. Prefill measures `append_tokens` through device synchronization; decode measures 127 steps after first-token selection, with final synchronization. Every length/configuration starts in a fresh process/model; two requests use separate generators/caches. Short-context means may be sensitive to launch overhead; raw samples are retained. No logits readback, cache hashing, accuracy probes or tracing occurs inside timed regions.

Each model cell is **prefill seconds / decode milliseconds per token / peak total HIP GiB**. Timing is the mean of two requests; memory is the larger peak across those requests. Total = weights + reserved KV allocations + Other, including retained EP pools. Weights are **14.8338 GiB** in every arm; raw receipts include KV payload, allocator padding, Other, and separate prefill/decode peaks. The PR1366 allocation counter tracks live HIP device plus mapped-host requested bytes; it excludes driver-internal allocations, CPU interpreter memory, page rounding and WDDM physical residency. Current main allocates the KV on device by default; the reporting helper recognizes both device and mapped-host allocations without changing that policy. Counter reads are outside timed regions and allocation hooks are active equally for all paths. This is not total physical system-RAM consumption. Reserved capacity is prompt plus 128 output tokens plus 128 spare tokens for a single turn.

| Context | No Ring: prefill s / decode ms / peak GiB | Ring: prefill s / decode ms / peak GiB | TurboQuant: prefill s / decode ms / peak GiB |
| --- | ---: | ---: | ---: |
| 128 | 0.646 / 74.338 / 15.723 | 0.698 / 73.648 / 15.723 | 0.743 / 74.044 / 15.706 |
| 256 | 1.008 / 74.384 / 15.767 | 0.990 / 74.174 / 15.767 | 1.045 / 74.312 / 15.746 |
| 512 | 1.600 / 74.315 / 15.971 | 1.651 / 74.162 / 15.971 | 1.679 / 74.647 / 15.938 |
| 1K | 2.851 / 74.823 / 16.360 | 2.876 / 74.640 / 16.360 | 2.953 / 74.997 / 16.306 |
| 2K | 5.305 / 75.300 / 17.007 | 5.276 / 75.062 / 17.007 | 5.386 / 75.764 / 16.903 |
| 4K | 10.062 / 75.784 / 18.299 | 10.269 / 75.795 / 18.299 | 10.320 / 75.892 / 18.104 |
| 8K | 20.140 / 77.273 / 20.885 | 20.139 / 77.173 / 20.885 | 20.530 / 77.527 / 20.506 |
| 16K | 40.848 / 79.912 / 26.056 | 41.452 / 79.761 / 26.056 | 41.703 / 80.001 / 25.368 |
| 32K | 85.884 / 85.204 / 36.398 | 88.587 / 85.205 / 36.398 | 89.145 / 84.275 / 35.027 |
| 64K | 191.381 / 98.837 / 57.082 | 202.259 / 95.369 / 57.082 | 203.823 / 92.326 / 54.351 |

## 3. Eight-turn append model performance

Two independent eight-turn conversations per mode, one persistent generator/cache per conversation. Each turn appends exactly 4,096 new deterministic random prompt tokens and generates exactly 128 tokens. Minimum length suppresses EOS early termination for equal work; generated histories are free-running and need not have identical token IDs. Context includes prior generated tokens, so turn 8 prefill ends at 33,664 tokens. Appending processes any outstanding last generated token through the companion's normal lifecycle, included in append time. This is a synthetic append-throughput workload; the separate semantic multi-turn/LLM-judge investigation remains private.

Cells use the same **prefill s / decode ms / peak total HIP GiB** format and memory scope as table 2. Full-conversation reserved cache capacity is 33,920 tokens in each mode; peaks are recorded for each turn, and retained allocator pools remain charged to Other.

| Turn / context | No Ring: prefill s / decode ms / peak GiB | Ring: prefill s / decode ms / peak GiB | TurboQuant: prefill s / decode ms / peak GiB |
| --- | ---: | ---: | ---: |
| 1 / 4,096 | 10.102 / 76.503 / 20.107 | 10.160 / 76.430 / 20.107 | 10.292 / 76.440 / 18.570 |
| 2 / 8,320 | 10.043 / 76.464 / 20.107 | 10.034 / 76.750 / 20.107 | 10.552 / 76.956 / 18.570 |
| 3 / 12,544 | 10.253 / 77.672 / 20.107 | 10.495 / 77.751 / 20.107 | 11.049 / 78.170 / 18.570 |
| 4 / 16,768 | 10.550 / 79.183 / 20.107 | 10.849 / 79.276 / 20.107 | 11.837 / 78.948 / 18.570 |
| 5 / 20,992 | 10.836 / 80.520 / 20.107 | 11.252 / 80.482 / 20.107 | 12.219 / 80.065 / 18.570 |
| 6 / 25,216 | 11.170 / 81.752 / 20.107 | 11.645 / 81.716 / 20.107 | 12.771 / 81.405 / 18.570 |
| 7 / 29,440 | 11.471 / 83.365 / 20.107 | 12.029 / 83.357 / 20.107 | 13.322 / 82.344 / 18.570 |
| 8 / 33,664 | 11.765 / 84.699 / 20.107 | 12.432 / 84.743 / 20.107 | 13.973 / 83.501 / 18.570 |

Memory breakdown at the longest model cases, GiB (**weights + KV + Other = peak total**):

- Single-turn 64K, no-ring: 14.8338 + 4.0156 + 38.2322 = 57.0816.
- Single-turn 64K, ring: 14.8338 + 4.0156 + 38.2322 = 57.0816.
- Single-turn 64K, turboquant: 14.8338 + 1.0353 + 38.4822 = 54.3512.
- Turn 8, no-ring: 14.8338 + 2.0703 + 3.2026 = 20.1066.
- Turn 8, ring: 14.8338 + 2.0703 + 3.2026 = 20.1066.
- Turn 8, turboquant: 14.8338 + 0.5338 + 3.2026 = 18.5701.

Initial 128-token timing-only model attempts were superseded when memory accounting was requested; the final model campaign reran those cases in fresh processes. Two memory-attribution setup assertions were corrected before their cases reached inference. These attempts remain archived separately and are excluded from the final dataset; no slow sample from the final protocol was discarded.

KV attribution uses the known 32 full-attention cache tensors and allocator-rounded sizes, checked against live allocations. At 512-token dense cases, recurrent state shares the same size class; the budget counts only 32 KV buffers and charges the extras to Other. The receipt explicitly marks ambiguous cache-address attribution. Total peak memory is directly counted and does not depend on that split.

## Evidence and limits

The accompanying `turboquant-current-performance.json` contains every measured timing, resolved kernel launch configuration, binary hashes, prompt hashes and provenance. Reproduce with the explicit manual performance command documented in `test/long_context/README.md`; it is not part of default CI. Two model repeats and three kernel rounds characterize this machine/run, not statistical confidence intervals or all workloads. Initial DLL/kernel setup can be much larger than subsequent calls, especially at short context; no samples were removed from the raw records. Kernel medians do not describe cold process startup. Multi-turn accuracy, numerical deviation, answers and judge scores are deliberately excluded from this report and the PR update. Historical 256K numerical limitations remain unresolved by timing these kernels. Windows RMT workflow is unchanged.
