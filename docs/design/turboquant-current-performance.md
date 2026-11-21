<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# TurboQuant performance on OGA 0.19 / ORT 1.31

Fresh three-path measurements use source `d8006b9f25c8b585f74f2be4ca8aa816e6bf223a`, the same HIP-EP/kernel
DLLs and the same Qwen3.8-27B INT4 group-128 weights in every arm. GenAI 0.19.0
with HIP-EP main's PR #2875 was built as a matching core/Python wheel against ORT
SDK 1.31.0 and used with ORT runtime 1.31.0. No TurboQuant-specific GenAI patch
is applied. These tables replace the earlier OGA 0.17 campaign; its evidence
remains available at the preceding Git revision.

At 64K, no-Ring / Ring / TurboQuant model decode latency is
95.002 / 91.919 /
89.119 ms/token. The no-Ring-to-TurboQuant
decode speed ratio is **1.066x**. Their peak tracked HIP allocation is
52.580 /
52.580 /
49.850 GiB. This is a timing and
memory experiment; numerical accuracy and judge findings remain private.

## Protocol

Windows Radeon 8060S (`gfx1151`), AC power, serial GPU jobs, HIP 7.15.26333,
Python 3.13, NumPy 2.4.3, two OpenMP/OpenBLAS threads, strict GPU offload.
**Zero warm-up calls**; all first samples are retained. Debug/performance tracing,
accuracy probes, logits readback and cache hashing are disabled for timing.
Model construction/JIT and explicit harness buffer/generator setup are outside
timing; inference-time allocation and deferred initialization remain included.
Kernel mode order rotates across rounds; model mode order rotates by length.

No Ring uses upstream GQA and FP16 KV; Ring explicitly selects dense Ring;
TurboQuant selects Ring prefill and MSE-only packed four-bit decode/append.
The model is `amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, revision
`9b37cf56a001720ba7ce98ff80ea651b02e5931b`. All modes use identical prompt
token arrays; greedy generated histories may differ.

## 1. Single-turn kernels through 256K

B=1, 24Q/4KV/D256, FP16 Q/output. Times are **milliseconds**, using three-round
medians of synchronized host wall time. Prefill measures full causal attention;
TurboQuant includes a separately timed full-cache encoding stage. Decode
includes attention scan/merge and TurboQuant rotations, and excludes KV append
for every mode. Each decode round measures 30 calls. There is no graph replay,
explicit cache displacement or online GQA tuning. Medians do not represent cold
startup; raw samples preserve first-use costs.

| Context | No Ring prefill ms | No Ring decode ms/token | Ring prefill ms | Ring decode ms/token | TurboQuant prefill ms | TurboQuant decode ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.282 | 0.0411 | 0.222 | 0.0223 | 2.112 | 0.0399 |
| 256 | 0.228 | 0.0479 | 0.273 | 0.0246 | 0.690 | 0.0402 |
| 512 | 0.363 | 0.0567 | 0.694 | 0.0378 | 1.322 | 0.0546 |
| 1K | 0.841 | 0.0692 | 1.209 | 0.0517 | 2.676 | 0.0802 |
| 2K | 2.466 | 0.0914 | 3.257 | 0.0684 | 5.786 | 0.1010 |
| 4K | 9.162 | 0.1574 | 11.676 | 0.1129 | 17.238 | 0.1420 |
| 8K | 41.361 | 0.2221 | 47.579 | 0.1505 | 58.842 | 0.2283 |
| 16K | 152.337 | 0.4173 | 178.530 | 0.3490 | 195.872 | 0.3659 |
| 32K | 558.001 | 0.7242 | 655.563 | 0.6829 | 686.507 | 0.6629 |
| 64K | 2336.335 | 1.6801 | 2692.479 | 1.2850 | 2761.306 | 1.1482 |
| 128K | 9610.752 | 3.1834 | 10724.387 | 2.6050 | 10757.720 | 2.3204 |
| 256K | 39192.764 | 6.1527 | 42502.924 | 4.9325 | 42645.415 | 4.6786 |

## 2. Single-turn model through 64K

Each cell is **prefill seconds / decode milliseconds per token / peak tracked HIP
GiB**. Timing is the mean of two measured requests; memory is their larger peak.
Each length/mode starts in a fresh process/model; the two requests have separate
generators/caches. Prefill times `append_tokens` through device synchronization.
Decode times 127 model steps after first-token selection; every request produces
128 output tokens. KV updates/compression are included. Reserved capacity is
prompt length + 256 tokens. Short-context means are sensitive to first-use costs.

| Context | No Ring: prefill s / decode ms / peak GiB | Ring: prefill s / decode ms / peak GiB | TurboQuant: prefill s / decode ms / peak GiB |
| --- | ---: | ---: | ---: |
| 128 | 0.730 / 70.463 / 15.723 | 0.812 / 71.296 / 15.723 | 0.827 / 71.602 / 15.706 |
| 256 | 1.031 / 71.412 / 15.767 | 1.055 / 71.258 / 15.767 | 1.104 / 71.466 / 15.746 |
| 512 | 1.713 / 71.066 / 15.936 | 1.617 / 71.748 / 15.936 | 1.694 / 71.919 / 15.903 |
| 1K | 2.742 / 72.236 / 16.290 | 2.748 / 71.791 / 16.290 | 2.844 / 73.053 / 16.235 |
| 2K | 4.988 / 72.949 / 16.866 | 5.006 / 72.935 / 16.866 | 5.170 / 72.902 / 16.762 |
| 4K | 9.524 / 73.621 / 18.018 | 9.506 / 73.418 / 18.018 | 9.663 / 73.779 / 17.823 |
| 8K | 19.033 / 75.234 / 20.322 | 19.035 / 74.087 / 20.322 | 19.159 / 74.174 / 19.943 |
| 16K | 38.389 / 76.653 / 24.931 | 38.934 / 76.898 / 24.931 | 39.193 / 76.840 / 24.243 |
| 32K | 80.995 / 82.291 / 34.147 | 83.445 / 81.985 / 34.147 | 84.162 / 80.601 / 32.776 |
| 64K | 181.048 / 95.002 / 52.580 | 192.578 / 91.919 / 52.580 | 193.136 / 89.119 / 49.850 |

## 3. Eight-turn append model

Cells again show **prefill s / decode ms/token / peak tracked HIP GiB**. Each mode
runs two independent conversations, each with one persistent generator/cache.
Every turn appends 4,096 new deterministic random prompt tokens and generates
128 tokens. Pending-token flush is included in append time. Minimum length
suppresses early EOS for equal work. Capacity is 33,920 tokens; turn-eight
prefill ends at 33,664 tokens. These synthetic throughput conversations are
separate from semantic multi-turn accuracy/LLM-judge evaluation.

| Turn / context | No Ring: prefill s / decode ms / peak GiB | Ring: prefill s / decode ms / peak GiB | TurboQuant: prefill s / decode ms / peak GiB |
| --- | ---: | ---: | ---: |
| 1 / 4,096 | 9.448 / 72.959 / 19.823 | 9.442 / 72.977 / 19.823 | 9.649 / 72.724 / 18.286 |
| 2 / 8,320 | 9.455 / 73.467 / 19.823 | 9.594 / 73.140 / 19.823 | 10.007 / 73.152 / 18.286 |
| 3 / 12,544 | 9.727 / 74.570 / 19.823 | 9.910 / 74.388 / 19.823 | 10.532 / 74.287 / 18.286 |
| 4 / 16,768 | 10.036 / 75.631 / 19.823 | 10.311 / 75.748 / 19.823 | 11.147 / 75.186 / 18.286 |
| 5 / 20,992 | 10.318 / 77.329 / 19.823 | 10.705 / 77.229 / 19.823 | 11.634 / 76.146 / 18.286 |
| 6 / 25,216 | 10.614 / 78.766 / 19.823 | 11.085 / 78.588 / 19.823 | 12.243 / 77.595 / 18.286 |
| 7 / 29,440 | 10.922 / 79.875 / 19.823 | 11.504 / 80.192 / 19.823 | 12.857 / 78.721 / 18.286 |
| 8 / 33,664 | 11.264 / 81.343 / 19.823 | 11.925 / 81.224 / 19.823 | 13.498 / 80.317 / 18.286 |

## Memory breakdown

The PR1366 counter hooks `hipgpu.dll`'s `hipMalloc`, `hipHostMalloc`, `hipFree`
and `hipHostFree` imports. It measures peak live **requested HIP device plus
mapped-host allocation bytes through those imports**, including retained EP
pools. Allocations outside those imports, such as runtime/BLAS-internal
allocations, are excluded. It also excludes page rounding, CPU interpreter
memory and WDDM physical residency; this is not total system RAM or all-process
GPU memory usage. The same counter/hooks apply to every mode. Weight blobs are
verified against live allocations. KV counts the model's 32 full-attention cache
tensors at their allocator size class; surplus matching allocations remain in
Other. Exact cache-address ambiguity, if any, is marked in each receipt.
Other equals peak total minus weights and reserved KV; it includes scratch,
recurrent state and retained pools. Counter reads are outside timing, while
allocation hooks are active equally in all modes. Prefill/decode peaks are
recorded separately in the raw data.

Packed KV **payload** is 132 versus 512 bytes per vector: **3.8788x smaller** at
the same capacity. Allocator padding can change the ratio of reserved KV bytes.
Total model memory does not shrink by the payload ratio.

Displayed GiB values are rounded; unrounded byte totals reconcile exactly.

| Case | Mode | Weights GiB | Reserved KV GiB | Other GiB | Peak tracked GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Single-turn 64K | no-ring | 14.8338 | 4.0156 | 33.7311 | 52.5805 |
| Single-turn 64K | ring | 14.8338 | 4.0156 | 33.7311 | 52.5805 |
| Single-turn 64K | turboquant | 14.8338 | 1.0353 | 33.9811 | 49.8501 |
| Turn 8 | no-ring | 14.8338 | 2.0703 | 2.9189 | 19.8230 |
| Turn 8 | ring | 14.8338 | 2.0703 | 2.9189 | 19.8230 |
| Turn 8 | turboquant | 14.8338 | 0.5338 | 2.9189 | 18.2864 |

At 64K, TurboQuant's Other allocation is **0.2500 GiB**
larger than dense attention. This matches the `2 * newbytes` workspace reserved
for dense fresh-prompt K/V in the [measured source](https://github.com/ROCm/hip-ep/blob/d8006b9f25c8b585f74f2be4ca8aa816e6bf223a/lib/Runtime/real/gqa.cpp#L3654):
`2 * 65,536 * 4 * 256 * 2` bytes. The allocator retains this workspace, so its
cost remains included. Existing history stays packed during subsequent append;
append uses the bounded workspace documented in the design.

## Evidence and limits

[`turboquant-current-performance.json`](turboquant-current-performance.json)
retains all 45 measured case receipts, timing samples, launch configurations,
binary/source hashes, prompt hashes and memory breakdowns for every model case.
Reproduce with the [manual performance command](../../test/long_context/README.md#performance-only-campaign-windows).
Long tests remain excluded from default CI.

Two model repeats and three kernel rounds characterize this machine/run; they
do not establish confidence intervals or generalize across workloads. A change
from the previous campaign cannot be attributed solely to OGA because main's
inference kernels also changed. This timing campaign does not resolve the
strict encoder or historical synthetic-256K numerical limitations documented
in the [design](turboquant-kv-cache.md#validation-and-known-limitations).
Accuracy reports, answers, logits and judge scores remain local pending review.
