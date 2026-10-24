<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
<!-- ring-main-refresh-1259 -->
# Ring attention: baseline refresh after PR #1259

Fresh Windows measurements compare old-main GQA, updated-main GQA, and both Ring selections. All four use the same model, runtime, toolchain, and measurement protocol. Historical lx01 measurements are retained separately below.

- Updated Ring source: `e99e1ae046a9dda10417a95135e46e750bc8f3d8`; upstream main `65d1f5a276402d1ab33474f268af9350791a3cba` (includes #1259).
- Old-main control: `5f545550038f35d2c4742a9ff8ae4ff63b9efad8`, based on main `a4f9826b809c9483fe2353eea876c4c2257a5ad4`, compiled with `HIP_EP_GQA_IMPL=existing`.
- Main changed beyond #1259, including GEMV tuning. The revision comparison does not isolate the effect of the HpG=6 change alone.
- Model: `amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, revision `9b37cf56a001720ba7ce98ff80ea651b02e5931b`; INT4 group128 weights, FP16 KV, batch 1.
- Windows 11 Enterprise, Ryzen AI MAX+ PRO 395 / Radeon 8060S (`gfx1151`), driver `32.0.22050.4002`, AC power, Balanced plan. TheRock/HIP 7.15, ONNX Runtime 1.31.0 and GenAI 0.17.0 for every configuration.
- One diagnostic/warmup request plus three measured requests, identical seeded prompt prefixes, unchunked prefill, 128 greedy output tokens, fresh Generator state per request. Serial GPU execution after builds finish; HIP-synchronized wall-clock timings.
- Case order: 4K/16K old main, new main, Ring-Q, Ring-KV; 32K new main, Ring-Q, Ring-KV, old main; 64K Ring-Q, Ring-KV, old main, new main. Requests were grouped by case rather than interleaved; the separate late 32K control below measures variation over the campaign.
- An AC disconnect interrupted the first new-main 32K case. That entire case was excluded and rerun after power was restored; the table uses only the completed rerun.

## Findings

- At 64K, the new-main decode median was 17.24% higher than the fresh old-main control. Ring-Q and Ring-KV were 4.67% and 5.09% lower in throughput than new main in this campaign. The historical Ring decode advantage is not reproduced against this updated baseline; the small gaps are subject to the variability caveat below.
- All four 64K prefill allocation peaks are approximately 56.90 GiB; their spread is only 13.12 MiB. These measurements do not establish a material full-model memory reduction for Ring.
- Repeating the unchanged new-main 32K case late in the campaign changed prefill by +20.25% and decode throughput by -3.50%. This substantial prefill variation prevents a reliable prefill ranking from these grouped runs. Small Ring-versus-new-main decode gaps are also comparable with the observed variation and should not be treated as proven regressions. The primary tables are retained as measured, with the control shown separately.

## Prefill

Median full-model prefill seconds; lower is better.

| Input tokens | Old-main GQA | New-main GQA | Ring-Q | Ring-KV |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 9.468 | 9.488 | 9.436 | 9.443 |
| 16,384 | 40.463 | 40.595 | 40.323 | 40.039 |
| 32,768 | 104.295 | 90.666 | 95.312 | 99.047 |
| 65,536 | 238.818 | 242.934 | 230.192 | 227.006 |

<details>
<summary>Prefill seconds across the three measured requests (minimum–maximum)</summary>

| Tokens | Old-main GQA | New-main GQA | Ring-Q | Ring-KV |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 9.448–9.547 | 9.482–9.511 | 9.415–9.514 | 9.429–9.453 |
| 16,384 | 40.392–40.618 | 40.530–40.771 | 40.253–40.512 | 40.010–40.163 |
| 32,768 | 103.433–106.444 | 88.093–92.021 | 94.021–96.506 | 98.560–100.022 |
| 65,536 | 236.783–240.314 | 241.200–250.813 | 230.108–230.236 | 226.898–227.675 |

</details>

## Decode

Median full-model throughput for the 127 tokens after first-token selection; higher is better.

| Input tokens | Old-main GQA tok/s | New-main GQA tok/s | Ring-Q tok/s | Ring-KV tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 12.339 | 12.793 | 12.703 | 12.807 |
| 16,384 | 11.327 | 12.104 | 11.860 | 11.921 |
| 32,768 | 9.749 | 11.411 | 10.750 | 10.579 |
| 65,536 | 8.161 | 9.568 | 9.121 | 9.081 |

Decode throughput changes, computed from the medians above. Small differences should be read alongside the three-run ranges, not as statistically established gains.

| Tokens | New main vs old main | Ring-Q vs new main | Ring-KV vs new main |
| ---: | ---: | ---: | ---: |
| 4,096 | +3.68% | -0.70% | +0.11% |
| 16,384 | +6.87% | -2.02% | -1.51% |
| 32,768 | +17.04% | -5.79% | -7.29% |
| 65,536 | +17.24% | -4.67% | -5.09% |

<details>
<summary>Decode throughput ranges across the three measured requests</summary>

| Tokens | Old-main GQA | New-main GQA | Ring-Q | Ring-KV |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 12.088–12.354 | 12.628–12.807 | 12.626–12.703 | 12.714–12.825 |
| 16,384 | 11.308–11.328 | 11.967–12.118 | 11.734–11.991 | 11.821–12.012 |
| 32,768 | 9.251–9.972 | 11.402–11.431 | 10.692–10.782 | 10.477–10.660 |
| 65,536 | 8.068–8.309 | 9.547–9.646 | 9.119–9.155 | 9.059–9.084 |

</details>

## Repeatability check at 32K

After the main campaign, new-main GQA was repeated with the same prompt, model, runtime, harness, and DLL hashes. Both runs use one warmup and three measured requests. The primary table is preserved; the repeat is not substituted for a more favorable result. Compare implementation differences with the variation observed over the campaign.

| Metric | Primary median (range) | Late repeat median (range) | Late vs primary |
| --- | ---: | ---: | ---: |
| Prefill seconds | 90.666 (88.093–92.021) | 109.030 (106.318–112.858) | +20.25% |
| Decode tokens/s | 11.411 (11.402–11.431) | 11.012 (10.709–11.035) | -3.50% |

The repeat passed the same prefill and aligned first-decode logit thresholds; 128/128 generated token positions matched the primary baseline.


## Memory

GiB. Total = model constants + verified KV allocations + other. Peak is the maximum live requested HIP device **plus mapped-host** payload allocated through `hipgpu.dll` within the phase, across measured requests. KV counts the 32 verified full-attention K/V buffers. Other includes recurrent state, activations, workspace, and retained allocation pools. Driver allocations, page rounding, and CPU interpreter memory do not count. This is full-model EP allocation accounting, not physical VRAM residency or isolated attention scratch. Prefill and decode peaks can be equal because prefill allocation pools remain live during decode.

| Tokens | Implementation | Model | KV | Prefill other | Prefill total peak | Decode other | Decode total peak |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | Old-main GQA | 14.8338 | 0.2578 | 3.0227 | 18.1143 | 3.0227 | 18.1143 |
| 4,096 | New-main GQA | 14.8338 | 0.2578 | 3.0234 | 18.1149 | 3.0234 | 18.1149 |
| 4,096 | Ring-Q | 14.8338 | 0.2578 | 3.0223 | 18.1138 | 3.0223 | 18.1138 |
| 4,096 | Ring-KV | 14.8338 | 0.2578 | 3.0223 | 18.1138 | 3.0223 | 18.1138 |
| 16,384 | Old-main GQA | 14.8338 | 1.0078 | 10.0316 | 25.8732 | 10.0316 | 25.8732 |
| 16,384 | New-main GQA | 14.8338 | 1.0078 | 10.0298 | 25.8714 | 10.0298 | 25.8714 |
| 16,384 | Ring-Q | 14.8338 | 1.0078 | 10.0287 | 25.8703 | 10.0287 | 25.8703 |
| 16,384 | Ring-KV | 14.8338 | 1.0078 | 10.0287 | 25.8703 | 10.0287 | 25.8703 |
| 32,768 | Old-main GQA | 14.8338 | 2.0078 | 19.3769 | 36.2184 | 19.3769 | 36.2184 |
| 32,768 | New-main GQA | 14.8338 | 2.0078 | 19.3717 | 36.2133 | 19.3717 | 36.2133 |
| 32,768 | Ring-Q | 14.8338 | 2.0078 | 19.3706 | 36.2122 | 19.3706 | 36.2122 |
| 32,768 | Ring-KV | 14.8338 | 2.0078 | 19.3706 | 36.2122 | 19.3706 | 36.2122 |
| 65,536 | Old-main GQA | 14.8338 | 4.0078 | 38.0673 | 56.9089 | 38.0673 | 56.9089 |
| 65,536 | New-main GQA | 14.8338 | 4.0078 | 38.0556 | 56.8972 | 38.0556 | 56.8972 |
| 65,536 | Ring-Q | 14.8338 | 4.0078 | 38.0545 | 56.8961 | 38.0545 | 56.8961 |
| 65,536 | Ring-KV | 14.8338 | 4.0078 | 38.0545 | 56.8961 | 38.0545 | 56.8961 |

## Validation

All four builds passed the same ONNX attention suite, covering geometry, fallback behavior, 64K prefill, decode regimes, and the 65,664-token cache boundary. Separate full-model diagnostic traces confirm dispatch: each trace exercises 16 full-attention prefill calls and 2,048 decode calls. The existing builds use fused GQA prefill; the Ring builds select their respective Ring prefill implementation. These trace timings are excluded from the tables.

| Implementation | ONNX attention tests | Full-model decode route |
| --- | --- | --- |
| Old-main GQA | 29 passed | legacy (2,048 calls) |
| New-main GQA | 29 passed | flash (2,048 calls) |
| Ring-Q | 29 passed | ring_q (2,048 calls) |
| Ring-KV | 29 passed | ring_kv (2,048 calls) |

All 16 full-model cases were checked against the build manifests: loaded DLL hashes, runtime versions, model revision, and harness hash match the intended configuration. Allocation tracking reported no failures or unknown frees, and each model + KV + other subtotal equals its phase peak. The benchmarked merged code commit also passed all nine GitHub CI checks; build-time compiler/runtime checks and local repository pre-commit checks passed.

## Accuracy

Full-model comparisons below use new-main GQA as reference. Decode logits are compared only when their one-token generated prefixes match. Output-position agreement is reported independently; it is not a semantic quality benchmark. Every measured request reproduced its configuration's diagnostic output exactly. All prefill and aligned first-decode comparisons passed cosine >= 0.999 and relative L2 <= 0.05.

| Tokens | Implementation | Prefill cosine / rel. L2 | First-decode cosine / rel. L2 | Matching output positions / 128 |
| ---: | --- | --- | --- | ---: |
| 4,096 | Old-main GQA | 0.99999724 / 0.002375 | 0.99994766 / 0.010248 | 128 |
| 4,096 | New-main GQA | 1.00000000 / 0.000000 | 1.00000000 / 0.000000 | 128 |
| 4,096 | Ring-Q | 0.99999767 / 0.002168 | 0.99995212 / 0.009786 | 128 |
| 4,096 | Ring-KV | 0.99999767 / 0.002168 | 0.99995212 / 0.009786 | 128 |
| 16,384 | Old-main GQA | 0.99999649 / 0.003601 | 0.99995582 / 0.009401 | 128 |
| 16,384 | New-main GQA | 1.00000000 / 0.000000 | 1.00000000 / 0.000000 | 128 |
| 16,384 | Ring-Q | 0.99999715 / 0.002440 | 0.99995295 / 0.009838 | 128 |
| 16,384 | Ring-KV | 0.99999715 / 0.002440 | 0.99995295 / 0.009838 | 128 |
| 32,768 | Old-main GQA | 0.99999735 / 0.002483 | 0.99995065 / 0.009935 | 128 |
| 32,768 | New-main GQA | 1.00000000 / 0.000000 | 1.00000000 / 0.000000 | 128 |
| 32,768 | Ring-Q | 0.99999706 / 0.002429 | 0.99995230 / 0.010002 | 128 |
| 32,768 | Ring-KV | 0.99999706 / 0.002429 | 0.99995230 / 0.010002 | 128 |
| 65,536 | Old-main GQA | 0.99999704 / 0.002565 | 0.99993599 / 0.011636 | 128 |
| 65,536 | New-main GQA | 1.00000000 / 0.000000 | 1.00000000 / 0.000000 | 128 |
| 65,536 | Ring-Q | 0.99999603 / 0.003591 | 0.99994608 / 0.010550 | 128 |
| 65,536 | Ring-KV | 0.99999603 / 0.003591 | 0.99994608 / 0.010550 | 128 |

## Historical lx01 results (earlier baseline)

These are the original Linux results, not the fresh Windows old-main control. Source `6cda375` based on `1c9a454`, TheRock 7.13, ORT 1.30 and GenAI 0.14. Do not compute a before/after speedup across these two environments.

| Tokens | Old GQA prefill s | Ring-Q prefill s | Ring-KV prefill s | Old GQA decode tok/s | Ring-Q decode tok/s | Ring-KV decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 9.467 | 9.697 | 9.488 | 12.238 | 12.631 | 12.630 |
| 32,768 | 87.723 | 105.253 | 87.023 | 9.615 | 10.778 | 11.021 |
| 65,536 | 221.179 | 280.256 | 221.246 | 7.650 | 9.295 | 9.273 |

Historical memory counters measured DRM VRAM+GTT and excluded mapped-host KV. They are **not comparable** with the complete requested HIP allocation payload above; no historical total or other-memory subtotal is reconstructed. The archived 128K/256K diagnostics also remain historical and are not evidence for updated-main performance or production Ring support beyond its current limits.
