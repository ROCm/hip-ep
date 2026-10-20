<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen four-way attention comparison

These are historical measurements of the frozen installations identified
below and in the accompanying JSON, before the synchronization with upstream
`65d1f5a`. That upstream update includes a new HpG=6 baseline decoder and an
ORT 1.31 default. The tables have not been rerun against those changes and
must not be read as performance comparisons against current `main`.

[Machine-readable metrics and validation provenance](turboquant-validation.json).

Windows 11 Enterprise 10.0.26200; Ryzen AI MAX+ PRO 395 / Radeon 8060S gfx1151; 128 GiB installed UMA, 63.776 GiB OS-managed after firmware reservation; driver 32.0.22050.4002. ROCm HIP compiler 7.15.26333, ORT 1.30.0, GenAI 0.17.0. Qwen3.8-27B has INT4 group-128 weights, FP16 activations, 16 full-attention layers and 48 linear/recurrent layers. Batch one; CPU decoder fallback disabled.

Baseline is the production `existing` GQA selection on upstream a4f9826 plus the Ring/TurboQuant integration branch, with quantization disabled. Ring-Q and Ring-KV select their respective prefill schedules and share the same dense decoder. TurboQuant uses Ring-KV FP16 prefill, then the MSE-only 4-bit QR cache and natural-exponential compressed decoder. The installed binaries are separately identified in the accompanying metrics JSON.

These Windows measurements share one model revision and harness. Historical lx01/Linux Ring PR figures are not mixed into these tables: OS, GenAI version, UMA reservation and memory-accounting scope differ.

## Prefill and decode

Medians exclude the diagnostic/warmup request. Model/generator construction, cache allocation and prompt construction are outside the timers; TTFT starts at `append_tokens`. Full-prefill prompts use random token IDs from 100 through 19,999 with NumPy seed 20261014. Three measured requests per full-prefill case through 64K; one at 128K. Generate 128 tokens with a fresh cache/state per request; decode times the 127 steps after first-token selection. At 256K, skip prefill, initialize identical seeded random KV/recurrent history, warm up eight steps, and measure 128 full-model decode steps including logits readback and greedy selection. Sequential blocks are not an alternating paired speedup experiment.

| Prompt/context | Variant | Prefill s | TTFT s | Decode ms/token | Decode tok/s | Measured requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | Baseline | 0.345 | 0.345 | 81.652 | 12.247 | 3 |
| 128 | Ring-Q | 0.338 | 0.338 | 80.101 | 12.484 | 3 |
| 128 | Ring-KV | 0.332 | 0.333 | 77.964 | 12.826 | 3 |
| 128 | TurboQuant | 0.337 | 0.337 | 78.692 | 12.708 | 3 |
| 256 | Baseline | 0.627 | 0.628 | 80.061 | 12.490 | 3 |
| 256 | Ring-Q | 0.630 | 0.631 | 79.785 | 12.534 | 3 |
| 256 | Ring-KV | 0.618 | 0.619 | 78.783 | 12.693 | 3 |
| 256 | TurboQuant | 0.637 | 0.638 | 79.263 | 12.616 | 3 |
| 512 | Baseline | 1.247 | 1.247 | 79.825 | 12.527 | 3 |
| 512 | Ring-Q | 1.236 | 1.237 | 79.642 | 12.556 | 3 |
| 512 | Ring-KV | 1.249 | 1.250 | 78.968 | 12.663 | 3 |
| 512 | TurboQuant | 1.245 | 1.246 | 78.778 | 12.694 | 3 |
| 1K | Baseline | 2.409 | 2.409 | 80.229 | 12.464 | 3 |
| 1K | Ring-Q | 2.416 | 2.416 | 79.307 | 12.609 | 3 |
| 1K | Ring-KV | 2.405 | 2.406 | 79.238 | 12.620 | 3 |
| 1K | TurboQuant | 2.421 | 2.421 | 79.692 | 12.548 | 3 |
| 2K | Baseline | 4.661 | 4.661 | 80.855 | 12.368 | 3 |
| 2K | Ring-Q | 4.662 | 4.662 | 79.571 | 12.567 | 3 |
| 2K | Ring-KV | 4.691 | 4.691 | 80.531 | 12.418 | 3 |
| 2K | TurboQuant | 4.725 | 4.726 | 79.884 | 12.518 | 3 |
| 4K | Baseline | 9.403 | 9.403 | 82.325 | 12.147 | 3 |
| 4K | Ring-Q | 9.374 | 9.374 | 81.003 | 12.345 | 3 |
| 4K | Ring-KV | 9.438 | 9.438 | 80.609 | 12.406 | 3 |
| 4K | TurboQuant | 9.427 | 9.427 | 80.584 | 12.409 | 3 |
| 8K | Baseline | 19.241 | 19.241 | 84.389 | 11.850 | 3 |
| 8K | Ring-Q | 19.187 | 19.187 | 82.450 | 12.129 | 3 |
| 8K | Ring-KV | 19.154 | 19.154 | 82.904 | 12.062 | 3 |
| 8K | TurboQuant | 19.295 | 19.296 | 82.014 | 12.193 | 3 |
| 16K | Baseline | 40.094 | 40.095 | 89.589 | 11.162 | 3 |
| 16K | Ring-Q | 39.907 | 39.908 | 86.310 | 11.586 | 3 |
| 16K | Ring-KV | 39.637 | 39.637 | 86.554 | 11.554 | 3 |
| 16K | TurboQuant | 39.838 | 39.838 | 84.215 | 11.874 | 3 |
| 32K | Baseline | 86.604 | 86.605 | 99.077 | 10.093 | 3 |
| 32K | Ring-Q | 85.966 | 85.966 | 93.153 | 10.735 | 3 |
| 32K | Ring-KV | 85.126 | 85.127 | 92.930 | 10.761 | 3 |
| 32K | TurboQuant | 85.682 | 85.683 | 88.077 | 11.354 | 3 |
| 64K | Baseline | 198.325 | 198.326 | 120.163 | 8.322 | 3 |
| 64K | Ring-Q | 196.459 | 196.459 | 106.794 | 9.364 | 3 |
| 64K | Ring-KV | 195.117 | 195.118 | 107.342 | 9.316 | 3 |
| 64K | TurboQuant | 191.487 | 191.488 | 97.487 | 10.258 | 3 |
| 128K | Baseline | 583.755 | 583.755 | 164.881 | 6.065 | 1 |
| 128K | Ring-Q | 552.265 | 552.266 | 137.608 | 7.267 | 1 |
| 128K | Ring-KV | 550.367 | 550.367 | 141.745 | 7.055 | 1 |
| 128K | TurboQuant | 540.340 | 540.340 | 120.639 | 8.289 | 1 |
| 256K* | Baseline | — | — | 228.578 | 4.375 | 1 |
| 256K* | Ring-Q | — | — | 197.804 | 5.055 | 1 |
| 256K* | Ring-KV | — | — | 196.676 | 5.085 | 1 |
| 256K* | TurboQuant | — | — | 154.805 | 6.460 | 1 |

*256K is a synthetic performance experiment with no prefill or TTFT. The recorded TurboQuant-versus-Ring-KV first-step numerical screen fails for this workload; see the accuracy section. It is not a semantic long-context accuracy result.

## Memory decomposition

GiB. Phase peaks are per-process Windows GPU Total Committed sampled at a target 20 ms interval, not physical residency or whole-system RAM. The prefill memory phase includes first-token selection; decode covers the remaining 127 steps. Weights are the two verified constant allocations. KV is the verified allocation of 32 mapped-host cache buffers, including allocator padding, counted once. Other = phase peak − weights − KV and includes activations, recurrent state, scratch, allocator pools and runtime/driver allocations. Decode retains prefill allocations. The metrics JSON separately records exact EP allocation-event peaks.

| Prompt/context | Variant | Weights | KV | Other prefill | Prefill peak | Other decode | Decode peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | Baseline | 14.8338 | 0.0156 | 1.3902 | 16.2396 | 1.3902 | 16.2396 |
| 128 | Ring-Q | 14.8338 | 0.0156 | 1.3902 | 16.2396 | 1.3902 | 16.2396 |
| 128 | Ring-KV | 14.8338 | 0.0156 | 1.3902 | 16.2396 | 1.3902 | 16.2396 |
| 128 | TurboQuant | 14.8338 | 0.0049 | 1.4010 | 16.2396 | 1.4010 | 16.2396 |
| 256 | Baseline | 14.8338 | 0.0234 | 1.5424 | 16.3996 | 1.5424 | 16.3996 |
| 256 | Ring-Q | 14.8338 | 0.0234 | 1.5336 | 16.3908 | 1.5336 | 16.3908 |
| 256 | Ring-KV | 14.8338 | 0.0234 | 1.5336 | 16.3908 | 1.5336 | 16.3908 |
| 256 | TurboQuant | 14.8338 | 0.0068 | 1.4875 | 16.3281 | 1.4875 | 16.3281 |
| 512 | Baseline | 14.8338 | 0.0391 | 1.7136 | 16.5864 | 1.7136 | 16.5864 |
| 512 | Ring-Q | 14.8338 | 0.0391 | 1.7048 | 16.5776 | 1.7048 | 16.5776 |
| 512 | Ring-KV | 14.8338 | 0.0391 | 1.6421 | 16.5149 | 1.6421 | 16.5149 |
| 512 | TurboQuant | 14.8338 | 0.0117 | 1.6694 | 16.5149 | 1.6694 | 16.5149 |
| 1K | Baseline | 14.8338 | 0.0703 | 2.0396 | 16.9437 | 2.0396 | 16.9437 |
| 1K | Ring-Q | 14.8338 | 0.0703 | 1.9681 | 16.8722 | 1.9681 | 16.8722 |
| 1K | Ring-KV | 14.8338 | 0.0703 | 1.9681 | 16.8722 | 1.9681 | 16.8722 |
| 1K | TurboQuant | 14.8338 | 0.0195 | 2.0228 | 16.8761 | 2.0228 | 16.8761 |
| 2K | Baseline | 14.8338 | 0.1328 | 2.5711 | 17.5376 | 2.5711 | 17.5376 |
| 2K | Ring-Q | 14.8338 | 0.1328 | 2.4996 | 17.4661 | 2.4996 | 17.4661 |
| 2K | Ring-KV | 14.8338 | 0.1328 | 2.4996 | 17.4661 | 2.4996 | 17.4661 |
| 2K | TurboQuant | 14.8338 | 0.0352 | 2.5972 | 17.4661 | 2.5972 | 17.4661 |
| 4K | Baseline | 14.8338 | 0.2578 | 3.7437 | 18.8353 | 3.7437 | 18.8353 |
| 4K | Ring-Q | 14.8338 | 0.2578 | 3.7350 | 18.8265 | 3.7350 | 18.8265 |
| 4K | Ring-KV | 14.8338 | 0.2578 | 3.6722 | 18.7638 | 3.6722 | 18.7638 |
| 4K | TurboQuant | 14.8338 | 0.0703 | 3.8048 | 18.7089 | 3.8048 | 18.7089 |
| 8K | Baseline | 14.8338 | 0.5078 | 6.0890 | 21.4306 | 6.0890 | 21.4306 |
| 8K | Ring-Q | 14.8338 | 0.5078 | 6.0175 | 21.3591 | 6.0175 | 21.3591 |
| 8K | Ring-KV | 14.8338 | 0.5078 | 6.0175 | 21.3591 | 6.0175 | 21.3591 |
| 8K | TurboQuant | 14.8338 | 0.1328 | 6.1102 | 21.0767 | 6.1102 | 21.0767 |
| 16K | Baseline | 14.8338 | 1.0078 | 10.7895 | 26.6311 | 10.7895 | 26.6311 |
| 16K | Ring-Q | 14.8338 | 1.0078 | 10.7180 | 26.5596 | 10.7180 | 26.5596 |
| 16K | Ring-KV | 14.8338 | 1.0078 | 10.7180 | 26.5596 | 10.7180 | 26.5596 |
| 16K | TurboQuant | 14.8338 | 0.2656 | 10.8406 | 25.9400 | 10.8406 | 25.9400 |
| 32K | Baseline | 14.8338 | 2.0078 | 20.1820 | 37.0236 | 20.1820 | 37.0236 |
| 32K | Ring-Q | 14.8338 | 2.0078 | 20.0994 | 36.9410 | 20.0994 | 36.9410 |
| 32K | Ring-KV | 14.8338 | 2.0078 | 20.0994 | 36.9410 | 20.0994 | 36.9410 |
| 32K | TurboQuant | 14.8338 | 0.5176 | 20.2819 | 35.6333 | 20.2819 | 35.6333 |
| 64K | Baseline | 14.8338 | 4.0078 | 38.9558 | 57.7974 | 38.9558 | 57.7974 |
| 64K | Ring-Q | 14.8338 | 4.0078 | 38.8621 | 57.7037 | 38.8621 | 57.7037 |
| 64K | Ring-KV | 14.8338 | 4.0078 | 38.8621 | 57.7037 | 38.8621 | 57.7037 |
| 64K | TurboQuant | 14.8338 | 1.0333 | 39.1643 | 55.0313 | 39.1643 | 55.0313 |
| 128K | Baseline | 14.8338 | 8.0078 | 76.5660 | 99.4075 | 76.5660 | 99.4075 |
| 128K | Ring-Q | 14.8338 | 8.0078 | 76.4504 | 99.2919 | 76.4504 | 99.2919 |
| 128K | Ring-KV | 14.8338 | 8.0078 | 76.4504 | 99.2919 | 76.4504 | 99.2919 |
| 128K | TurboQuant | 14.8338 | 2.0645 | 76.9919 | 93.8902 | 76.9919 | 93.8902 |
| 256K* | Baseline | 14.8338 | 16.0000 | — | — | 1.0834 | 31.9172 |
| 256K* | Ring-Q | 14.8338 | 16.0000 | — | — | 0.9880 | 31.8218 |
| 256K* | Ring-KV | 14.8338 | 16.0000 | — | — | 0.9880 | 31.8218 |
| 256K* | TurboQuant | 14.8338 | 4.1250 | — | — | 1.0743 | 20.0330 |

Full-request cache capacity is prompt length + 128; synthetic capacity is 262,144. TurboQuant stores 128 code bytes plus one FP32 norm per 256-wide vector: 512/132 = 3.8788× payload reduction. Actual reserved-buffer savings vary with size-class padding at shorter contexts. The 256K total is smaller because the prefill activation pools were never allocated.

## Accuracy and incomplete measurements

Performance completion is separate from accuracy. The matching-prefix numerical screen is cosine ≥ 0.999 and relative L2 ≤ 0.05. Generated token agreement is evaluated independently; tolerance-passing logit differences can change greedy sequences. This single seeded prompt per length is not broad task-accuracy or perplexity qualification. Exact per-variant comparisons against the current Windows baseline are retained in the metrics JSON.

- Ring-Q 256: 127/128 output positions match baseline; first divergence at generated token 10.
- Ring-KV 256: 127/128 output positions match baseline; first divergence at generated token 10.
- TurboQuant 256: 3/128 output positions match baseline; first divergence at generated token 3.
- TurboQuant 512: 1/128 output positions match baseline; first divergence at generated token 2.
- Ring-Q 1K: 3/128 output positions match baseline; first divergence at generated token 4.
- Ring-KV 1K: 3/128 output positions match baseline; first divergence at generated token 4.
- Ring-Q 2K: 3/128 output positions match baseline; first divergence at generated token 4.
- Ring-KV 2K: 3/128 output positions match baseline; first divergence at generated token 4.
- TurboQuant 2K: 3/128 output positions match baseline; first divergence at generated token 4.
- Ring-Q 16K: 19/128 output positions match baseline; first divergence at generated token 2.
- Ring-KV 16K: 19/128 output positions match baseline; first divergence at generated token 2.
- Ring-Q 128K: 64/128 output positions match baseline; first divergence at generated token 2.
- Ring-KV 128K: 64/128 output positions match baseline; first divergence at generated token 2.
- TurboQuant 128K: 64/128 output positions match baseline; first divergence at generated token 2.
- Ring-Q 256K first-decode numerical gate fails against baseline: cosine 0.892228, relative L2 0.465593.
- Ring-KV 256K first-decode numerical gate fails against baseline: cosine 0.892228, relative L2 0.465593.
- TurboQuant 256K first-decode numerical gate fails against baseline: cosine 0.902733, relative L2 0.440642.

The earlier TurboQuant-vs-Ring-KV checks also observed sequence divergence at 256, 512, 1K and 16K despite passing first-step logits, and a failed 256K synthetic numerical gate. The original 128K TurboQuant diagnostic used archived lx01 Ring-KV logits after the earlier Windows Ring-KV attempt stopped. This consolidation additionally compares against current Windows baseline and Ring-KV controls wherever they completed. Strict encoder parity still has two code-byte boundary differences at 4K, and Windows codec constants differ from the frozen lx01 export. These findings require accuracy qualification and keep this contribution in draft.

The Windows synthetic source history is byte-identical across compared variants, but differs from the archived Linux source despite the same seed/distribution. No 256K prefill or semantic 256K accuracy claim is made.

Depends on [Ring attention PR #1252](https://github.com/ROCm/hip-ep/pull/1252). Merge that PR first, then rebase this contribution onto main. The numerical limitations above must be resolved or explicitly scoped before release.

## Build and test validation

- CPU reference: 22 tests passed.
- Windows full build: baseline 606 compiler tests passed / 11 unsupported; Ring-Q and the frozen Ring-KV/TurboQuant build 608 passed / 9 unsupported. All three GPU-independent runtime checks passed for each build.
- Parent Ring regression coverage on Windows: 29 ONNX numeric tests and 68 direct-kernel cases passed before TurboQuant model measurements.
- Final packaged TurboQuant scripts: eight extended direct-kernel cases and three long-history cases (131,072 / 131,200 / 262,144) passed against their guarded FP64 oracle. GenAI shared-cache lifecycle and invalid-codec error propagation passed.
- Strict encoder parity: first 14 cases pass; the 4096-vector case still fails with two code-byte differences. The gate is unchanged and the failure is retained.
- Full-model comparison: 48 completed rows and 0 explicitly failed/incomplete rows across four variants and 12 context lengths. Timing completion and accuracy results are reported separately.
- Required `pre-commit run --all-files`: passed. Formatting left all non-comment compiler/runtime/HIP implementation tokens unchanged; the published profiling shim was compiled separately.
