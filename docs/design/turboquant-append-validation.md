<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# TurboQuant append-prefill validation

Multi-token append prefill is implemented and validated on Windows Radeon 8060S (`gfx1151`), using Qwen3.8-27B, ORT 1.30.0, the frozen Windows TurboQuant codec, and an isolated GenAI 0.17 companion build. The [machine-readable receipt](turboquant-append-validation.json) identifies tested source and library hashes. The original single-turn benchmark tables remain unchanged.

**GenAI dependency:** stock GenAI 0.17 rejects repeated `append_tokens()` for AMDGPU before calling HIP-EP. The [companion patch](../../tools/turboquant/patches/onnxruntime-genai-0.17-amdgpu-append.patch) enables AMDGPU continuation and flushes a pending selected token before the next prompt. It applies cleanly to the release sources and was built separately; the stock Python environment was preserved. No ONNX cache-layout/allocation change or rewind change is required.

## Checks

- HIP-EP build: 608 compiler tests passed, 9 unsupported; all 3 runtime checks passed.
- CPU codec reference: 22 tests passed.
- Seven append-kernel cases passed against FP64 attention over reconstructed packed history plus causal FP16 new K/V. Tolerances remain `atol=0.005`, `rtol=0.005`.
- Guarded allocations, short workspace/capacity rejection, partial tiles, 128-query chunk boundaries, maximum finite FP16 queries and unchanged old cache bytes passed.
- Sixteen comparisons of existing dense Ring and TurboQuant decode against the frozen PR binaries were byte-identical.
- Ring-KV append was GPU-tested; the Ring-Q specialization compiled successfully.
- GenAI same-generator tests passed for initial prompts of 1 and 4 tokens, two appended prompts, resumed decode and an unprocessed selected token. Invalid codec rejection still passed. Append-logit error was at most 0.0009765625 in the checked 16 dimensions. The companion C++ capability assertion is included but the upstream C++ suite was not run.

## Direct append attention

Times are diagnostic synchronized wall times over three calls after an initial correctness launch, not full-model throughput. The oracle uses the exact stored indices/norms, so this checks kernel arithmetic separately from quantization error relative to original dense history.

| Cached tokens | New tokens | Max absolute error | Relative L2 | Attention ms | Scratch MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.0009766 | 0.000111 | 0.189 | 0.1176 |
| 15 | 17 | 0.0039062 | 0.000446 | 0.301 | 0.9992 |
| 32 | 33 | 0.0039062 | 0.000552 | 0.319 | 1.9396 |
| 129 | 129 | 0.0039062 | 0.000669 | 1.190 | 7.5234 |
| 4,096 | 257 | 0.0039062 | 0.000894 | 4.484 | 7.5234 |
| 131,072 | 2 | 0.0003662 | 0.001203 | 21.810 | 0.1176 |
| 262,142 | 2 | 0.0006104 | 0.001590 | 40.980 | 0.1176 |

All newly appended bytes matched the frozen GPU encoder. The 4,096-history / 257-new-token case still has one code byte differing from the NumPy encoder near a centroid boundary. This is recorded separately from the append regression; the existing strict assertion in `check_hip.py` remains unchanged.

## Qwen same-generator and dense control

The TQ run used 4,096 initial tokens, eight generated tokens, a 33-token append, eight generated tokens, a 129-token append, and eight generated tokens. All old packed rows in all 16 full-attention layers stayed byte-identical across appends. Logits remained finite. The dense control consumed the same TQ-selected token IDs, so logits are compared on identical histories; this is not a greedy token-agreement claim.

| Past tokens | New prompt | TQ prefill s | Dense Ring prefill s | Logit cosine | Relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4,096 | 10.0920 | 9.6373 | 1.00000000 | 0.000000 |
| 4,104 | 33 | 0.2918 | 0.2831 | 0.99994274 | 0.010918 |
| 4,145 | 129 | 0.6341 | 0.6107 | 0.99994244 | 0.011272 |

All matching-prefix screens passed the existing cosine >= 0.999 / relative L2 <= 0.05 thresholds. Dense Ring's final append was also compared with a fresh prefill of the entire 4,274-token history: cosine **0.99999375**, relative L2 **0.003565**. This checks multi-turn state/position handling independently of KV quantization.

These are single-run diagnostics, including initial-call effects, rather than warmed median benchmarks. Follow-up TQ prefill took about 292 ms and 634 ms; no append speedup over dense Ring is claimed.

## Memory and remaining limits

Persistent full-attention KV payload remains **3.8788x smaller** (512/132 bytes per vector). At the tested capacity of 4,608 tokens across 16 full-attention layers, that is **74.25 MiB** of TQ KV versus **288 MiB** of FP16 KV. Weights, recurrent state, padding and other allocations are separate. These are payload calculations from verified cache shapes, not a new WDDM process-memory measurement.

Append scratch is capped at **7,888,896 bytes (7.5234 MiB)** independently of history length. A runtime can retain a larger workspace allocated during fresh prefill; the cap describes append requirements, not total retained allocation. Old history is unpacked only into small on-chip tiles.

The original strict encoder failures, generated-sequence divergences at some contexts and failed synthetic 256K full-model comparisons remain release blockers. Passing the append kernel at the 256K capacity boundary is not a semantic 256K model-quality qualification. The PR stays draft and still depends on Ring PR #1252. Rewind and prefix-cache sharing are outside this validation.
