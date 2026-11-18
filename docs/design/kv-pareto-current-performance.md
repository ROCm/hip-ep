<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# KV-Pareto current performance

## Kernel performance

Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), 128 GB RAM, Linux, TheRock 7.13, ORT 1.31 and main's pinned LLVM, with rocMLIR disabled (`ENABLE_ROCMLIRTRITON=OFF`). **One attention layer**, B1/H24/G4/D256, FP16 Q/K/V/output. All modes use the same binaries.

Warm-up is performed before timing, with three independent rounds. Each context/mode starts in a fresh process. One untimed prefill and 16 untimed decode calls precede measurement. Each round measures three prefills through 64K or one at 128K/256K, then 128 synchronized ORT decode calls using GPU-resident I/O. Tables report medians of round medians. Compilation, setup and uploads are excluded. Decode includes KV append/encoding for every mode, unlike hip-ep PR #1402's direct-kernel table. Speedup is `FP16 baseline / KV-Pareto`, calculated before rounding. Values above 1 favor KV-Pareto. Values below 1 indicate a regression.

Comparisons use unrounded medians against No Ring / Ring, in that order. All measured contexts from 128 through 256K are shown.

**Prefill — ms, lower is better**

Latency reduction is `100 × (1 − KV-Pareto / baseline)`. Positive percentages favor KV-Pareto, and negative percentages indicate regressions.

| Context | No Ring (FP16 KV) | Ring (FP16 KV) | KV-Pareto (INT8 KV) | Latency reduction vs No Ring / Ring |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.20 | 0.25 | 0.22 | -10.0% / +12.9% |
| 256 | 0.30 | 0.13 | 0.35 | -13.7% / -161.3% |
| 512 | 0.19 | 0.21 | 0.21 | -12.7% / 0.0% |
| 1K | 0.52 | 0.63 | 0.59 | -13.3% / +6.4% |
| 2K | 1.86 | 2.30 | 2.02 | -8.3% / +12.1% |
| 4K | 6.76 | 8.33 | 7.03 | -4.0% / +15.6% |
| 8K | 25.84 | 32.22 | 26.43 | -2.3% / +18.0% |
| 16K | 102.23 | 127.13 | 103.12 | -0.9% / +18.9% |
| 32K | 407.29 | 513.77 | 407.28 | 0.0% / +20.7% |
| 64K | 1669.45 | 2094.38 | 1665.38 | +0.2% / +20.5% |
| 128K | 6803.31 | 8697.23 | 6808.57 | -0.1% / +21.7% |
| 256K | 29585.61 | 36550.75 | 30009.83 | -1.4% / +17.9% |

**Decode — ms per token, lower is better**

Speedup is `baseline / KV-Pareto`. Values above 1 favor KV-Pareto.

| Context | No Ring (FP16 KV) | Ring (FP16 KV) | KV-Pareto (INT8 KV) | Speedup vs No Ring / Ring |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.0509 | 0.0545 | 0.0454 | 1.12× / 1.20× |
| 256 | 0.0535 | 0.0676 | 0.0465 | 1.15× / 1.45× |
| 512 | 0.0574 | 0.0712 | 0.0473 | 1.21× / 1.51× |
| 1K | 0.0690 | 0.0723 | 0.0510 | 1.35× / 1.42× |
| 2K | 0.0884 | 0.0923 | 0.0663 | 1.33× / 1.39× |
| 4K | 0.1086 | 0.1129 | 0.0853 | 1.27× / 1.32× |
| 8K | 0.1450 | 0.1469 | 0.1063 | 1.36× / 1.38× |
| 16K | 0.3654 | 0.3711 | 0.1715 | 2.13× / 2.16× |
| 32K | 0.6663 | 0.6699 | 0.3679 | 1.81× / 1.82× |
| 64K | 1.4395 | 1.2612 | 0.6838 | 2.11× / 1.84× |
| 128K | 2.7729 | 2.4876 | 1.3046 | 2.13× / 1.91× |
| 256K | 5.5214 | 4.8656 | 2.5671 | 2.15× / 1.90× |

At 64K, decode speedup is **2.11× versus No Ring / 1.84× versus Ring**. At 128K and 256K the speedups versus No Ring are **2.13× / 2.15×**. These are attention-layer results, not full-model measurements at those lengths.

## Model performance

`amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, INT4 group-128 weights, batch 1, revision `9b37cf56a001720ba7ce98ff80ea651b02e5931b`. All modes use identical weights and synthetic prompt arrays (seed 1234), the official GenAI 0.19 wheel and ORT 1.31. GPU jobs run serially with two OpenMP/OpenBLAS threads. Mode order rotates by context.

Each context/mode uses **three independent fresh-process rounds**, with one measured request or conversation per round. Single-turn rounds warm the model session with an untimed same-length prefill and four selected tokens. Append rounds first run an untimed eight-turn conversation. Measurement then uses an independent generator/cache. **TTFT runs from prompt append through first-token selection and device synchronization.** Token delivery, tokenization, model load/JIT and generator setup are excluded. The first four selected tokens per turn are excluded from decode timing. The remaining 124 timed model steps complete 128 output tokens and include KV updates. Timing disables debug/performance tracing and accuracy/cache/logits readbacks.

Each cell is **median TTFT seconds / median decode tokens per second** across the three rounds. Per-round TPS is `1,000 / mean decode ms/token`. TPS gains are **vs No Ring / Ring** in that order, using unrounded timings. **Positive percentages mean improvement. Negative percentages mean regression.** All measured contexts and turns are shown.

**Single-turn model, 128–64K.** Each context/mode starts in a fresh process/model with an independent measured generator/cache per round. Capacity is prompt length + 256.

| Context | No Ring (FP16 KV) | Ring (FP16 KV) | KV-Pareto (INT8 KV) | TPS gain vs No Ring / Ring |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.29 / 15.042 | 0.29 / 15.020 | 0.29 / 15.074 | +0.2% / +0.4% |
| 256 | 0.48 / 15.038 | 0.48 / 15.006 | 0.48 / 15.006 | -0.2% / 0.0% |
| 512 | 0.93 / 15.017 | 0.94 / 14.968 | 0.94 / 15.062 | +0.3% / +0.6% |
| 1K | 2.09 / 14.961 | 2.07 / 14.933 | 2.05 / 15.026 | +0.4% / +0.6% |
| 2K | 4.26 / 14.886 | 4.25 / 14.866 | 4.33 / 15.001 | +0.8% / +0.9% |
| 4K | 8.41 / 14.746 | 8.50 / 14.736 | 8.40 / 14.926 | +1.2% / +1.3% |
| 8K | 17.76 / 14.479 | 17.90 / 14.461 | 17.91 / 14.778 | +2.1% / +2.2% |
| 16K | 38.81 / 14.007 | 39.07 / 13.967 | 38.74 / 14.515 | +3.6% / +3.9% |
| 32K | 81.72 / 13.114 | 84.54 / 13.103 | 82.22 / 13.999 | +6.7% / +6.8% |
| 64K | 189.00 / 11.232 | 198.65 / 11.630 | 187.46 / 13.061 | +16.3% / +12.3% |

At 64K, model decode speedup is **1.16× (+16.3%) versus No Ring / 1.12× (+12.3%) versus Ring**.

**Eight-turn append model.** Three independent conversations per mode, each retaining one generator/cache. Every turn appends 4,096 prompt tokens and generates 128 tokens, with early EOS suppressed. TTFT includes pending-token flush. Capacity is 33,920. The final prompt ends at 33,664 tokens.

| Turn / context | No Ring (FP16 KV) | Ring (FP16 KV) | KV-Pareto (INT8 KV) | TPS gain vs No Ring / Ring |
| --- | ---: | ---: | ---: | ---: |
| 1 / 4,096 | 8.89 / 14.766 | 8.89 / 14.725 | 8.93 / 14.923 | +1.1% / +1.3% |
| 2 / 8,320 | 9.21 / 14.497 | 9.23 / 14.444 | 9.25 / 14.781 | +2.0% / +2.3% |
| 3 / 12,544 | 9.46 / 14.243 | 9.61 / 14.197 | 9.52 / 14.641 | +2.8% / +3.1% |
| 4 / 16,768 | 9.71 / 13.982 | 9.92 / 13.945 | 9.73 / 14.498 | +3.7% / +4.0% |
| 5 / 20,992 | 9.99 / 13.744 | 10.28 / 13.720 | 10.11 / 14.359 | +4.5% / +4.7% |
| 6 / 25,216 | 10.25 / 13.521 | 10.65 / 13.480 | 10.26 / 14.238 | +5.3% / +5.6% |
| 7 / 29,440 | 10.53 / 13.288 | 10.97 / 13.263 | 10.53 / 14.101 | +6.1% / +6.3% |
| 8 / 33,664 | 10.81 / 13.065 | 11.35 / 13.054 | 10.80 / 13.977 | +7.0% / +7.1% |

## Memory breakdown

**Memory.** KV payload is **1.83× smaller**, including scales and zero points. Reserved KV below is calculated at the configured capacity. Peak GTT is whole-device Linux GPU-mapped system memory sampled every 0.5 seconds, taking the largest measured peak across three rounds. Warm-up memory samples are excluded from the request peaks. It includes weights, workspace and other processes. It is not total model RAM or hip-ep PR #1402's Windows HIP allocation counter. Sampling can miss brief peaks.

| Case | Reserved KV: FP16 / KV-Pareto GiB | No Ring GTT GiB | Ring GTT GiB | KV-Pareto GTT GiB |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.023 / 0.013 | 16.002 | 16.002 | 15.992 |
| 256 | 0.031 / 0.017 | 16.057 | 16.057 | 16.045 |
| 512 | 0.047 / 0.026 | 16.213 | 16.213 | 16.193 |
| 1K | 0.078 / 0.043 | 16.609 | 16.609 | 16.529 |
| 2K | 0.141 / 0.077 | 17.186 | 17.186 | 17.123 |
| 4K | 0.266 / 0.145 | 18.318 | 18.318 | 18.193 |
| 8K | 0.516 / 0.282 | 20.574 | 20.574 | 20.324 |
| 16K | 1.016 / 0.555 | 25.270 | 25.270 | 24.832 |
| 32K | 2.016 / 1.102 | 34.330 | 34.330 | 33.518 |
| 64K | 4.016 / 2.196 | 52.449 | 52.449 | 50.887 |
| Turn 1 / 4,096 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |
| Turn 2 / 8,320 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |
| Turn 3 / 12,544 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |
| Turn 4 / 16,768 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |
| Turn 5 / 20,992 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |
| Turn 6 / 25,216 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |
| Turn 7 / 29,440 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |
| Turn 8 / 33,664 | 2.070 / 1.132 | 20.131 | 20.131 | 19.355 |

**Numerical correctness.** Cache bytes match the reference quantizer exactly. Attention is checked against FP64 using the same dequantized history and direct FP16 new K/V. Maximum relative L2 across the 108 attention legs was **3.50e-04 decode / 8.62e-04 prefill**. H16/H24 GenAI checks had maximum absolute logits error below **5.8e-4**. Synthetic timing prompts and generated-token agreement do not establish general answer quality.

## Test plan

Measured package: `cd6d2871`, based on main `fc443726`. The branch now merges main `c56497b2` and updates the kernel-test interface. Production GQA and KV-Pareto kernels are unchanged from the measured build. These interface checks use the preserved package and do not replace the performance campaign.

- **Build:** fresh compiler/runtime/gfx1151 package, all eight Ring/TurboQuant/KV-Pareto runtime syntax combinations and repository hooks passed.
- **Kernel:** 27 checks passed for exact encoding, contracts/rejections, H16/H24 prefill/append and decode numerics.
- **UT interface:** all manual pre-commit hooks passed. The updated driver passed 27 tier-1 checks, custom decode/append examples and online prefill with shared timing. Seven invalid custom inputs were rejected.
- **Operator:** KV-Pareto enabled **9 passed / 1 intentional skip**, also with Ring enabled. Default **1 passed / 9 intentional skips**. Covers shared GPU caches, alternating append/decode, exact payload/scale/zero-point bytes and invalid width/options/alias requests.
- **Shared GQA:** **12 Ring geometry/fallback/shared-cache, 2 host-length and 4 dense key-bias prefill tests passed** with both runtime flags enabled. Includes FP16/FP32 layer lengths and full/sliding attention with left/right padding.
- **Compiler/runtime:** 3 packed-width/dtype/host-length LIT cases and 4 CTests passed: OutputAllocatorUnitTest, StaticPlugins, RuntimeLoweringContractsUnit and LdsBarrierArchGate.
- **Full LIT:** **745 passed, 9 unsupported, 5 failed**. Four names match prior host failures. `Dialect/hip-fuse-rocmlir-qlinear-epilogue.mlir` is newly observed. A separate pure-main build was not run.
- **GenAI 0.19:** H16/H24 allocation, exact bytes, repeated append, pending-token flush, old-row preservation and error rejection passed. Full Qwen packed-cache smoke passed.
- **Performance:** **108 attention legs and 162 request-turn records across 99 model processes** completed. Prompt hashes match across modes. Minimum streaming health was 218.5 GB/s for attention and 221.6 GB/s for models (130 GB/s threshold).

**CI.** Adds `gqa/kv_pareto/Xfp16_Wi8_Yfp16` to the Strix Halo kernel-test job. Tier 1 checks exact bytes and attention numerics. Optional tiers 2/3, including single-call 256K prefill, were not rerun. Windows CI is pending. This Linux/TheRock 7.13 campaign differs from hip-ep PR #1402's Windows/HIP 7.15 environment.

```bash
HIPDNN_EP_USE_KV_PARETO=1 python -m pytest \
  test/numeric/tests/test_gqa_kv_pareto.py \
  --ep-name hipgpu --ep-dll /path/to/libhipgpu.so
```

Packaged Windows CI: `hipdnn-kernel-ut.exe --mode lookup --coverage 1`.


## Evidence and limits

Unsupported: other GPUs, batch > 1, internal RoPE, packed QKV, attention/key bias, sinks, sliding windows, softcap, smooth softmax, QK output and beam-cache reordering. Model results cover synthetic prompts through 64K and eight turns through 33,664 tokens. **256K results cover one attention layer**. Three independent rounds characterize this machine/run.

[Performance report](https://github.com/ROCm/hip-ep/blob/users/rohrajes/kv-pareto/docs/design/kv-pareto-current-performance.md), [attention receipts](https://github.com/ROCm/hip-ep/blob/users/rohrajes/kv-pareto/docs/design/kv-pareto-kernel-performance.json) and [model receipts](https://github.com/ROCm/hip-ep/blob/users/rohrajes/kv-pareto/docs/design/kv-pareto-model-performance.json) retain raw samples, provenance, prompts and memory scope.

Claude Opus 5.5 assisted implementation, tests and documentation. Codex assisted main synchronization, GenAI integration and the fresh performance campaign.
