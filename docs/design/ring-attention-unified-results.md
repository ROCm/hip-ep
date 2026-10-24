<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Unified Ring measurements

Add opt-in FP16 Ring attention through the existing ONNX GroupQueryAttention and KV-cache interfaces; existing GQA remains the default.

Ring now combines KV-sharing prefill with partitioned FP32 decode under one `HIP_EP_ENABLE_RING_ATTENTION=ON/OFF` option (default OFF). Enable it in a fresh build with `--cmake_extra_defines HIP_EP_ENABLE_RING_ATTENTION=ON`; the former Ring-Q/Ring-KV selector is replaced.

## Kernel performance

Windows Radeon 8060S (`gfx1151`), B=1, H=24, KV heads=4, D=256, FP16 Q/K/V/output. Main: `d5979f76`. Positive percentages favor Ring.

**Prefill — lower latency is better**

| Context | Main ms | New Ring ms | Latency reduction |
| --- | --- | --- | --- |
| 4K | 12.130 | 12.501 | -3.1% |
| 16K | 194.501 | 180.161 | +7.4% |
| 32K | 767.231 | 726.184 | +5.4% |
| 64K | 3,036.175 | 2,980.941 | +1.8% |
| 128K | 12,134.084 | 11,975.086 | +1.3% |
| 256K | 64,937.127 | 49,763.236 | +23.4% |

**Decode — lower latency is better; includes scan and merge**

| Context | Main ms | New Ring ms | Latency reduction |
| --- | --- | --- | --- |
| 4K | 0.0999 | 0.0877 | +12.3% |
| 16K | 0.3438 | 0.3402 | +1.1% |
| 32K | 0.6638 | 0.6742 | -1.6% |
| 64K | 1.6430 | 1.3253 | +19.3% |
| 128K | 3.3518 | 2.6104 | +22.1% |
| 256K | 6.6795 | 5.1396 | +23.1% |

Prefill: three-round medians. Decode: seven-round warm-cache graph medians; with 64 MiB cache displacement, Ring's reductions at 64K/128K/256K are **18.4% / 20.6% / 23.8%**. At 4K the displaced-cache difference is 0.2%.

## Model performance

`amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, INT4 group-128 weights, FP16 KV, batch 1, unchunked prompts and 128 greedy output tokens. Current results average **two timed requests per case**, with no further warmups after the initial 4K case. Previous main (`a4f9826b`) uses saved three-run medians; current main is `d5979f76`.

**Time to first token — lower is better**

| Context | Previous main TTFT s | Current main TTFT s | New Ring TTFT s | TTFT reduction |
| --- | --- | --- | --- | --- |
| 4K | 9.468 | 9.436 | 9.629 | -2.05% |
| 16K | 40.464 | 40.605 | 40.457 | +0.36% |
| 32K | 104.296 | 87.382 | 87.334 | +0.05% |
| 64K | 238.819 | 200.847 | 202.244 | -0.70% |

**Decode throughput — higher is better; 127 tokens after first-token selection**

| Context | Previous main Decode tok/s | Current main Decode tok/s | New Ring Decode tok/s | Throughput gain |
| --- | --- | --- | --- | --- |
| 4K | 12.339 | 13.052 | 12.909 | -1.10% |
| 16K | 11.327 | 11.873 | 12.137 | +2.22% |
| 32K | 9.749 | 11.334 | 11.434 | +0.89% |
| 64K | 8.161 | 9.779 | 10.213 | +4.44% |

## Why Ring is useful

- **Multi-accelerator:** mergeable block summaries provide a path to future distributed attention.
- **KV locality:** tiled traversal provides a foundation for future paged-cache access.
- **Grouping:** six query heads reuse each staged KV tile.
- **Concurrency:** independent query tiles and history partitions expose parallel GPU work.
- **Long context:** 19–23% lower decode kernel latency at 64K–256K; 4.4% higher model TPS at 64K.

## Historical improvement

The initial **October 17** version of this PR already beat then-main on long-context model decode: Ring-KV delivered **+14.6% at 32K and +21.2% at 64K** in the original lx01 runs. Main subsequently caught up in measured performance: the [Windows refresh including #1259](https://github.com/ROCm/hip-ep/pull/1252#issuecomment-6304595831) improved main's 64K decode by **17.2%** over old main. The new Ring model result is **25.1% above previous main and 4.4% above current main at 64K**. Historical Linux and current Windows campaigns remain separate comparisons.

<details>
<summary>Memory, validation and measurement details</summary>

**Total = model parameters + KV cache + other.** Peaks count live requested HIP device plus mapped-host payload, including retained pools; they exclude driver allocations and CPU interpreter memory. Prefill/decode peaks round to the same values here.

| Context | Model GiB | KV GiB | Main other GiB | Ring other GiB | Main peak GiB | Ring peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | 14.8338 | 0.2578 | 3.0234 | 3.0224 | 18.1149 | 18.1140 |
| 16K | 14.8338 | 1.0078 | 10.0298 | 10.0294 | 25.8714 | 25.8710 |
| 32K | 14.8338 | 2.0078 | 19.3717 | 19.3714 | 36.2133 | 36.2129 |
| 64K | 14.8338 | 4.0078 | 38.0556 | 38.0552 | 56.8972 | 56.8968 |

Other includes activations, recurrent state, workspace and retained pools. KV storage is unchanged. Ring requests no explicit global prefill scratch and up to 2.27 MiB decode scratch through 256K.

Validation: **51 Ring ONNX cases**, **47 shared default-build cases**, and **28 targeted native checks** passed, including underflow and partition/context boundaries. Benchmark outputs passed FP32-reference checks through 256K; long prefill checks sample rows across all valid keys. Compiler/runtime checks and repository pre-commit hooks passed. Full-model timing covers 4K–64K; multi-accelerator execution, paged-KV integration and multi-request concurrency are future work.

Model timings use free greedy generation; exact token agreement is not required. A separate 128-token-prompt diagnostic found same-prefix logit cosine 0.9986 / relative L2 5.46% at decode step 41, with the same argmax token; this is a numerical diagnostic, not a model-quality benchmark. No additional accuracy probes were added to the two-request timing campaign.

Main 4K reuses its first two already-completed measurements; its third measurement is excluded. Other cases use their first two timed requests without per-case warmup; model load/compilation is excluded. The 256K kernel-prefill samples ranged 61.1–82.6 s for main and 49.7–50.4 s for Ring. Model runtime: HIP 7.15, ORT 1.31.0, GenAI 0.17.0; AC power and serial GPU execution. Model revision: `9b37cf56a001720ba7ce98ff80ea651b02e5931b`.

Design and enablement: [Ring attention](https://github.com/amdrajeevp1/hip-ep/blob/rajeevp/ring-attn/docs/design/ring-attention.md). Current samples and provenance: [unified results](https://github.com/amdrajeevp1/hip-ep/blob/rajeevp/ring-attn/docs/design/ring-attention-unified-results.md). Historical measurements: [original validation](https://github.com/ROCm/hip-ep/blob/ef91813063ded06acb6f41218d354177f8e70b02/docs/design/ring-attention-validation.md) and [previous Windows comparison](https://github.com/ROCm/hip-ep/pull/1252#issuecomment-6304595831).

OpenAI Codex assisted implementation, experiments, tests and documentation. METIS exports are authorized for MIT contribution; source attribution is retained.

</details>

## Raw samples and provenance

[Measurement data](ring-attention-unified-results.json) contains the selected model samples, kernel timing samples, allocation accounting, input hashes and binary hashes. Kernel sources and binaries are unchanged between these kernel and model measurements.
