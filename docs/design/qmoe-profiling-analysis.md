<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# QMoE Profiling Analysis — GPT-OSS-120B on gfx1151

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | GPT-OSS-120B (w-uint4-pergroup-asym-awq, fp16 activations) |
| GPU | gfx1151 (RDNA 3.5) |
| TheRock SDK | 7.11.0 |
| Chunk size | 512 tokens |
| Prompt | 2048 tokens (4 × 512 chunks) |
| Generation | 128 tokens |
| Profiling | `HIPDNN_EP_PERF=1` |

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden size | 2880 |
| Intermediate size | 2880 |
| Experts | 128 per layer |
| Top-k | 2 |
| Query heads | 64, head dim 64 (total attention dim = 4096) |
| KV heads | 8 (GQA ratio = 8) |
| Vocab | 201,088 |
| Weight quantization | 4-bit (MatMulNBits) for attention projections and expert FFNs |
| Router | fp16 GEMM (not quantized) |

## Per-Layer Operator Flow

Each of the 36 transformer layers executes the following sequence:

```
Input
  │
  ├─► [1] skip_layernorm (pre-attention)
  │     512×2880
  │
  ├─► [2] matmul_nbits — QKV projection
  │     m=512, n=5120, k=2880
  │     (Q=4096 + K=512 + V=512 = 5120, packed)
  │
  ├─► [3] gqa — Grouped Query Attention
  │     batch=1, seq=512, kv_len=16384, heads=64, dim=64
  │
  ├─► [4] matmul_nbits — Output projection
  │     m=512, n=2880, k=4096
  │
  ├─► [5] elementwise — Residual add
  │     512×2880
  │
  ├─► [6] skip_layernorm (pre-MoE)
  │     512×2880
  │
  ├─► [7] gemm — Router (fp16)
  │     m=512, n=128, k=2880
  │
  ├─► [8] QMoE fused op ──────────────────────────────────
  │     │
  │     ├─ [8a] topk routing (GPU kernel)
  │     ├─ [8b] D2H sync — copy expert assignments to host + hipStreamSynchronize
  │     ├─ [8c] alloc — 8× hipMalloc for scratch buffers
  │     │
  │     └─ for each active expert (avg ~74 of 128):
  │          ├─ [8d] H2D — upload token_ids + weights
  │          ├─ [8e] gather — extract token subset from input
  │          ├─ [8f] fc1 — matmul_nbits: [count, 2880] × [5760, 2880]ᵀ → [count, 5760]
  │          ├─ [8g] swiglu — SiLU(gate) ⊙ up → [count, 2880]
  │          ├─ [8h] fc2 — matmul_nbits: [count, 2880] × [2880, 2880]ᵀ → [count, 2880]
  │          └─ [8i] scatter_add — weighted accumulate to output
  │
  └─► [9] elementwise — Residual add
        512×2880
```

Final ops (executed once after all 36 layers):

```
  ├─► layernorm (final)           512×2880
  └─► matmul (LM head, fp16)     m=512, n=201088, k=2880
```

## Prefill Profiling Results (First 512-Token Chunk)

Total compute time: **31,362 ms**

### Full Operator Breakdown

| Operator | Calls | GPU (ms) | CPU (ms) | GPU % |
|----------|------:|----------:|----------:|------:|
| **qmoe/fc1** | 2669 | **20,973** | 20,228 | **68.9%** |
| **qmoe/fc2** | 2669 | **7,762** | 7,801 | **25.5%** |
| matmul_nbits (attention) | 72 | 1,278 | 1,250 | 4.2% |
| — QKV: m=512,n=5120,k=2880 | 36 | 646 | 616 | 2.1% |
| — O proj: m=512,n=2880,k=4096 | 36 | 632 | 634 | 2.1% |
| gqa | 36 | 115 | 116 | 0.4% |
| qmoe/h2d | 2669 | 102 | 1,571 | 0.3% |
| qmoe/gather | 2669 | 53 | 5 | 0.2% |
| qmoe/d2h_sync | 36 | 45 | 222 | 0.1% |
| elementwise | 72 | 35 | 5 | 0.1% |
| matmul (LM head) | 1 | 25 | 0.2 | 0.1% |
| skip_layernorm | 72 | 13 | 10 | 0.0% |
| qmoe/swiglu | 2669 | 11 | 5 | 0.0% |
| layernorm | 1 | 11 | 10 | 0.0% |
| gemm (router) | 36 | 7 | 3 | 0.0% |
| qmoe/alloc | 36 | 3 | 7 | 0.0% |
| qmoe/scatter | 2669 | 2 | 5 | 0.0% |
| qmoe/topk | 36 | 1 | 1 | 0.0% |
| **TOTAL** | | **30,450** | | |

### QMoE Phase Summary

| Phase | GPU (ms) | GPU % of QMoE | Note |
|-------|----------:|--------------:|------|
| fc1 (matmul_nbits) | 20,973 | 72.5% | n=5760 (2× inter for SwiGLU fusion) |
| fc2 (matmul_nbits) | 7,762 | 26.8% | n=2880 |
| h2d (per-expert) | 102 | 0.4% | 2669 small memcpy launches |
| gather | 53 | 0.2% | |
| d2h_sync | 45 | 0.2% | includes hipStreamSynchronize |
| swiglu | 11 | 0.0% | |
| alloc | 3 | 0.0% | |
| scatter | 2 | 0.0% | |
| topk | 1 | 0.0% | |
| **Total QMoE** | **28,952** | **100%** | **95.1% of total prefill** |

### fc1 Shape Distribution (Prefill, 512 Tokens)

The 2669 fc1 calls span a wide range of `m` values (tokens per expert), from 1 to 217:

| Tokens per expert (m) | Expert count | Typical GPU time per call |
|-----------------------:|-------------:|--------------------------:|
| 1–5 | ~800 | 0.2–0.6 ms |
| 6–15 | ~700 | 0.8–4.0 ms |
| 16–50 | ~600 | 4–9 ms |
| 51–100 | ~300 | 60–70 ms |
| 100–217 | ~50 | 70–340 ms |

Most expert invocations have small `m` (few tokens), but the long tail of large experts dominates total time.

## Decode Profiling Results (Single Token, Steady State)

Total compute time: **~21.5 ms** per token (~14.5 tok/s)

| Operator | Calls | GPU (ms) | CPU (ms) | GPU % |
|----------|------:|----------:|----------:|------:|
| matmul_nbits (attention) | 72 | 6.4 | 0.2 | 29.6% |
| matmul (LM head) | 1 | 5.1 | 0.0 | 23.8% |
| gqa | 36 | 3.8 | 1.7 | 17.9% |
| qmoe/d2h_sync | 36 | 2.2 | 18.6 | 10.0% |
| elementwise | 72 | 2.0 | 0.2 | 9.2% |
| skip_layernorm | 72 | 1.0 | 0.6 | 4.8% |
| qmoe/alloc | 36 | 0.8 | 0.5 | 3.4% |
| gemm (router) | 36 | 0.0 | 0.6 | 0.1% |
| qmoe/topk | 36 | 0.0 | 0.1 | 0.1% |
| **TOTAL** | | **21.5** | | |

In decode, QMoE fc1/fc2 are sub-millisecond per expert (m=1) and don't appear
in the aggregated table. The bottleneck shifts to attention matmul_nbits (30%)
and the LM head matmul (24%).

Note: `qmoe/d2h_sync` has 2.2 ms GPU but **18.6 ms CPU** in decode — the
`hipStreamSynchronize` blocks the CPU for 18.6 ms while waiting for prior GPU
work to complete (topk routing + memcpy). This is the most expensive QMoE
overhead in decode.

## Key Observations

1. **QMoE fc1 + fc2 account for 94.4% of prefill GPU time.** The `hip_matmul_nbits`
   kernel is the sole bottleneck. All other operations combined are < 6%.

2. **fc1 is 2.7× slower than fc2** because fc1's output dimension is 5760
   (2 × inter_size, SwiGLU fusion packs gate and up projections) vs fc2's 2880.

3. **GPU time ≈ CPU time** for fc1/fc2, indicating near-synchronous execution.
   Root cause: the `hip_matmul_nbits` autotune mechanism benchmarks all
   kernel configurations on first call per `(M,N,K,block_size)` shape.
   QMoE produces ~100+ unique M values in a 512-token chunk, triggering
   ~12,000 `hipEventSynchronize` calls during the first prefill chunk
   (see "Root Cause Analysis" section below for details).

4. **Expert loop is fully serial.** The current implementation iterates over
   128 experts sequentially, calling `hip_matmul_nbits` individually for each.
   With 512 tokens and k=2, approximately 74 experts are active per layer,
   each processing an average of ~14 tokens (but highly variable: 1 to 217).

5. **Per-expert H2D overhead** (`qmoe/h2d`): 102 ms GPU but **1,571 ms CPU**.
   The 2669 small `hipMemcpyAsync` calls accumulate significant CPU-side launch
   overhead (~0.6 ms each).

6. **hipMalloc per layer** (`qmoe/alloc`): 36 calls × 8 allocations each. Only
   2.5 ms GPU / 7 ms CPU — not a bottleneck currently, but unnecessary
   repeated allocation on every layer invocation.

## Benchmark Results: Baseline vs Approach A (skip_autotune)

All benchmarks: `-r 5 -w 1` (5 measured iterations, 1 warmup), `HIPDNN_EP_PERF=1`,
prompt=2048 tokens (4 × 512-token chunks), generation=128 tokens.

**Approach A**: `skip_autotune=1` for QMoE fc1/fc2 calls only. Uses heuristic
kernel selection (`pickDefaultWmmaConfig` / `pickDefaultGemvConfig`) instead of
benchmarking all configs. Attention `matmul_nbits` unchanged (still autotunes).
No `bucketM`.

### High-Level Metrics

| Metric | Baseline | Approach A | Speedup |
|--------|--------:|-----------:|--------:|
| TTFT avg (ms) | 22,188 | 7,503 | **3.0×** |
| TTFT stddev (ms) | 7,625 | 227 | 33.6× more stable |
| TTFT p50 (ms) | 23,420 | 7,429 | 3.2× |
| Decode TPS (avg) | 22.0 | 23.1 | 1.05× |
| Warmup TTFT (ms) | 117,641 | 9,328 | **12.6×** |
| E2E avg (ms) | 40,623 | 26,470 | 1.5× |
| Output quality | Coherent | Coherent | — |
| Errors/NaN/Inf | None | None | — |

### Per-Iteration TTFT (GPU time, ms)

| Iteration | Baseline | Approach A |
|-----------|--------:|-----------:|
| WARMUP | 117,641 | 9,328 |
| iter 1 | 29,759 | 7,815 |
| iter 2 | 30,349 | 7,362 |
| iter 3 | 16,978 | 7,429 |
| iter 4 | 23,420 | 7,709 |
| iter 5 | 10,436 | 7,199 |

Baseline TTFT varies 10.4–30.3s due to autotune cache warming across
iterations. Approach A is stable at 7.2–7.8s because heuristic selection
is deterministic — no autotune overhead at all.

### Per-Chunk TTFT Breakdown (avg over 5 measured iterations, ms)

| Chunk | Baseline | Approach A | Speedup |
|-------|--------:|-----------:|--------:|
| Chunk 1 | 3,926 | 1,892 | 2.1× |
| Chunk 2 | 5,354 | 1,889 | 2.8× |
| Chunk 3 | 5,855 | 1,950 | 3.0× |
| Chunk 4 | 7,053 | 1,772 | 4.0× |

Baseline chunks get progressively slower because later chunks encounter
new M values not yet in the autotune cache (different expert-token
distributions per chunk). Approach A chunks are consistent (~1.8–1.9s)
since no autotune occurs.

### Op-Level Comparison (Chunk 1, avg over 5 iters)

| Operator | Baseline GPU | Approach A GPU | Speedup |
|----------|------------:|--------------:|--------:|
| qmoe/fc1 | 2,814 ms | 1,573 ms | 1.8× |
| qmoe/fc2 | 789 ms | 10 ms | **79×** |
| qmoe/h2d | 140 ms | 128 ms | 1.1× |
| qmoe/gather | 70 ms | 68 ms | 1.0× |
| elementwise | 30 ms | 30 ms | 1.0× |
| matmul (LM head) | 26 ms | 25 ms | 1.0× |
| matmul_nbits (attn) | 21 ms | 19 ms | 1.1× |
| qmoe/d2h_sync | 12 ms | 15 ms | 0.8× |
| gemm (router) | 7 ms | 9 ms | 0.8× |
| skip_layernorm | 6 ms | 6 ms | 1.0× |
| qmoe/swiglu | 3 ms | 3 ms | 1.0× |
| gqa | 3 ms | 3 ms | 1.0× |
| qmoe/scatter | 2 ms | 2 ms | 1.0× |
| **TOTAL** | **3,926** | **1,892** | **2.1×** |

The fc2 79× speedup is because baseline fc2 runs autotune on ~100 unique
M values (each running 60+ kernel configs × 6 launches). Approach A just
runs the actual computation once per call.

fc1 shows a smaller 1.8× improvement. In baseline, fc1 autotuning likely
overlaps with fc2 (same M values already cached for fc1 by the time fc2
runs). In Approach A, fc1 is the pure GPU compute time for 3218 expert
calls.

### CPU Time Analysis

| Phase | Baseline CPU | Approach A CPU |
|-------|------------:|---------------:|
| qmoe/fc1 | 1,942 ms | **15 ms** |
| qmoe/fc2 | 800 ms | **13 ms** |
| qmoe/h2d | 2,024 ms | 3,029 ms |

Baseline fc1/fc2 CPU time is high because `hipEventSynchronize` in autotune
blocks the CPU. Approach A CPU is near-zero — pure async kernel launch overhead.

h2d CPU is *higher* in Approach A (3,029 vs 2,024 ms) because the GPU
finishes faster, making `hipMemcpyAsync` CPU overhead more visible in the
profiling timeline.

### Validation Summary

| Check | Result |
|-------|--------|
| Errors in log (grep for error/ERROR/NaN/Inf) | **None** |
| QMoE phases present in all 4 chunks × 5 iters | **Yes** |
| Expert call counts (fc1 calls per chunk) | 3042–3254 (consistent with baseline 3208–3257) |
| Output text quality | Coherent, same pattern as baseline |
| Decode TPS | 23.1 (slightly better than baseline 22.0) |

## Root Cause Analysis: CPU Time ≈ GPU Time in fc1/fc2

### The Problem

The profiling data shows CPU time nearly equals GPU time for `qmoe/fc1`
(20,228 ms CPU vs 20,973 ms GPU) and `qmoe/fc2` (7,801 ms vs 7,762 ms).
If kernel launches were truly asynchronous, CPU time would be a small fraction
of GPU time (just launch overhead). Near-equality means the CPU blocks on
almost every kernel launch.

### Root Cause: Per-Shape Autotune with `hipEventSynchronize`

The `hip_matmul_nbits` function
(`3rd-party/custom_kernels/hip/matmul_nbits_kernel.hip`) has an autotune
mechanism that benchmarks all kernel configurations on the first call for
each unique `(M, N, K, block_size)` shape:

**WMMA path** (M ≥ 16): `getOrTuneWmmaConfig()` — benchmarks **60+
configurations**, each running 1 warmup + 5 timed iterations, with
`hipEventSynchronize(ev1)` per config. That's ~360 kernel launches and
~60 `hipEventSynchronize` calls per unique shape.

**GEMV path** (M < 16): `getOrTuneGemvConfig()` — benchmarks **28
configurations** with the same pattern. ~168 kernel launches and ~28
`hipEventSynchronize` calls per unique shape.

### Why QMoE Triggers Massive Autotune

QMoE calls `hip_matmul_nbits` with varying `M = count` (tokens assigned
to each expert). In a 512-token prefill chunk with 128 experts per layer:

- `M` ranges from **1 to 217** across 2669 expert invocations
- There are **~100+ unique M values** (see shape distribution)
- fc1 shape `(M, 5760, 2880)` and fc2 shape `(M, 2880, 2880)` each
  produce ~100 unique autotune keys
- Total autotune overhead: ~200 unique shapes × (60 configs × 6 launches
  × `hipEventSynchronize`) ≈ **~72,000 kernel launches + ~12,000
  synchronization points** during the first prefill chunk

The `hipEventSynchronize` calls are the critical bottleneck — each one
blocks the CPU until the GPU completes all preceding work, destroying
any pipeline parallelism.

### GEMV Col-Major Path Adds Extra Overhead

For experts with M < 16 (the majority — ~800 of 2669 calls have M ≤ 5),
the GEMV col-major path additionally:

1. Calls `ensureAColBuffer()` → may trigger `hipMalloc` on size change
2. Launches `transpose2d_fp16_kernel` to convert A from row → col-major
3. Launches the GEMV kernel
4. Launches `transpose2d_fp16_kernel` again to convert output back

These extra transpose kernels double the kernel launch count for small-M
experts.

### Global Static Buffers Cause Reallocation Churn

`hip_matmul_nbits` uses global static buffers (`g_dq_buf`, `g_a_col_buf`,
`g_out_col_buf`, `g_zp_fp16_buf`) that grow on demand via `hipMalloc`/
`hipFree`. When QMoE alternates between experts with different M values,
these buffers may need resizing, adding more synchronous GPU memory
operations.

### Summary

| Factor | Impact |
|--------|--------|
| Per-shape autotune (hipEventSynchronize) | **Primary** — ~12,000 sync points in first chunk |
| Extra transposes for M < 16 | **Moderate** — doubles kernel count for small experts |
| Global buffer reallocation | **Minor** — only triggers on size increase |
| Mutex lock in autotune cache | **Negligible** — QMoE is single-threaded |

## Comparative Analysis: Attention matmul_nbits vs QMoE matmul_nbits

Both attention projections and QMoE expert FFNs call the same
`hip_matmul_nbits` function, which uses the same autotune mechanism.
Both exhibit CPU ≈ GPU time:

| Caller | Calls | GPU (ms) | CPU (ms) | CPU/GPU |
|--------|------:|---------:|---------:|--------:|
| matmul_nbits (attention) | 72 | 1,278 | 1,250 | 97.8% |
| qmoe/fc1 | 2,669 | 20,973 | 20,228 | 96.4% |
| qmoe/fc2 | 2,669 | 7,762 | 7,801 | 100.5% |

The critical difference is the **number of unique shapes** that trigger
autotune:

### Attention matmul_nbits: 2 Unique Shapes

Attention projections use fixed dimensions across all 36 layers:

| Shape | Calls | Autotune runs |
|-------|------:|--------------:|
| QKV: m=512, n=5120, k=2880 | 36 | **1** (layer 0 only) |
| O proj: m=512, n=2880, k=4096 | 36 | **1** (layer 0 only) |
| **Total** | **72** | **2** |

After layer 0 warms the cache, the remaining 70 calls across layers 1–35
are all cache hits. The autotune cost is amortized over 72 calls.

### QMoE matmul_nbits: ~200 Unique Shapes

QMoE's `M = count` (tokens per expert) varies from 1 to 217. Each
unique M value produces a distinct autotune cache key for both fc1 and fc2:

| Shape family | N | K | Unique M values | Autotune runs |
|-------------|---:|----:|----------------:|--------------:|
| fc1 | 5,760 | 2,880 | ~100 | ~100 |
| fc2 | 2,880 | 2,880 | ~100 | ~100 |
| **Total** | | | | **~200** |

The autotune cache (`matmul_nbits_kernel.hip` L480, L1328) is keyed by
`"M_N_K_blocksize"` — identical N, K, block_size but different M values
produce different keys:

```
"1_5760_2880_32_0"    → autotune (28 configs × hipEventSynchronize)
"2_5760_2880_32_0"    → autotune
"3_5760_2880_32_0"    → autotune
...
"217_5760_2880_32_0"  → autotune
```

### Autotune Cost Comparison

| | Attention | QMoE |
|--|--:|--:|
| Total calls | 72 | 5,338 |
| Unique shapes | 2 | ~200 |
| Autotune runs | 2 | ~200 |
| hipEventSynchronize (est.) | ~120 | **~12,000** |
| Autotune kernel launches (est.) | ~720 | **~72,000** |
| Autotune calls / total calls | 2.8% | 3.7% |

Despite similar autotune-to-total-call ratios, the absolute count of
synchronization points differs by **100×**. Each `hipEventSynchronize`
blocks the CPU until the GPU completes all preceding work, making the
autotune overhead the dominant cost in QMoE's prefill latency.

### Additional Overhead: GEMV Col-Major Transpose

For experts with M < 16 (~800 of 2669 calls), `hip_matmul_nbits` takes
the GEMV col-major path which launches extra transpose kernels:

1. `transpose2d_fp16_kernel`: A row-major → col-major (skipped for M=1)
2. `matmul_nbits_gemv_kernel`: the actual GEMV
3. `transpose2d_fp16_kernel`: output col-major → row-major (skipped for M=1)

For M=2..15, each matmul_nbits call launches **3 kernels** instead of 1.
Attention matmul_nbits (M=512) always takes the WMMA path and never
needs transposes.
