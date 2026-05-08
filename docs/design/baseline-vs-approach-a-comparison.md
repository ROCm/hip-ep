<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Baseline vs Approach A: Per-Iteration Per-Chunk Comparison

Test config: GPT-OSS-120B, gfx1151, chunk_size=512, prompt=2048 (4 chunks),
`-r 5 -w 1`, `HIPDNN_EP_PERF=1`. All times in **milliseconds** (GPU time from op profiling).

## 1. Per-Iteration Per-Chunk TOTAL GPU Time (ms)

### Baseline

| Iteration | Chunk 1 | Chunk 2 | Chunk 3 | Chunk 4 | **Sum (TTFT)** |
|-----------|--------:|--------:|--------:|--------:|---------------:|
| WARMUP | 50,935 | 18,618 | 16,171 | 31,916 | 117,641 |
| iter 1 | 6,199 | 7,547 | 5,167 | 10,847 | **29,759** |
| iter 2 | 5,776 | 6,800 | 9,205 | 8,569 | **30,349** |
| iter 3 | 2,852 | 5,521 | 3,832 | 4,774 | **16,978** |
| iter 4 | 2,575 | 4,302 | 9,855 | 6,689 | **23,420** |
| iter 5 | 2,230 | 2,601 | 1,217 | 4,388 | **10,436** |
| **Avg (1–5)** | **3,926** | **5,354** | **5,855** | **7,053** | **22,188** |
| Stddev | 1,793 | 1,886 | 3,370 | 2,571 | 7,625 |

### Approach A (skip_autotune)

| Iteration | Chunk 1 | Chunk 2 | Chunk 3 | Chunk 4 | **Sum (TTFT)** |
|-----------|--------:|--------:|--------:|--------:|---------------:|
| WARMUP | 3,277 | 2,015 | 2,024 | 2,011 | 9,328 |
| iter 1 | 1,836 | 2,024 | 2,001 | 1,953 | **7,815** |
| iter 2 | 1,923 | 1,832 | 1,926 | 1,682 | **7,362** |
| iter 3 | 1,962 | 1,929 | 1,984 | 1,554 | **7,429** |
| iter 4 | 1,969 | 1,894 | 2,002 | 1,843 | **7,709** |
| iter 5 | 1,771 | 1,765 | 1,836 | 1,828 | **7,199** |
| **Avg (1–5)** | **1,892** | **1,889** | **1,950** | **1,772** | **7,503** |
| Stddev | 80 | 95 | 66 | 149 | 227 |

### Per-Chunk Speedup (avg)

| | Chunk 1 | Chunk 2 | Chunk 3 | Chunk 4 | **Total** |
|--|--------:|--------:|--------:|--------:|----------:|
| Baseline avg | 3,926 | 5,354 | 5,855 | 7,053 | 22,188 |
| Approach A avg | 1,892 | 1,889 | 1,950 | 1,772 | 7,503 |
| **Speedup** | **2.1×** | **2.8×** | **3.0×** | **4.0×** | **3.0×** |

Baseline gets slower from Chunk 1 → Chunk 4 because later chunks encounter
new M values requiring additional autotune runs. Approach A is flat across
all chunks (~1.8–1.95s) since heuristic selection has no warm-up cost.

## 2. Decode TPS Per Iteration

| Iteration | Baseline (ms/tok) | Baseline TPS | Approach A (ms/tok) | Approach A TPS |
|-----------|------------------:|-------------:|--------------------:|---------------:|
| WARMUP | 46.4 | 21.5 | 42.9 | 23.3 |
| iter 1 | 45.9 | 21.8 | 42.5 | 23.5 |
| iter 2 | 45.9 | 21.8 | 44.4 | 22.5 |
| iter 3 | 45.9 | 21.8 | 44.4 | 22.5 |
| iter 4 | 44.6 | 22.4 | 43.1 | 23.2 |
| iter 5 | 44.5 | 22.5 | 42.4 | 23.6 |
| **Avg (1–5)** | **45.4** | **22.0** | **43.4** | **23.1** |

Decode TPS improves ~5% with Approach A. Decode calls QMoE with M=1 per
expert; heuristic selection avoids the first-call autotune overhead.

## 3. Op-Level Breakdown Per Chunk (avg over 5 measured iterations)

### Chunk 1

| Operator | Baseline GPU | B % | A GPU | A % | GPU Speedup |
|----------|------------:|----:|------:|----:|------------:|
| qmoe/fc1 | 2,814 | 71.7% | 1,573 | 83.1% | 1.8× |
| qmoe/fc2 | 789 | 20.1% | 10 | 0.5% | **79×** |
| qmoe/h2d | 140 | 3.6% | 128 | 6.8% | 1.1× |
| qmoe/gather | 70 | 1.8% | 68 | 3.6% | 1.0× |
| elementwise | 30 | 0.8% | 30 | 1.6% | 1.0× |
| matmul (LM head) | 26 | 0.7% | 25 | 1.3% | 1.0× |
| matmul_nbits (attn) | 21 | 0.5% | 19 | 1.0% | 1.1× |
| qmoe/d2h_sync | 12 | 0.3% | 15 | 0.8% | 0.8× |
| gemm (router) | 7 | 0.2% | 9 | 0.5% | — |
| skip_layernorm | 6 | 0.1% | 6 | 0.3% | 1.0× |
| qmoe/swiglu | 3 | 0.1% | 3 | 0.1% | 1.0× |
| gqa | 3 | 0.1% | 3 | 0.2% | 1.0× |
| qmoe/scatter | 2 | 0.1% | 2 | 0.1% | 1.0× |
| **TOTAL** | **3,926** | **100%** | **1,892** | **100%** | **2.1×** |

### Chunk 2

| Operator | Baseline GPU | B % | A GPU | A % | GPU Speedup |
|----------|------------:|----:|------:|----:|------------:|
| qmoe/fc1 | 3,798 | 70.9% | 1,547 | 81.9% | 2.5× |
| qmoe/fc2 | 1,221 | 22.8% | 10 | 0.5% | **122×** |
| qmoe/h2d | 149 | 2.8% | 151 | 8.0% | 1.0× |
| qmoe/gather | 70 | 1.3% | 67 | 3.6% | 1.0× |
| elementwise | 30 | 0.6% | 31 | 1.6% | 1.0× |
| matmul (LM head) | 26 | 0.5% | 25 | 1.3% | 1.0× |
| matmul_nbits (attn) | 21 | 0.4% | 19 | 1.0% | 1.1× |
| qmoe/d2h_sync | 14 | 0.3% | 13 | 0.7% | 1.1× |
| gqa | 6 | 0.1% | 6 | 0.3% | 1.0× |
| gemm (router) | 6 | 0.1% | 7 | 0.4% | — |
| skip_layernorm | 6 | 0.1% | 6 | 0.3% | 1.0× |
| qmoe/swiglu | 4 | 0.1% | 3 | 0.1% | 1.3× |
| qmoe/scatter | 2 | 0.0% | 2 | 0.1% | 1.0× |
| **TOTAL** | **5,354** | **100%** | **1,889** | **100%** | **2.8×** |

### Chunk 3

| Operator | Baseline GPU | B % | A GPU | A % | GPU Speedup |
|----------|------------:|----:|------:|----:|------------:|
| qmoe/fc1 | 4,177 | 71.3% | 1,615 | 82.8% | 2.6× |
| qmoe/fc2 | 1,355 | 23.1% | 12 | 0.6% | **113×** |
| qmoe/h2d | 137 | 2.3% | 139 | 7.1% | 1.0× |
| qmoe/gather | 69 | 1.2% | 68 | 3.5% | 1.0× |
| elementwise | 30 | 0.5% | 31 | 1.6% | 1.0× |
| matmul (LM head) | 25 | 0.4% | 25 | 1.3% | 1.0× |
| matmul_nbits (attn) | 21 | 0.4% | 19 | 1.0% | 1.1× |
| qmoe/d2h_sync | 12 | 0.2% | 14 | 0.7% | 0.9× |
| gqa | 8 | 0.1% | 8 | 0.4% | 1.0× |
| gemm (router) | 6 | 0.1% | 7 | 0.3% | — |
| skip_layernorm | 6 | 0.1% | 6 | 0.3% | 1.0× |
| qmoe/swiglu | 4 | 0.1% | 3 | 0.1% | 1.3× |
| qmoe/scatter | 2 | 0.0% | 2 | 0.1% | 1.0× |
| **TOTAL** | **5,855** | **100%** | **1,950** | **100%** | **3.0×** |

### Chunk 4

| Operator | Baseline GPU | B % | A GPU | A % | GPU Speedup |
|----------|------------:|----:|------:|----:|------------:|
| qmoe/fc1 | 5,018 | 71.1% | 1,450 | 81.9% | 3.5× |
| qmoe/fc2 | 1,704 | 24.2% | 11 | 0.6% | **155×** |
| qmoe/h2d | 142 | 2.0% | 128 | 7.2% | 1.1× |
| qmoe/gather | 69 | 1.0% | 64 | 3.6% | 1.1× |
| elementwise | 30 | 0.4% | 30 | 1.7% | 1.0× |
| matmul (LM head) | 26 | 0.4% | 25 | 1.4% | 1.0× |
| matmul_nbits (attn) | 21 | 0.3% | 19 | 1.1% | 1.1× |
| qmoe/d2h_sync | 14 | 0.2% | 14 | 0.8% | 1.0× |
| gqa | 10 | 0.1% | 10 | 0.6% | 1.0× |
| gemm (router) | 5 | 0.1% | 7 | 0.4% | — |
| skip_layernorm | 6 | 0.1% | 6 | 0.3% | 1.0× |
| qmoe/swiglu | 4 | 0.1% | 3 | 0.1% | 1.3× |
| qmoe/scatter | 2 | 0.0% | 2 | 0.1% | 1.0× |
| **TOTAL** | **7,053** | **100%** | **1,772** | **100%** | **4.0×** |

## 4. QMoE Phase Summary (avg across all 4 chunks, measured iterations)

| Phase | Baseline GPU | B % of Total | B % of QMoE | A GPU | A % of Total | A % of QMoE | Speedup |
|-------|------------:|------------:|------------:|------:|------------:|------------:|--------:|
| qmoe/fc1 | 3,952 | 71.3% | 72.5% | 1,545 | 82.2% | 86.8% | 2.6× |
| qmoe/fc2 | 1,267 | 22.8% | 23.2% | 11 | 0.6% | 0.6% | **115×** |
| qmoe/h2d | 142 | 2.6% | 2.6% | 137 | 7.3% | 7.7% | 1.0× |
| qmoe/gather | 69 | 1.2% | 1.3% | 67 | 3.6% | 3.8% | 1.0× |
| qmoe/d2h_sync | 13 | 0.2% | 0.2% | 14 | 0.7% | 0.8% | — |
| qmoe/swiglu | 3 | 0.1% | 0.1% | 3 | 0.2% | 0.2% | 1.0× |
| qmoe/scatter | 2 | 0.0% | 0.0% | 2 | 0.1% | 0.1% | 1.0× |
| qmoe/alloc | 1 | 0.0% | 0.0% | 1 | 0.1% | 0.1% | 1.0× |
| qmoe/topk | 1 | 0.0% | 0.0% | 1 | 0.1% | 0.1% | 1.0× |
| **QMoE subtotal** | **5,450** | **98.2%** | **100%** | **1,780** | **94.8%** | **100%** | **3.1×** |
| **TOTAL (all ops)** | **5,547** | **100%** | | **1,876** | **100%** | | **3.0×** |

### fc1/fc2 Percentage Shift

| | Baseline | Approach A |
|--|--------:|----------:|
| fc1 % of TOTAL | 71.3% | 82.2% |
| fc2 % of TOTAL | 22.8% | 0.6% |
| fc1+fc2 % of TOTAL | **94.1%** | **82.8%** |
| fc1 : fc2 ratio | 3.1 : 1 | 140 : 1 |

Baseline fc1+fc2 together account for 94% of total prefill time. After
skip_autotune, fc2 drops to negligible (0.6%) because its actual compute
is trivial — baseline fc2 was almost entirely autotune overhead. fc1
remains dominant at 82% of total, now representing pure GPU compute.

## 5. Observations

1. **fc2 is the biggest winner** — 115× speedup because baseline fc2 spends
   almost all its time in autotune (60+ configs × `hipEventSynchronize` per
   unique M). With skip_autotune, fc2's actual compute is trivial (~10ms for
   ~3200 calls across 36 layers).

2. **fc1 improves 2.6× but is still the dominant cost** — even without
   autotune, fc1 takes ~1.5s per chunk (83% of Approach A's total). fc1's
   output dimension is 5760 (2× inter_size for SwiGLU) vs fc2's 2880.

3. **Baseline later chunks are progressively slower** — Chunk 4 is 1.8× slower
   than Chunk 1 because different expert-token distributions expose new M values
   requiring additional autotune. Approach A chunks are uniformly ~1.9s.

4. **TTFT variance eliminated** — Baseline stddev 7,625ms (34% of mean),
   Approach A stddev 227ms (3% of mean). The autotune warm-up pattern is
   non-deterministic across iterations.

5. **Non-QMoE ops unchanged** — elementwise, matmul_nbits (attention),
   skip_layernorm, gqa all show identical times, confirming the change is
   isolated to QMoE.

6. **h2d CPU is higher in Approach A** — because GPU finishes faster, the
   profiling scope captures more CPU-side `hipMemcpyAsync` launch overhead.
   GPU h2d time is the same (~130–150ms).
