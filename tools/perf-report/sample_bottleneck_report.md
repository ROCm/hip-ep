# Bottleneck report -- Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml

Probe dir: `probe_20260705_225537`  |  roofline peak: 242 GB/s (measured (BW-probe read))  |  gen=128 reps=5 warmup=2

> Phase 1 report: per-op GB/s and % peak are shape-model ESTIMATES (marked est). Phase 2 replaces them with EP-measured bytes.

## A. Headline (model_benchmark)

| prompt | TTFT (s) | prefill tok/s | decode tok/s | decode ms/tok | peak WS (GB) | kernel errors |
|---|---|---|---|---|---|---|
| 128 | 0.594 | 215 | 40.2 | 24.90 | 22.0 | False |
| 512 | 1.531 | 334 | 41.2 | 24.28 | 23.4 | False |
| 2048 | 4.722 | 434 | 39.2 | 25.51 | 27.8 | False |

## B. Decode per-op roofline (steady token; total GPU 47.4 ms/Compute)

| op | calls | gpu ms | % decode | bytes (MB, est) | GB/s (est) | % peak (est) |
|---|---|---|---|---|---|---|
| matmul_nbits | 391 | 17.2 | 36.3% | 1094.0 | 64 | 25% |
| qmoe | 40 | 5.3 | 11.1% | 566.6 | 108 | 42% |
| elementwise | 440 | 5.1 | 10.7% | 6.3 | 1 | - |
| linear_attention | 30 | 3.6 | 7.6% | 63.5 | 18 | 7% |
| activation | 180 | 3.3 | 7.0% | 1.2 | - | - |
| cast | 192 | 2.8 | 6.0% | 2.8 | 1 | - |
| skip_layernorm | 80 | 2.4 | 5.0% | 1.3 | 1 | - |
| transpose | 100 | 1.8 | 3.8% | 2.3 | 1 | 1% |
| layernorm | 51 | 1.6 | 3.3% | 0.7 | - | - |
| gqa | 10 | 1.5 | 3.2% | - | - | - |
| power | 120 | 1.5 | 3.1% | - | - | - |
| reduce_sum | 61 | 0.7 | 1.6% | 0.2 | - | - |
| causal_conv | 30 | 0.3 | 0.7% | - | - | - |
| rotary_emb | 20 | 0.2 | 0.4% | - | - | - |
| gather | 9 | 0.2 | 0.3% | - | - | - |
| sub | 1 | 0.0 | 0.0% | - | - | - |

- Top-down aggregate memory roofline (profiler-independent): ~1.74 GB/token (byte-model, ~62% of ops covered) x 40.2 tok/s = ~70 GB/s achieved = ~29% of 242 GB/s peak.
  Memory-roofline decode ceiling ~= 139 tok/s (peak/bytes-per-token); measured 40.2 tok/s.

## C. matmul_nbits per-shape roofline (est)

| shape | calls | gpu ms | % decode | bits | bytes (MB) | GB/s | % peak | class |
|---|---|---|---|---|---|---|---|---|
| m=1,n=512,k=2048 | 100 | 3.9 | 8.3% | 4 | 59.5 | 15 | 6% | overhead-bound |
| m=1,n=8192,k=2048 | 40 | 3.7 | 7.8% | 8 | 378.3 | 102 | 40% | headroom |
| m=1,n=248320,k=2048 | 1 | 1.9 | 4.1% | 4 | 286.6 | 148 | 58% | near-roofline |
| m=1,n=4096,k=2048 | 30 | 1.7 | 3.6% | 8 | 141.9 | 83 | 32% | headroom |
| m=1,n=2048,k=4096 | 40 | 1.7 | 3.5% | 8 | 189.2 | 114 | 44% | headroom |
| m=1,n=32,k=2048 | 60 | 1.3 | 2.7% | 8 | 2.5 | 2 | 1% | overhead-bound |
| m=1,n=256,k=2048 | 40 | 1.1 | 2.3% | 4 | 12.0 | 11 | 4% | overhead-bound |
| m=1,n=2048,k=512 | 40 | 1.0 | 2.1% | 4 | 23.8 | 24 | 9% | overhead-bound |
| m=1,n=1,k=2048 | 40 | 0.9 | 1.9% | 4 | 0.2 | 0 | 0% | overhead-bound |

## D. qmoe

- decode: 40 calls, 5.3 ms/Compute, 11.1% of decode GPU time.
- Sparsity realized (only k=8 active experts read); see prefill for the bigger qmoe cost.

## E. Launch critical-path A/B (p128 decode)

- EP launch_gap (wall-gpu) over 505 decode Computes: median 0% of wall is host launch/dispatch not hidden by GPU (range 0-33%).
  => GPU-compute-bound -> prioritize kernel efficiency over fusion.

| mode | decode tok/s | decode ms/tok |
|---|---|---|
| normal (async) | 42.3 | 23.64 |
| HIP_LAUNCH_BLOCKING=1 | 11.3 | 88.54 |

- Serializing launches changes decode 23.64 -> 88.54 ms/tok (3.75x). ~64.90 ms/tok of launch overhead is currently HIDDEN by overlap.
- Interpretation: launch overhead is mostly OVERLAPPED behind compute -> fusion yields little decode TPS; focus on the big compute kernels.

## F. Prefill / TTFT

- TTFT vs prompt (from section A): p128=0.59s, p512=1.53s, p2048=4.72s

- Prefill Compute per-op GPU breakdown (total 1580.9 ms):

| op | calls | gpu ms | % prefill |
|---|---|---|---|
| qmoe | 40 | 1297.8 | 82.1% |
| gqa | 10 | 98.9 | 6.3% |
| linear_attention | 30 | 72.6 | 4.6% |
| matmul_nbits | 391 | 59.9 | 3.8% |
| elementwise | 440 | 11.8 | 0.7% |
| transpose | 100 | 7.8 | 0.5% |
| activation | 180 | 6.5 | 0.4% |
| cast | 192 | 6.0 | 0.4% |

- Prefill matmul_nbits compute roofline (p128): 500.2 GFLOP over ~23 ms (TTFT 0.59s x 4% matmul) = ~22.1 TFLOP/s = ~38% of ~59 TFLOP/s peak (peak ASSUMED -- confirm gfx1151 fp16 WMMA).

## G. Cold start

- Minimal-run (l=8,g=1) process wall: 29.7 s (model load + compile + weight upload + first-inference autotune, load-dominated).
- EP phase timers (measured, summed over cold-start events):
  - autotune: 10.051 s (n=32)
  - weight_upload: 3.608 s (n=3)

## (aux) Isolated per-shape kernel time (single_op seq1, gs128 graphs)

| op graph | shape | calls | gpu ms |
|---|---|---|---|
| MatMulNBits_down_proj | m=1,n=2048,k=512 | 1 | 0.000 |
| MatMulNBits_gate_proj | m=1,n=512,k=2048 | 1 | 0.000 |
| MatMulNBits_in_proj_b | m=1,n=32,k=2048 | 1 | 0.000 |
| MatMulNBits_in_proj_qkv | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_in_proj_z | m=1,n=4096,k=2048 | 1 | 0.100 |
| MatMulNBits_lm_head | m=1,n=248320,k=2048 | 1 | 1.800 |
| MatMulNBits_out_proj | m=1,n=2048,k=4096 | 1 | 0.100 |
| MatMulNBits_o_proj | m=1,n=2048,k=4096 | 1 | 0.100 |
| MatMulNBits_q_proj | m=1,n=8192,k=2048 | 1 | 0.000 |
| MatMulNBits_router | m=1,n=256,k=2048 | 1 | 0.000 |
| MatMulNBits_shared_expert_gate | m=1,n=1,k=2048 | 1 | 0.000 |
| QMoE | 1x2048x512,e=256 | 1 | 0.100 |

## H. Ranked decode bottlenecks (by GPU-time share)

1. **matmul_nbits** -- 36.3% of decode (25% peak). Split into medium-N GEMV kernel efficiency (headroom) + small-N launch/batching (overhead-bound); see section C.
2. **qmoe** -- 11.1% of decode (42% peak). Grouped-expert GEMM (biggest prefill/TTFT lever); decode kernel BW tuning.
3. **elementwise** -- 10.7% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
4. **linear_attention** -- 7.6% of decode (7% peak). Investigate per section E (launch-bound?) and roofline % above.
5. **activation** -- 7.0% of decode (n/a). Small-op tail -- fuse only if section E shows launches on the critical path.
6. **cast** -- 6.0% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
