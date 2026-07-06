# Bottleneck report -- Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml

Probe dir: `probe_20260705_205931`  |  roofline peak: 256 GB/s  |  gen=128 reps=5 warmup=2

> Phase 1 report: per-op GB/s and % peak are shape-model ESTIMATES (marked est). Phase 2 replaces them with EP-measured bytes.

## A. Headline (model_benchmark)

| prompt | TTFT (s) | prefill tok/s | decode tok/s | decode ms/tok | peak WS (GB) | kernel errors |
|---|---|---|---|---|---|---|
| 128 | 0.571 | 224 | 39.5 | 25.31 | 22.0 | False |
| 512 | 1.641 | 312 | 40.7 | 24.57 | 23.3 | False |
| 2048 | 5.661 | 362 | 32.8 | 30.45 | 27.8 | False |

## B. Decode per-op roofline (steady token; total GPU 63.5 ms/Compute)

| op | calls | gpu ms | % decode | bytes (MB, est) | GB/s (est) | % peak (est) |
|---|---|---|---|---|---|---|
| matmul_nbits | 391 | 20.2 | 31.9% | 1094.0 | 54 | 21% |
| elementwise | 440 | 7.7 | 12.1% | 6.3 | 1 | - |
| qmoe | 40 | 6.4 | 10.0% | 566.6 | 89 | 35% |
| activation | 180 | 5.4 | 8.5% | 1.2 | - | - |
| cast | 192 | 4.5 | 7.0% | 2.8 | 1 | - |
| linear_attention | 30 | 4.2 | 6.7% | - | - | - |
| transpose | 100 | 3.2 | 5.0% | 2.3 | 1 | - |
| skip_layernorm | 80 | 3.1 | 4.9% | 1.3 | - | - |
| layernorm | 51 | 2.6 | 4.0% | 0.7 | - | - |
| power | 120 | 2.0 | 3.1% | - | - | - |
| gqa | 10 | 1.9 | 3.0% | - | - | - |
| causal_conv | 30 | 0.9 | 1.4% | - | - | - |
| reduce_sum | 61 | 0.9 | 1.4% | 0.2 | - | - |
| rotary_emb | 20 | 0.2 | 0.4% | - | - | - |
| gather | 9 | 0.2 | 0.3% | - | - | - |
| sub | 1 | 0.0 | 0.1% | - | - | - |

## C. matmul_nbits per-shape roofline (est)

| shape | calls | gpu ms | % decode | bits | bytes (MB) | GB/s | % peak | class |
|---|---|---|---|---|---|---|---|---|
| m=1,n=512,k=2048 | 100 | 4.3 | 6.8% | 4 | 59.5 | 14 | 5% | overhead-bound |
| m=1,n=8192,k=2048 | 40 | 3.5 | 5.4% | 8 | 378.3 | 109 | 43% | headroom |
| m=1,n=4096,k=2048 | 30 | 2.4 | 3.8% | 8 | 141.9 | 58 | 23% | headroom |
| m=1,n=2048,k=4096 | 40 | 2.3 | 3.7% | 8 | 189.2 | 81 | 32% | headroom |
| m=1,n=248320,k=2048 | 1 | 1.9 | 3.0% | 4 | 286.6 | 152 | 60% | near-roofline |
| m=1,n=256,k=2048 | 40 | 1.6 | 2.6% | 4 | 12.0 | 7 | 3% | overhead-bound |
| m=1,n=2048,k=512 | 40 | 1.6 | 2.5% | 4 | 23.8 | 15 | 6% | overhead-bound |
| m=1,n=32,k=2048 | 60 | 1.3 | 2.1% | 8 | 2.5 | 2 | 1% | overhead-bound |
| m=1,n=1,k=2048 | 40 | 1.3 | 2.0% | 4 | 0.2 | 0 | 0% | overhead-bound |

## D. qmoe

- decode: 40 calls, 6.4 ms/Compute, 10.0% of decode GPU time.
- Sparsity realized (only k=8 active experts read); see prefill for the bigger qmoe cost.

## E. Launch critical-path A/B (p128 decode)

- EP launch_gap (wall-gpu) over 505 decode Computes: median 0% of wall is host launch/dispatch not hidden by GPU (range 0-74%).
  => GPU-compute-bound -> prioritize kernel efficiency over fusion.

| mode | decode tok/s | decode ms/tok |
|---|---|---|
| normal (async) | 32.0 | 31.20 |
| HIP_LAUNCH_BLOCKING=1 | 10.9 | 92.04 |

- Serializing launches changes decode 31.20 -> 92.04 ms/tok (2.95x). ~60.84 ms/tok of launch overhead is currently HIDDEN by overlap.
- Interpretation: launch overhead is mostly OVERLAPPED behind compute -> fusion yields little decode TPS; focus on the big compute kernels.

## F. Prefill / TTFT

- TTFT vs prompt (from section A): p128=0.57s, p512=1.64s, p2048=5.66s

- Prefill Compute per-op GPU breakdown (total 1826.2 ms):

| op | calls | gpu ms | % prefill |
|---|---|---|---|
| qmoe | 40 | 1499.6 | 82.1% |
| gqa | 10 | 123.9 | 6.8% |
| linear_attention | 30 | 74.5 | 4.1% |
| matmul_nbits | 391 | 66.3 | 3.6% |
| elementwise | 440 | 15.6 | 0.9% |
| transpose | 100 | 8.6 | 0.5% |
| cast | 192 | 8.0 | 0.4% |
| layernorm | 51 | 6.4 | 0.4% |

## G. Cold start

- Minimal-run (l=8,g=1) process wall: 31.9 s (model load + compile + weight upload + first-inference autotune, load-dominated).
- EP phase timers (measured, summed over cold-start events):
  - autotune: 10.112 s (n=32)
  - weight_upload: 4.995 s (n=3)

## (aux) Isolated per-shape kernel time (single_op seq1, gs128 graphs)

| op graph | shape | calls | gpu ms |
|---|---|---|---|
| MatMulNBits_down_proj | m=1,n=2048,k=512 | 1 | 0.000 |
| MatMulNBits_gate_proj | m=1,n=512,k=2048 | 1 | 0.000 |
| MatMulNBits_in_proj_b | m=1,n=32,k=2048 | 1 | 0.000 |
| MatMulNBits_in_proj_qkv | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_in_proj_z | m=1,n=4096,k=2048 | 1 | 0.100 |
| MatMulNBits_lm_head | m=1,n=248320,k=2048 | 1 | 2.100 |
| MatMulNBits_out_proj | m=1,n=2048,k=4096 | 1 | 0.100 |
| MatMulNBits_o_proj | m=1,n=2048,k=4096 | 1 | 0.000 |
| MatMulNBits_q_proj | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_router | m=1,n=256,k=2048 | 1 | 0.000 |
| MatMulNBits_shared_expert_gate | m=1,n=1,k=2048 | 1 | 0.000 |
| QMoE | 1x2048x512,e=256 | 1 | 0.000 |

## H. Ranked decode bottlenecks (by GPU-time share)

1. **matmul_nbits** -- 31.9% of decode (21% peak). Split into medium-N GEMV kernel efficiency (headroom) + small-N launch/batching (overhead-bound); see section C.
2. **elementwise** -- 12.1% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
3. **qmoe** -- 10.0% of decode (35% peak). Grouped-expert GEMM (biggest prefill/TTFT lever); decode kernel BW tuning.
4. **activation** -- 8.5% of decode (n/a). Small-op tail -- fuse only if section E shows launches on the critical path.
5. **cast** -- 7.0% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
6. **linear_attention** -- 6.7% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
