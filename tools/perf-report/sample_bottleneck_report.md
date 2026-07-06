# Bottleneck report -- Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml

Probe dir: `probe_20260705_212700`  |  roofline peak: 256 GB/s  |  gen=128 reps=5 warmup=2

> Phase 1 report: per-op GB/s and % peak are shape-model ESTIMATES (marked est). Phase 2 replaces them with EP-measured bytes.

## A. Headline (model_benchmark)

| prompt | TTFT (s) | prefill tok/s | decode tok/s | decode ms/tok | peak WS (GB) | kernel errors |
|---|---|---|---|---|---|---|
| 128 | 0.654 | 196 | 37.8 | 26.44 | 21.0 | False |
| 512 | 1.591 | 322 | 39.2 | 25.48 | 22.4 | False |
| 2048 | 4.718 | 434 | 41.7 | 24.00 | 27.8 | False |

## B. Decode per-op roofline (steady token; total GPU 45.1 ms/Compute)

| op | calls | gpu ms | % decode | bytes (MB, est) | GB/s (est) | % peak (est) |
|---|---|---|---|---|---|---|
| matmul_nbits | 391 | 15.5 | 34.4% | 1094.0 | 70 | 28% |
| qmoe | 40 | 5.5 | 12.1% | 566.6 | 104 | 41% |
| elementwise | 440 | 4.9 | 10.8% | 6.3 | 1 | 1% |
| linear_attention | 30 | 3.5 | 7.7% | - | - | - |
| activation | 180 | 3.2 | 7.2% | 1.2 | - | - |
| cast | 192 | 2.8 | 6.2% | 2.8 | 1 | - |
| skip_layernorm | 80 | 2.3 | 5.1% | 1.3 | 1 | - |
| transpose | 100 | 1.9 | 4.3% | 2.3 | 1 | - |
| layernorm | 51 | 1.8 | 4.0% | 0.7 | - | - |
| gqa | 10 | 1.5 | 3.2% | - | - | - |
| power | 120 | 1.1 | 2.5% | - | - | - |
| reduce_sum | 61 | 0.5 | 1.1% | 0.2 | - | - |
| causal_conv | 30 | 0.3 | 0.7% | - | - | - |
| rotary_emb | 20 | 0.1 | 0.3% | - | - | - |
| gather | 9 | 0.1 | 0.3% | - | - | - |
| sub | 1 | 0.0 | 0.0% | - | - | - |

- Top-down aggregate memory roofline (profiler-independent): ~1.68 GB/token (byte-model, ~56% of ops covered) x 37.8 tok/s = ~63 GB/s achieved = ~25% of 256 GB/s peak.
  Memory-roofline decode ceiling ~= 153 tok/s (peak/bytes-per-token); measured 37.8 tok/s.

## C. matmul_nbits per-shape roofline (est)

| shape | calls | gpu ms | % decode | bits | bytes (MB) | GB/s | % peak | class |
|---|---|---|---|---|---|---|---|---|
| m=1,n=512,k=2048 | 100 | 3.3 | 7.3% | 4 | 59.5 | 18 | 7% | overhead-bound |
| m=1,n=8192,k=2048 | 40 | 2.8 | 6.3% | 8 | 378.3 | 133 | 52% | near-roofline |
| m=1,n=2048,k=4096 | 40 | 2.0 | 4.4% | 8 | 189.2 | 96 | 37% | headroom |
| m=1,n=4096,k=2048 | 30 | 1.9 | 4.3% | 8 | 141.9 | 73 | 29% | headroom |
| m=1,n=248320,k=2048 | 1 | 1.9 | 4.3% | 4 | 286.6 | 148 | 58% | near-roofline |
| m=1,n=256,k=2048 | 40 | 1.1 | 2.5% | 4 | 12.0 | 10 | 4% | overhead-bound |
| m=1,n=32,k=2048 | 60 | 1.1 | 2.3% | 8 | 2.5 | 2 | 1% | overhead-bound |
| m=1,n=2048,k=512 | 40 | 0.7 | 1.6% | 4 | 23.8 | 34 | 13% | headroom |
| m=1,n=1,k=2048 | 40 | 0.6 | 1.4% | 4 | 0.2 | 0 | 0% | overhead-bound |

## D. qmoe

- decode: 40 calls, 5.5 ms/Compute, 12.1% of decode GPU time.
- Sparsity realized (only k=8 active experts read); see prefill for the bigger qmoe cost.

## E. Launch critical-path A/B (p128 decode)

- EP launch_gap (wall-gpu) over 505 decode Computes: median 0% of wall is host launch/dispatch not hidden by GPU (range 0-34%).
  => GPU-compute-bound -> prioritize kernel efficiency over fusion.

| mode | decode tok/s | decode ms/tok |
|---|---|---|
| normal (async) | 41.9 | 23.86 |
| HIP_LAUNCH_BLOCKING=1 | 11.1 | 90.06 |

- Serializing launches changes decode 23.86 -> 90.06 ms/tok (3.77x). ~66.20 ms/tok of launch overhead is currently HIDDEN by overlap.
- Interpretation: launch overhead is mostly OVERLAPPED behind compute -> fusion yields little decode TPS; focus on the big compute kernels.

## F. Prefill / TTFT

- TTFT vs prompt (from section A): p128=0.65s, p512=1.59s, p2048=4.72s

- Prefill Compute per-op GPU breakdown (total 1564.8 ms):

| op | calls | gpu ms | % prefill |
|---|---|---|---|
| qmoe | 40 | 1285.9 | 82.2% |
| gqa | 10 | 93.2 | 6.0% |
| linear_attention | 30 | 73.4 | 4.7% |
| matmul_nbits | 391 | 59.9 | 3.8% |
| elementwise | 440 | 11.5 | 0.7% |
| transpose | 100 | 7.7 | 0.5% |
| layernorm | 51 | 7.3 | 0.5% |
| cast | 192 | 5.8 | 0.4% |

## G. Cold start

- Minimal-run (l=8,g=1) process wall: 28.3 s (model load + compile + weight upload + first-inference autotune, load-dominated).
- EP phase timers (measured, summed over cold-start events):
  - autotune: 10.036 s (n=32)
  - weight_upload: 3.568 s (n=3)

## (aux) Isolated per-shape kernel time (single_op seq1, gs128 graphs)

| op graph | shape | calls | gpu ms |
|---|---|---|---|
| MatMulNBits_down_proj | m=1,n=2048,k=512 | 1 | 0.000 |
| MatMulNBits_gate_proj | m=1,n=512,k=2048 | 1 | 0.000 |
| MatMulNBits_in_proj_b | m=1,n=32,k=2048 | 1 | 0.100 |
| MatMulNBits_in_proj_qkv | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_in_proj_z | m=1,n=4096,k=2048 | 1 | 0.100 |
| MatMulNBits_lm_head | m=1,n=248320,k=2048 | 1 | 1.900 |
| MatMulNBits_out_proj | m=1,n=2048,k=4096 | 1 | 0.000 |
| MatMulNBits_o_proj | m=1,n=2048,k=4096 | 1 | 0.100 |
| MatMulNBits_q_proj | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_router | m=1,n=256,k=2048 | 1 | 0.000 |
| MatMulNBits_shared_expert_gate | m=1,n=1,k=2048 | 1 | 0.000 |
| QMoE | 1x2048x512,e=256 | 1 | 0.000 |

## H. Ranked decode bottlenecks (by GPU-time share)

1. **matmul_nbits** -- 34.4% of decode (28% peak). Split into medium-N GEMV kernel efficiency (headroom) + small-N launch/batching (overhead-bound); see section C.
2. **qmoe** -- 12.1% of decode (41% peak). Grouped-expert GEMM (biggest prefill/TTFT lever); decode kernel BW tuning.
3. **elementwise** -- 10.8% of decode (1% peak). Investigate per section E (launch-bound?) and roofline % above.
4. **linear_attention** -- 7.7% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
5. **activation** -- 7.2% of decode (n/a). Small-op tail -- fuse only if section E shows launches on the critical path.
6. **cast** -- 6.2% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
