# Bottleneck report -- Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml

Probe dir: `probe_20260705_220455`  |  roofline peak: 256 GB/s  |  gen=128 reps=5 warmup=2

> Phase 1 report: per-op GB/s and % peak are shape-model ESTIMATES (marked est). Phase 2 replaces them with EP-measured bytes.

## A. Headline (model_benchmark)

| prompt | TTFT (s) | prefill tok/s | decode tok/s | decode ms/tok | peak WS (GB) | kernel errors |
|---|---|---|---|---|---|---|
| 128 | 0.583 | 220 | 41.5 | 24.07 | 21.1 | False |
| 512 | 1.521 | 337 | 42.0 | 23.78 | 22.4 | False |
| 2048 | 6.823 | 300 | 37.2 | 26.84 | 27.7 | False |

## B. Decode per-op roofline (steady token; total GPU 49.4 ms/Compute)

| op | calls | gpu ms | % decode | bytes (MB, est) | GB/s (est) | % peak (est) |
|---|---|---|---|---|---|---|
| matmul_nbits | 391 | 18.3 | 37.1% | 1094.0 | 60 | 23% |
| qmoe | 40 | 5.5 | 11.2% | 566.6 | 102 | 40% |
| elementwise | 440 | 5.1 | 10.3% | 6.3 | 1 | - |
| linear_attention | 30 | 3.6 | 7.4% | - | - | - |
| activation | 180 | 3.2 | 6.4% | 1.2 | - | - |
| cast | 192 | 2.6 | 5.2% | 2.8 | 1 | - |
| transpose | 100 | 2.2 | 4.5% | 2.3 | 1 | - |
| skip_layernorm | 80 | 2.0 | 4.1% | 1.3 | 1 | - |
| layernorm | 51 | 1.9 | 3.9% | 0.7 | - | - |
| power | 120 | 1.9 | 3.8% | - | - | - |
| gqa | 10 | 1.5 | 2.9% | - | - | - |
| reduce_sum | 61 | 0.7 | 1.4% | 0.2 | - | - |
| causal_conv | 30 | 0.4 | 0.8% | - | - | - |
| rotary_emb | 20 | 0.3 | 0.5% | - | - | - |
| gather | 9 | 0.2 | 0.4% | - | - | - |
| sub | 1 | 0.0 | 0.1% | - | - | - |

- Top-down aggregate memory roofline (profiler-independent): ~1.68 GB/token (byte-model, ~56% of ops covered) x 41.5 tok/s = ~70 GB/s achieved = ~27% of 256 GB/s peak.
  Memory-roofline decode ceiling ~= 153 tok/s (peak/bytes-per-token); measured 41.5 tok/s.

## C. matmul_nbits per-shape roofline (est)

| shape | calls | gpu ms | % decode | bits | bytes (MB) | GB/s | % peak | class |
|---|---|---|---|---|---|---|---|---|
| m=1,n=8192,k=2048 | 40 | 3.8 | 7.7% | 8 | 378.3 | 99 | 39% | headroom |
| m=1,n=512,k=2048 | 100 | 3.1 | 6.2% | 4 | 59.5 | 19 | 8% | overhead-bound |
| m=1,n=4096,k=2048 | 30 | 2.1 | 4.3% | 8 | 141.9 | 67 | 26% | headroom |
| m=1,n=248320,k=2048 | 1 | 1.9 | 3.8% | 4 | 286.6 | 153 | 60% | near-roofline |
| m=1,n=256,k=2048 | 40 | 1.8 | 3.6% | 4 | 12.0 | 7 | 3% | overhead-bound |
| m=1,n=2048,k=4096 | 40 | 1.7 | 3.5% | 8 | 189.2 | 111 | 43% | headroom |
| m=1,n=32,k=2048 | 60 | 1.5 | 3.1% | 8 | 2.5 | 2 | 1% | overhead-bound |
| m=1,n=1,k=2048 | 40 | 1.2 | 2.5% | 4 | 0.2 | 0 | 0% | overhead-bound |
| m=1,n=2048,k=512 | 40 | 1.2 | 2.4% | 4 | 23.8 | 20 | 8% | overhead-bound |

## D. qmoe

- decode: 40 calls, 5.5 ms/Compute, 11.2% of decode GPU time.
- Sparsity realized (only k=8 active experts read); see prefill for the bigger qmoe cost.

## E. Launch critical-path A/B (p128 decode)

- EP launch_gap (wall-gpu) over 505 decode Computes: median 1% of wall is host launch/dispatch not hidden by GPU (range 0-39%).
  => GPU-compute-bound -> prioritize kernel efficiency over fusion.

| mode | decode tok/s | decode ms/tok |
|---|---|---|
| normal (async) | 38.8 | 25.79 |
| HIP_LAUNCH_BLOCKING=1 | 10.7 | 93.54 |

- Serializing launches changes decode 25.79 -> 93.54 ms/tok (3.63x). ~67.75 ms/tok of launch overhead is currently HIDDEN by overlap.
- Interpretation: launch overhead is mostly OVERLAPPED behind compute -> fusion yields little decode TPS; focus on the big compute kernels.

## F. Prefill / TTFT

- TTFT vs prompt (from section A): p128=0.58s, p512=1.52s, p2048=6.82s

- Prefill Compute per-op GPU breakdown (total 1707.2 ms):

| op | calls | gpu ms | % prefill |
|---|---|---|---|
| qmoe | 40 | 1409.6 | 82.6% |
| gqa | 10 | 110.3 | 6.5% |
| linear_attention | 30 | 72.3 | 4.2% |
| matmul_nbits | 391 | 61.2 | 3.6% |
| elementwise | 440 | 12.9 | 0.8% |
| transpose | 100 | 8.9 | 0.5% |
| activation | 180 | 6.1 | 0.4% |
| cast | 192 | 6.0 | 0.4% |

## G. Cold start

- Minimal-run (l=8,g=1) process wall: 28.2 s (model load + compile + weight upload + first-inference autotune, load-dominated).
- EP phase timers (measured, summed over cold-start events):
  - autotune: 9.991 s (n=32)
  - weight_upload: 3.632 s (n=3)

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
| MatMulNBits_o_proj | m=1,n=2048,k=4096 | 1 | 0.000 |
| MatMulNBits_q_proj | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_router | m=1,n=256,k=2048 | 1 | 0.000 |
| MatMulNBits_shared_expert_gate | m=1,n=1,k=2048 | 1 | 0.000 |
| QMoE | 1x2048x512,e=256 | 1 | 0.100 |

## H. Ranked decode bottlenecks (by GPU-time share)

1. **matmul_nbits** -- 37.1% of decode (23% peak). Split into medium-N GEMV kernel efficiency (headroom) + small-N launch/batching (overhead-bound); see section C.
2. **qmoe** -- 11.2% of decode (40% peak). Grouped-expert GEMM (biggest prefill/TTFT lever); decode kernel BW tuning.
3. **elementwise** -- 10.3% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
4. **linear_attention** -- 7.4% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
5. **activation** -- 6.4% of decode (n/a). Small-op tail -- fuse only if section E shows launches on the critical path.
6. **cast** -- 5.2% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
