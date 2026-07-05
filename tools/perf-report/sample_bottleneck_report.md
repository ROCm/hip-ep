# Bottleneck report -- Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml

Probe dir: `probe_20260705_095129`  |  roofline peak: 256 GB/s  |  gen=128 reps=5 warmup=2

> Phase 1 report: per-op GB/s and % peak are shape-model ESTIMATES (marked est). Phase 2 replaces them with EP-measured bytes.

## A. Headline (model_benchmark)

| prompt | TTFT (s) | prefill tok/s | decode tok/s | decode ms/tok | peak WS (GB) | kernel errors |
|---|---|---|---|---|---|---|
| 128 | 0.653 | 196 | 36.9 | 27.12 | 21.0 | False |
| 512 | 1.633 | 313 | 40.3 | 24.83 | 23.3 | False |
| 2048 | 5.602 | 366 | 34.1 | 29.31 | 27.8 | False |

## B. Decode per-op roofline (steady token; total GPU 48.7 ms/Compute)

| op | calls | gpu ms | % decode | bytes (MB, est) | GB/s (est) | % peak (est) |
|---|---|---|---|---|---|---|
| matmul_nbits | 391 | 18.0 | 36.9% | 1725.1 | 96 | 37% |
| qmoe | 40 | 5.5 | 11.2% | - | - | - |
| elementwise | 440 | 4.8 | 9.9% | - | - | - |
| linear_attention | 30 | 3.6 | 7.4% | - | - | - |
| activation | 180 | 3.2 | 6.6% | - | - | - |
| cast | 192 | 2.7 | 5.6% | - | - | - |
| skip_layernorm | 80 | 2.3 | 4.7% | - | - | - |
| transpose | 100 | 2.2 | 4.6% | - | - | - |
| layernorm | 51 | 1.8 | 3.7% | - | - | - |
| power | 120 | 1.6 | 3.3% | - | - | - |
| gqa | 10 | 1.4 | 2.9% | - | - | - |
| reduce_sum | 61 | 0.6 | 1.3% | - | - | - |
| causal_conv | 30 | 0.5 | 1.0% | - | - | - |
| rotary_emb | 20 | 0.3 | 0.6% | - | - | - |
| gather | 9 | 0.1 | 0.3% | - | - | - |
| sub | 1 | 0.0 | 0.0% | - | - | - |

## C. matmul_nbits per-shape roofline (est)

| shape | calls | gpu ms | % decode | bits | bytes (MB) | GB/s | % peak | class |
|---|---|---|---|---|---|---|---|---|
| m=1,n=8192,k=2048 | 40 | 3.8 | 7.8% | 8 | 713.9 | 188 | 73% | near-roofline |
| m=1,n=512,k=2048 | 100 | 3.6 | 7.3% | 4 | 59.5 | 17 | 6% | overhead-bound |
| m=1,n=2048,k=4096 | 40 | 2.8 | 5.7% | 8 | 357.0 | 128 | 50% | near-roofline |
| m=1,n=4096,k=2048 | 30 | 2.1 | 4.4% | 8 | 267.8 | 128 | 50% | near-roofline |
| m=1,n=248320,k=2048 | 1 | 2.0 | 4.2% | 4 | 286.6 | 143 | 56% | near-roofline |
| m=1,n=32,k=2048 | 60 | 1.3 | 2.7% | 8 | 4.4 | 3 | 1% | overhead-bound |
| m=1,n=256,k=2048 | 40 | 1.1 | 2.2% | 4 | 12.0 | 11 | 4% | overhead-bound |
| m=1,n=1,k=2048 | 40 | 0.7 | 1.4% | 4 | 0.2 | 0 | 0% | overhead-bound |
| m=1,n=2048,k=512 | 40 | 0.6 | 1.2% | 4 | 23.8 | 40 | 15% | headroom |

## D. qmoe

- decode: 40 calls, 5.5 ms/Compute, 11.2% of decode GPU time.
- Sparsity realized (only k=8 active experts read); see prefill for the bigger qmoe cost.

## E. Launch critical-path A/B (p128 decode)

| mode | decode tok/s | decode ms/tok |
|---|---|---|
| normal (async) | 40.7 | 24.55 |
| HIP_LAUNCH_BLOCKING=1 | 11.3 | 88.74 |

- Serializing launches changes decode 24.55 -> 88.74 ms/tok (3.61x). ~64.19 ms/tok of launch overhead is currently HIDDEN by overlap.
- Interpretation: launch overhead is mostly OVERLAPPED behind compute -> fusion yields little decode TPS; focus on the big compute kernels.

## F. Prefill / TTFT

- TTFT vs prompt (from section A): p128=0.65s, p512=1.63s, p2048=5.60s
- Long-prompt TTFT is dominated by qmoe (most experts activated) + the huge-vocab lm_head. Per-op prefill attribution needs a prefill-phase [PERF] capture (Phase 2).

## G. Cold start

- Minimal-run (l=8,g=1) process wall: 32.5 s (model load + compile + weight upload + first-inference autotune, load-dominated).
- (No [COLDSTART] phase lines found -- rebuild with Phase 2 instrumentation.)

## (aux) Isolated per-shape kernel time (single_op seq1, gs128 graphs)

| op graph | shape | calls | gpu ms |
|---|---|---|---|
| MatMulNBits_down_proj | m=1,n=2048,k=512 | 1 | 0.000 |
| MatMulNBits_gate_proj | m=1,n=512,k=2048 | 1 | 0.000 |
| MatMulNBits_in_proj_b | m=1,n=32,k=2048 | 1 | 0.000 |
| MatMulNBits_in_proj_qkv | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_in_proj_z | m=1,n=4096,k=2048 | 1 | 0.100 |
| MatMulNBits_lm_head | m=1,n=248320,k=2048 | 1 | 1.800 |
| MatMulNBits_out_proj | m=1,n=2048,k=4096 | 1 | 0.000 |
| MatMulNBits_o_proj | m=1,n=2048,k=4096 | 1 | 0.100 |
| MatMulNBits_q_proj | m=1,n=8192,k=2048 | 1 | 0.100 |
| MatMulNBits_router | m=1,n=256,k=2048 | 1 | 0.000 |
| MatMulNBits_shared_expert_gate | m=1,n=1,k=2048 | 1 | 0.000 |
| QMoE | 1x2048x512,e=256 | 1 | 0.000 |

## H. Ranked decode bottlenecks (by GPU-time share)

1. **matmul_nbits** -- 36.9% of decode (37% peak). Split into medium-N GEMV kernel efficiency (headroom) + small-N launch/batching (overhead-bound); see section C.
2. **qmoe** -- 11.2% of decode (n/a). Grouped-expert GEMM (biggest prefill/TTFT lever); decode kernel BW tuning.
3. **elementwise** -- 9.9% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
4. **linear_attention** -- 7.4% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
5. **activation** -- 6.6% of decode (n/a). Small-op tail -- fuse only if section E shows launches on the critical path.
6. **cast** -- 5.6% of decode (n/a). Investigate per section E (launch-bound?) and roofline % above.
