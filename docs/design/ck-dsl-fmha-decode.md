<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ck_dsl single-pass GQA attention decode

## Why this exists

`CLAUDE.md` Future Improvements #1 (line 503):

> **Fused single-pass attention decode** that eliminates the reduce kernel —
> would unblock the L=256+ cliff and remove the K_SPLITS guesswork entirely.
> High effort, high impact.

This document is the empirical case for an externally-authored ck_dsl
kernel as that improvement, scoped to the *short-context* (skv < 256) regime
that today routes to `hip_gqa_fused_decode`. A long-context split-K
follow-on is left for a separate PR.

## TL;DR

On gfx1151 (Strix Halo, 16 CUs), at the Mistral 7B GQA shape (B=1, H=32,
G=8, d=128, sq=1, fp16), the ck_dsl `wmma_fmha_fwd` kernel run as a
decode kernel (Q broadcast across the 16-row WMMA M-tile) **beats
`hip_gqa_fused_decode` by 2.3-2.8× at skv ∈ {64, 128, 192}** — exactly the
skv range the EP currently routes to `fused_decode`.

Estimated end-to-end gain on the verified Mistral 7B AWQ b128 baseline
(`CLAUDE.md:482`: **48.2 tok/s** decode at L=128): **~52.8 tok/s, +9.5%**.

## Architecture

```
                    sq=1 GQA decode (current path)                                            sq=1 GQA decode (this PR, opt-in)

   ┌─────────────────────────────────────┐         ┌────────────────────────────────────────────────┐
   │ skv < 256  : hip_gqa_fused_decode    │         │ skv < 256  : ck_dsl wmma_fmha_fwd              │
   │   - cooperative dot product          │         │   - WMMA 16x16x16 single-pass                  │
   │   - D threads per CTA (= head_dim)   │         │   - one wave32 per (batch, query head)         │
   │   - online softmax via wave shfl     │         │   - Q broadcast (stride_q_token = 0)            │
   │   - one CTA per (B, H)               │         │   - O broadcast (stride_o_token = 0)            │
   │                                      │         │   - no extra scratch, no extra kernel launch    │
   ├──────────────────────────────────────┤         ├────────────────────────────────────────────────┤
   │ skv ≥ 256 : hip_gqa_flash_decode     │   ←─    │ skv ≥ 256 : hip_gqa_flash_decode (unchanged)   │
   │   - FA-2 split-K (K_SPLITS = 8)      │         │   - this PR does not touch this path           │
   │   - 8 CTAs per (B, H) + reduce kernel│         │                                                │
   └──────────────────────────────────────┘         └────────────────────────────────────────────────┘
```

The HSACO is generated from
[ck_dsl `instances/gfx1151/wmma_fmha_fwd.py`](https://github.com/ROCm/rocm-libraries/blob/users/vanantha/ck-dsl-prototype/projects/composablekernel/python/ck_dsl/instances/gfx1151/wmma_fmha_fwd.py)
with spec
`WmmaFmhaFwdSpec(head_size=128, num_query_heads=32, num_kv_heads=8, mask_mode="none")`
and checked into
[`3rd-party/ck_dsl_kernels/gfx1151/wmma_fmha_decode.hsaco`](../../3rd-party/ck_dsl_kernels/gfx1151/wmma_fmha_decode.hsaco)
(13 KB). The wrapper
[`lib/Runtime/real/ck_dsl_fmha_decode.cpp`](../../lib/Runtime/real/ck_dsl_fmha_decode.cpp)
embeds the HSACO bytes inline (`ck_dsl_fmha_decode_gfx1151_hsaco.h`),
calls `hipModuleLoadData` once per process, and exports
`ck_dsl_gqa_fmha_decode(...)` with the same C ABI as `hip_gqa_fused_decode`.

Dispatch hook in
[`lib/Runtime/real/gqa.cpp`](../../lib/Runtime/real/gqa.cpp): the existing
`fused_decode` `else` branch checks `HIPDNN_EP_CK_DSL_FMHA_DECODE`; when set,
the shim is called first and the baseline `hip_gqa_fused_decode` is the
fallback (the shim returns `-2` when the runtime spec doesn't match its
embedded HSACO — e.g. wrong arch, wrong (H,G,d), B≠1).

## The bandwidth insight that makes Q-broadcast free

Initial intuition says WMMA can't beat cooperative-dot-product GQA decode at
sq=1 because the 16-row WMMA M-tile means 16× compute waste. That intuition
is wrong at decode shapes:

```
Mistral 7B GQA decode @ skv=2048:
  KV reads (per CTA):   skv × d × 2 bytes × 2 tensors    = 1 MB
  FMAs    (per CTA):    skv × d × 2 GEMMs                = 1 MFMA
  Arithmetic intensity:  ~1 FMA / byte
  gfx1151 peak:          10 TFLOPS / ~250 GB/s            = ~40 FMA / byte

  -> we are 40× away from the compute limit. Multiplying compute by 16×
     (the M-padding cost) brings intensity to ~16 FMA/byte. Still memory-
     bound. Q-broadcast is essentially free at decode shapes; WMMA's
     per-instruction throughput then beats the baseline's many-shfl-cycle
     cooperative reductions by 2-3×.
```

The wrapper exploits this directly: `stride_q_token = 0` means all 16
"rows" of Q read the same memory; `stride_o_token = 0` means all 16 output
rows write to the same slot (every write produces the same value because Q
is broadcast). No scratch buffer, no Q replication kernel, no extra HIP
launches — the shim adds one `hipModuleLaunchKernel` call where the
baseline did one.

## Measured numbers

Mistral 7B GQA decode shape: **B=1, H=32, G=8, d=128, sq=1**. Hardware:
gfx1151 (Strix Halo, 16 CUs), TheRock dist ROCm 7.11. Mean of 100 iters
after 5 warmups, HIP events for timing.

Correctness (ck_dsl side): max_abs_diff ≤ 1.4e-3 vs numpy fp32 GQA
reference, bad_count = 0 / 4096 across every skv tested. Tolerance is the
standard 1 fp16 ULP threshold used by ck_dsl WMMA verify
(`examples/common/universal_gemm_verify`).

| skv | EP baseline path | EP baseline ms | ck_dsl shim ms | Speedup (>1 = ck_dsl win) |
|---:|---|---:|---:|---:|
| 64 | fused_decode | 0.0529 | **0.0234** | **2.26×** |
| 128 | fused_decode | 0.0883 | **0.0329** | **2.68×** |
| 192 | fused_decode | 0.1266 | **0.0454** | **2.79×** |
| 256 | flash_decode (K_SPLITS=8) | 0.0301 | 0.0580 | 0.52× (lose) |
| 384 | flash_decode (K_SPLITS=8) | 0.0364 | 0.0923 | 0.39× (lose) |
| 512 | flash_decode (K_SPLITS=8) | 0.0394 | 0.1165 | 0.34× (lose) |
| 1024 | flash_decode (K_SPLITS=8) | 0.0645 | 0.2308 | 0.28× (lose) |
| 2048 | flash_decode (K_SPLITS=8) | 0.1291 | 0.4723 | 0.27× (lose) |

The crossover at skv=256 is exactly the EP's `kFlashDecodeMinSkv=256`
dispatch boundary. Above 256 the EP correctly switches paths and ck_dsl
loses because it has no split-K; below, ck_dsl is uniformly 2.3-2.8×
faster. The PR scope is the win regime; the loss regime is unchanged.

## Estimated Mistral 7B AWQ b128 decode TPS impact

From `CLAUDE.md:482` verified perf snapshot. Per-token: 32 attention layers,
each at skv ≈ L during decode.

| L | Per-layer path (with this PR) | attn ms / token (baseline) | attn ms / token (with ck_dsl) | Saved | Old TPS | New TPS |
|---:|---|---:|---:|---:|---:|---:|
| 128 | fused_decode → ck_dsl shim | 0.088 × 32 = 2.82 | 0.033 × 32 = 1.05 | **1.77 ms** | 48.2 | **≈ 52.8 (+9.5%)** |
| 2048 | flash_decode (unchanged) | 0.129 × 32 = 4.13 | 0.129 × 32 = 4.13 | 0 ms | 42.6 | 42.6 (no change) |

The +9.5% at L=128 should generalize across the AWQ-quantized model family
that shares the dispatch heuristic (Llama-3.1-8B AWQ, Llama-3.1-8B asym,
etc. per the `flash_decode coverage by CI model` table at CLAUDE.md:459).
Quantitative impact on each model needs `model_benchmark.exe` re-runs.

## Caveats / what this PR does not address

- **Long-context (skv ≥ 256) is unchanged.** The "L=256+ TPS cliff" known
  limitation (CLAUDE.md:495) is not addressed. Closing it requires a ck_dsl
  split-K variant; estimated 4-8 hours of ck_dsl-side work, separate PR.
- **gfx1151 only.** The embedded HSACO is compiled for gfx1151. The shim
  rejects non-gfx1151 arches by returning -2 (fallback); the baseline path
  runs unchanged. Adding gfx1100/gfx1101/gfx1102 is a rebuild + a multi-arch
  dispatcher in the shim.
- **(H, G, d) = (32, 8, 128) only.** The embedded HSACO is compiled for the
  Mistral/Llama-3.x AWQ shape. Other shape combinations get the same -2
  fallback. Adding Phi-4 (40,10,128), Qwen2.5 (40,8,128), DeepSeek (64,8,128),
  etc. is one rebuilt HSACO per shape.
- **Opt-in only (`HIPDNN_EP_CK_DSL_FMHA_DECODE=1`).** Default routing is
  unchanged. After review + sanity benchmark on the EP team's CI, we can
  flip the default to `1`.
- **Build-time HSACO embed via xxd-style header is a one-shot script
  (`scripts/embed_hsaco.py`) rather than a CMake `add_custom_command`.**
  Trivial follow-up.
- **No CI tests added in this PR.** Hooking into the existing `test_gqa`
  and `model_benchmark`-based regression suite is the right place for
  validation but would inflate the PR scope; happy to add in a follow-up.

## Reproduction

ck_dsl side (Python, on a gfx1151 box with TheRock dist ROCm):

```bash
git clone https://github.com/ROCm/rocm-libraries -b users/vanantha/ck-dsl-prototype
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/composablekernel/python/ck_dsl
git checkout
export PYTHONPATH=$PWD/projects/composablekernel/python
export HIP_PATH=/path/to/therock-dist   # or %HIP_PATH% on Windows
python <PHASE_D_SCRIPT>   # see chat: phase_d_fmha_decode.py
```

Baseline side (hipcc):

```bash
hipcc bench_gqa_decode.cpp \
      3rd-party/custom_kernels/hip/gqa_kernel.hip \
      -I 3rd-party/custom_kernels/include \
      -o bench_gqa_decode.exe --offload-arch=gfx1151 -O3 -std=c++17

for s in 64 128 192 256 384 512 1024 2048; do
    ./bench_gqa_decode $s 5 100
done
```

The exact harness sources used to produce the table above are in the chat
attachments; happy to add them as repo test programs in a follow-up.
