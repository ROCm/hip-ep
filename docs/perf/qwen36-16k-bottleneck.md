<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen3.6-35B-A3B 16K prefill: where the time actually goes

SQTT (RGP) profile of a 16K VLM prefill, taken on repaired model weights. This
supersedes the withdrawn attribution in `qwen36-ttft-attribution.md`, which was
measured against a half-zeroed `text.onnx.data`.

## Operating point and build

`vlm_benchmark.py` with `dog.jpg` and `prompt_16k.txt`: **18,646 prompt tokens**
(16,823 text, 1,823 image), `max_length 33024`, `-e AMDGPU`. Build is
`feat/rgp-capture-fence-v2` with all three binaries redeployed together
(`hipgpu.dll`, `custom_kernels_gfx1151.dll`, `hip-compiler.dll`).

Uninstrumented TTFT at this point is **19,526 ms** (3 reps, stddev 293 ms)
against CI's 16,702 ms at 18,755 tokens. The attribution below is all measured at
that baseline; the v8 prefill kernel has since been optimised.

> **The 17,587 ms post-optimisation figure quoted below is not reproducible and
> should not be used.** The same binaries and the same `max_length 33024` now
> measure **14,734 ms**, and 14,455-14,610 ms with a right-sized KV cache. See
> "Second profiling round" for the current baseline and for why that number was
> probably instrumented.

Fitting the 4K and 16K points for each environment separates startup from
throughput:

| | fixed | per token |
|---|---:|---:|
| local, as first measured | 634 ms | 1.013 ms |
| local, warm and verified clean | 940 ms | **0.729 ms** |
| CI | 1,016 ms | 0.836 ms |

The first local row was fitted on runs that are now believed to be instrumented.
On the clean re-measurement local starts up slower than CI but is 13% *faster*
per token, so on these figures the per-token gap to CI has closed and reversed.

## The layer stack

Op call counts per prefill: `qmoe` 40, `matmul_nbits` 391, `causal_conv` 30,
`linear_attention` 30, `gqa` 10. That is a 40-layer hybrid stack: 30
linear-attention layers and 10 full-attention layers, every layer with its own
MoE block.

Each capture window contains exactly one `topk_routing` and one
`bucket_tokens`, i.e. exactly one MoE block, which confirms one window equals
one layer and lets per-layer cost be scaled by layer count.

## Per-layer cost

**Full-attention layer — 463.8 ms GPU busy, 943 dispatches, 0.5% idle**

| kernel | count | total ms | % layer | occ | limiter | bound |
|---|---:|---:|---:|---:|---|---|
| `gqa_flash_prefill_v8_kernel` | 1 | 386.6 | 83.3 | 25% | LDS | latency/low-occupancy |
<!-- The v8 row is the pre-optimisation measurement. It has since been cut to
     203.9 ms/call uninstrumented; see "Optimising the v8 prefill kernel" below,
     which also explains why its *instrumented* cost barely moved. -->

| `MatMulNBitsFp16GEMM` | 61 | 29.1 | 6.3 | 55% | VGPR | undetermined |
| `MatMulNBitsWMMA_NoZP` | 189 | 15.1 | 3.3 | 69% | VGPR | undetermined |
| `bucket_tokens_kernel` | 1 | 8.9 | 1.9 | 100% | - | undetermined |

**Linear-attention layer — 122.2 ms GPU busy, 490 dispatches, 1.4% idle**

| kernel | count | total ms | % layer |
|---|---:|---:|---:|
| `la_pf_pass3_wmma` | 37 | 30.8 | 25.2 |
| `MatMulNBitsFp16GEMM` | 24 | 20.4 | 16.7 |
| `la_pf_scan` | 37 | 16.0 | 13.1 |
| `la_pf_pass1_local` | 37 | 10.4 | 8.5 |
| `T5LayernormFwdContiguous` | 2 | 9.4 | 7.7 |
| `bucket_tokens_kernel` | 1 | 8.8 | 7.2 |
| `ew_bcast4d_kernel` | 1 | 5.2 | 4.3 |
| `cast_f16_to_f32` + `cast_f32_to_f16` | 2 | 5.0 | 4.1 |

The 37 repeats of each `la_pf_*` pass are the chunked scan over 18,646 tokens,
so all 37 belong to a single `linear_attention` op.

## Whole-prefill budget

Scaling measured per-call cost by call count:

| item | per call | calls | total ms |
|---|---:|---:|---:|
| `gqa_flash_prefill_v8` | 386.6 ms | 10 | 3,866 |
| `gqa_flash_prefill_v8`, after optimisation | 203.9 ms | 10 | 2,039 |
| linear attention (3 passes) | 57.2 ms | 30 | 1,716 |
| dense `MatMulNBits` (all variants) | - | - | ~1,050 |
| `bucket_tokens` | 8.8 ms | 40 | 352 |
| `topk_routing` | 1.2 ms | 40 | 48 |
| **10 full-attention layers, whole layer** | 463.8 ms | 10 | **4,638** |
| **30 linear-attention layers, whole layer** | 122.2 ms | 30 | **3,666** |

Modelled decoder GPU busy is therefore about **8,304 ms**, against a 19,526 ms
TTFT. The captures cover the decoder only; the remainder is the vision encoder,
the embedding model, `lm_head` and host time, none of which were captured. Do
not read the 8,304 ms as a complete TTFT decomposition.

> Superseded by "Second profiling round", which measures all three sessions. The
> vision encoder is 1.2 s, the embedding model 78 ms, and `lm_head` is not a cost
> at all: `text.onnx` already emits `logits` as `[batch, 1, 248320]`, so it is
> last-row-only. Host time and idle gaps together are under 20 ms.

## Conclusions

**The prefill is GPU-bound, not launch-bound.** Idle is 0.5% over 943
dispatches in the full-attention window and 1.4% over 490 in the
linear-attention window, with total launch overhead of 33.5 us and 9.9 us
respectively. The earlier claim that ~85% of TTFT was host-side kernel-launch
overhead does not survive contact with SQTT on a correct model.

**The single largest kernel is `gqa_flash_prefill_v8_kernel`** at 386.6 ms per
call and 3.9 s across the ten full-attention layers. It is the only kernel the
parser classes as `latency/low-occupancy`: 25% occupancy with **LDS as the
binding resource**. Its cost is cleanly quadratic in sequence length (17.0
ms/call at 3,985 tokens against 361 ms/call at 18,646, a 21.2x rise for a 21.9x
rise in n^2), and the runtime's own uninstrumented autotune independently
measures 361.4 ms/call for the winning config, so the SQTT figure is not an
instrumentation artifact.

This kernel has since been cut to 203.9 ms/call, taking TTFT to 17,587 ms. The
`latency/low-occupancy` classification turned out to be a red herring: raising
occupancy makes it slower, and the win came from removing redundant per-wave work.
Note also that once `BKV=16` won, the SQTT per-call figure *did* become an
instrumentation artifact, for the reason given in that section.

**The most egregious inefficiency is `bucket_tokens_kernel`.** It launches a
*single* 256-thread workgroup, 8 wavefronts, and holds the GPU for 8.8 ms:

```
bucket_tokens_kernel   threads: 256   workgroup: 256   wavefronts: 8      dur: 8,822.7 us
topk_routing_kernel    threads: 4,773,376        wavefronts: 149,168      dur: 1,227.7 us
```

Every other kernel in the MoE block is massively parallel; this one serialises
the whole device. At 40 layers it is ~350 ms, and it is followed by the largest
idle gap in either capture (1.0 ms, and 4.0 ms in the full-attention window).
`origin/perf/qwen-bucket-tokens` already carries a chunked rewrite of exactly
this kernel and is the obvious thing to re-measure first.

**Ten full-attention layers cost more than thirty linear-attention layers**
(4,638 ms against 3,666 ms), so attention-shape work, not layer count, sets the
scaling.

## Optimising the v8 prefill kernel

The kernel above was then optimised directly. Result at the same operating point:

| | before | after |
|---|---:|---:|
| per call, uninstrumented (runtime autotune, sq=18646) | 361.4 ms | **203.9 ms** |
| per call, standalone microbenchmark | 348.1 ms | **203.4 ms** |
| per call, SQTT-instrumented | 386.6 ms | 380.8 ms |
| config (ND, MT, BKV) | 4, 1, 32 | 2, 1, 16 |
| VGPRs / LDS bytes | 192 / 26112 | 240 / 11264 |
| occupancy / limiter | 25.0% / LDS | 31.2% / LDS |
| workgroup / wavefronts | 128 / 74,624 | 64 / 37,312 |
| **16K TTFT** (3 reps) | **19,526 ms** (sd 293) | **17,587 ms** (sd 286) |

**43% off the kernel and 1,939 ms (9.9%) off TTFT.** Accuracy improved at the same
time: `relL2` against the CPU reference went from 5.7e-04 to 3.3e-04 at sq=2048.

### Do not use the SQTT per-call time to A/B this kernel

The instrumented cost moved 1.5% while the uninstrumented cost fell 43%. The
instrumented figure is the artifact: SQTT overhead scales with instruction issue,
and the winning config uses `BKV=16`, which doubles the KV-tile trip count (583
tiles per block instead of 292) for half the work per tile. Three independent
uninstrumented measurements agree with each other and not with SQTT — the
standalone microbenchmark, the runtime's own autotune inside the real process, and
end-to-end TTFT. A 5.8 ms/call improvement over 10 calls cannot produce a 1,939 ms
TTFT drop, so SQTT is what is wrong here, not the other three.

### Occupancy was the wrong target

The premise going in was that 25% occupancy with LDS as the binding resource was
the problem. It was not. Measured at sq=18646, with compiler register/LDS figures
from `-Rpass-analysis=kernel-resource-usage`:

| cfg | vgpr | spill | occupancy | ms/call |
|---|---:|---:|---:|---:|
| 2,1,16 | 239 | 0 | 37.5% | **204** |
| 4,1,16 | 135 | 0 | 62.5% | 219 |
| 4,1,32 | 175 | 0 | 50.0% | 249 |
| 8,1,16 | 82 | 0 | 100.0% | 276 |

Occupancy is inversely correlated with speed across this whole sweep. The fully
occupant config is 35% slower than the winner, because occupancy here is a
function of `ND`, and raising `ND` adds cross-wave score-reduction traffic and
per-wave softmax cost that outweigh the extra residency. The final kernel is still
LDS-limited at 31.2% and that is fine. Six of the eleven instantiated configs also
spill (up to 768 registers), which no occupancy figure reveals — spilling
configurations had been sitting in the autotune candidate list.

### What actually paid, in order

1. **`BKV=16`** (absent from the candidate table). 348 -> 248 ms. Halves `V_lds`.
2. **Stop repeating the softmax in every wave.** The ablation mask showed softmax
   was 136 ms of 348 ms, and all `ND` waves computed it identically. Splitting its
   rows `ND` ways: 248 -> 215 ms. Handing it to one wave instead was tried first
   and is worse — the other waves then idle at the barrier while that copy sits on
   the critical path.
3. **fp32 accumulator.** The WMMA accumulator is already fp32, so keeping `O` in
   fp16 meant unpacking and repacking every tile. 215 -> 204 ms, and it is the
   accuracy improvement above.
4. **Pruning the candidate list** to non-spilling configs, which also cut the
   first-call autotune sweep from 11 candidates to 4 (~40 s each at this shape).

### What was measured and rejected

- **Reducing the 3 barriers per KV tile.** Ablating all three (mask bit 32) saves
  5.5 ms of 203 ms, so the restructuring the barriers would need cannot pay.
- **Skipping the `O` rescale when `corr == 1.0`**, which is most tiles once the
  running max settles: 0.3%, inside run-to-run noise. Those fp32 multiplies hide
  behind the tile's memory traffic.
- **Compacting `Sp_lds`.** Aimed at buying occupancy headroom, which the table
  above shows is not worth buying. It is also only ~10% of the kernel's LDS
  traffic, against `V_lds`'s 16 KB per tile.
- **`ND=8` with `__launch_bounds__`** to force `vgpr <= 128`: reaches 100%
  occupancy and is the slowest config measured.

### Where the remaining 203 ms sits

By ablation at the shipped config (`HIPDNN_PREFILL_V8_ABLATE`, bit per phase):

| phase | cost | share |
|---|---:|---:|
| value GEMM | 45.8 ms | 23% |
| softmax | 41.1 ms | 20% |
| V staging | 37.2 ms | 18% |
| score GEMM | 19.9 ms | 10% |
| K loads | 13.2 ms | 7% |
| barriers | 5.5 ms | 3% |

No single phase dominates any more, which is why this stopped here. The largest
remaining structural cost is that the value GEMM's `V` fragment needs 16
consecutive `kv` values at a fixed `d`, while the KV cache is `[B,G,seq,d]` — so
each fragment is a 16-instruction strided LDS gather. Storing `V` d-major would
make it two contiguous loads, but that is a KV-cache layout change spanning the
append/concat/decode kernels, not a change to this kernel.

### Reproducing

`test_gqa_prefill.cpp` gained the Qwen d=256 shapes (nothing covered the v8 kernel
before, so it had no correctness test at all) plus `--sq`, `--only`, and
`--dump`/`--ref` for checking a kernel change against a previous build's output at
sq=18646, where a CPU reference is infeasible. Two env knobs were added:
`HIPDNN_PREFILL_V8_CFG="ND,MT,BKV"` pins a config, and
`HIPDNN_PREFILL_V8_ABLATE` drives the in-kernel ablation mask, which previously
had no reachable caller.

## What could not be measured, and why

No bound class is better than `compute-or-memory (undetermined)` for anything
except the GQA kernel, because SPM counters had to be dropped. Per
`tools/rgp_parser/README.md` the SQTT buffer, the SPM counter buffers and the
model's activations all compete for the same physical memory on a shared-memory
APU. With `--rgp-counter-collection` on at 16K the workload slowed so far that
the fence could not arm inside a 10-minute window; without SPM it armed
normally. Separating memory-bound from compute-bound at 16K would need a bigger
memory budget than this box has.

Two further limits found the hard way:

- `--rgp-sqtt-buffer-size minimum` is enough for an MoE block (56 MB, 179
  dispatches) but **not** for a full-attention layer: the 386 ms attention
  kernel's token stream produced `TraceError` chunks and `sqtt=0`, an unusable
  capture. `default` works and yields 180 MB and 943 dispatches. `maximum`
  remains untested here and is documented as wedging the workload.
- The harness's fence-arm deadline was hardcoded to 600 s, which a 16K VLM
  cannot meet: model load, the per-process prefill autotune sweep and a cold
  first rep all precede the fence. It is now the `-ArmTimeoutSec` parameter.

The box hard-faulted once during this work (bugcheck `0x7F`, double fault); an
identical bugcheck is in the event log from earlier the same day, so it is a
recurring failure mode at this operating point rather than something a single
capture caused. Model weights were re-verified against the CI share afterwards
and are intact.

# Second profiling round: the whole TTFT

The first round fenced on one decoder op and modelled the decoder from two
single-layer windows, covering 8,304 ms of a 19,526 ms TTFT. This round measures
all three ONNX sessions and re-establishes the baseline. It changes the answer.

## The baseline was wrong, and the harness cannot prevent that

| tag | max_length | TTFT | within-run sd |
|---|---:|---:|---:|
| `gqa_v8_opt_16k` (first round) | 33,024 | 17,587 ms | 286 ms |
| `p0-16k-33k` (same binaries, same config) | 33,024 | **14,734 ms** | 77 ms |
| `p0-16k-tight` | 18,900 | **14,610 ms** | 15 ms |
| `p0-16k-tight-r2` | 18,900 | **14,455 ms** | 14 ms |

Warm between-run spread is 155 ms (1.1%) and within-run 14 ms, so the 2,853 ms
gap to the first-round number is not noise. Neither is it the kernel: the GQA
autotune measured 223.5, 224.1 and 233.3 ms/call across these runs, and decode
was *faster* in the slow run (19.93 against 20.09 ms/token), which rules out
thermal throttling.

The likely cause is that the first-round run was instrumented. Enabling
`HIPDNN_EP_TRACE_FILE` on the current build measures **17,545 ms**, which matches
the unexplained 17,587 ms to 0.24%.

**`Clear-HarnessProfilingEnv` cannot prevent this.** It removes `HIPDNN_EP_PERF`
and `HIPDNN_EP_DEBUG` only:

```powershell
# tools/perf-harness/common.ps1
Remove-Item Env:HIPDNN_EP_PERF, Env:HIPDNN_EP_DEBUG -EA SilentlyContinue
```

But `hipdnn_ep_perf_enabled()` is true if **either** `HIPDNN_EP_PERF` is set *or*
`HIPDNN_EP_TRACE_FILE` is non-empty (`lib/Runtime/debug_log.h:41-44`). A leaked
`HIPDNN_EP_TRACE_FILE` therefore turns on full instrumentation, costs ~21% of
TTFT, and no harness script clears it. It should be added to that list. Nothing
is set at User or Machine scope on this box, so the leak would have been
per-shell.

## KV-cache oversizing is real but small

`max_length 33024` for an 18,646-token prompt allocates 77% more KV cache than
the prompt needs, and it does show up: peak RSS 1,365 MB against 1,103 MB, a
262 MB difference that matches the KV geometry. But the TTFT cost is only
**124-280 ms** (14,734 against 14,455-14,610), about 1-2%. The hypothesis that
this explained seconds of TTFT is falsified.

## Whole-TTFT session decomposition

`HIPDNN_EP_TRACE_FILE` writes one Chrome trace per session on a shared absolute
`steady_clock`, so all three sessions lie on one timeline. Merged with
`tools/perf-report/merge_chrome_traces.py` (no `--zero`), the warm iteration is:

| bucket | ms | share |
|---|---:|---:|
| decoder session | 16,209.5 | 92.4% |
| vision session | 1,229.9 | 7.0% |
| embedding session | 77.9 | 0.4% |
| inter-session gaps | 15.9 | 0.1% |
| host inside session windows | 3.6 | 0.02% |
| **wall, first start to last end** | **17,533.2** | |

Against a 17,544.7 ms instrumented TTFT, that accounts for all but 11.5 ms
(0.07%). **The prefill is essentially 100% GPU kernel time**: there is no host
bottleneck, no launch-bound region and no idle to recover. This is the same
conclusion the first round reached from idle percentages, now established for the
whole TTFT rather than two windows.

The vision encoder is measured here for the first time. At 1.2 s it is 7% of
TTFT, not the multi-second unknown it was assumed to be. Its cost is dominated by
`mha_flash_prefill` and Tensile GEMMs, and it is still worth noting that the
config fix keeping it on the GPU is load-bearing: with `session_options` present
on the `vision` entry, ORT silently runs all 4,154 of its nodes on
`CPUExecutionProvider` for 29.2 s of node time.

## The EP's own per-op timings are not usable for attribution

The `[PERF]` table is not merely inflated, its **per-op attribution is wrong**.
Checked against RGP windows taken with no EP instrumentation:

| op | `[PERF]` ms/call | RGP ms/call | error |
|---|---:|---:|---|
| vision `gemm` (7296x1152x4304) | 4.43 | 4.61 | agrees |
| vision `gemm` (7296x4304x1152) | 4.23 | 4.23 | agrees |
| `causal_conv` | 30.6 | 31.5 | agrees |
| `gqa` | 27.2 | 380.0 | **14x low** |
| vision `multi_head_attention` | 0.56 | 20.34 | **36x low** |
| `linear_attention` | 8.6 | 56.7 | 6.6x low |
| vision `slice` | 5.68 | 0.34 | **17x high** |
| `cast` (596672) | 46.4 | 1.23 | **38x high** |

`[PERF]` puts vision MHA at 0.56 ms/call when that call is 245 GFLOP, whose floor
at 59.4 TFLOP/s is 4.13 ms — below the hardware limit, so it is provably wrong
rather than merely noisy.

The mechanism is the timing model. GPU time is the gap between consecutive
fenceless event markers on the in-order stream, so an op that blocks the host and
drains the queue is charged for work that was already outstanding. `slice` and
`cast` are exactly that: `slice`'s CPU time equals its GPU time (594 against
614 ms) and the vision session issues 199 `readback_scalar` calls, so those ops
absorb the attention and GEMM work queued ahead of them.

The practical rule: take **call counts and shapes** from `[PERF]`, since those are
plain counters, and take **all timing** from RGP. The "`transpose` is 25.7% and
`cast` is 21.4% of the decoder" reading that this table first produced does not
survive; `cast` is ~74 ms over the whole prefill, not 3,473 ms.

## Ranked by recoverable time

Per-call times from three RGP windows fenced on `multi_head_attention` (vision),
`gqa`, `causal_conv` and `qmoe`, all with `-Buf default`, no SPM and no EP
instrumentation; all four gated through `inspect_capture.py` as steady state.
Floors are `max(bytes/BW, FLOP/peak)` at 256 GB/s and 59.4 TFLOP/s, computed per
shape by `qwen_floors.py` alongside the captures.

| kernel | calls | ms/call | total ms | floor | util | recoverable | confidence |
|---|---:|---:|---:|---:|---:|---:|---|
| `gqa_flash_prefill_v8` | 10 | 224.00 | 2,240 | 47.95 | 21% | 1,761 | upper bound |
| `transpose_tiled` (18646x8192) | 60 | 26.70 | 1,602 | 2.39 | **9%** | **1,459** | high |
| `causal_conv_prefill` | 30 | 31.54 | 946 | 2.39 | **8%** | **875** | high |
| `MatMulNBits` (MoE, all variants) | 40 blk | 32.77 | 1,311 | 15.80 | 48% | 679 | medium |
| `T5LayernormFwdContiguous` | 111 | 5.45 | 605 | 0.90 | 16% | 505 | high |
| vision `mha_flash_prefill` | 27 | 20.34 | 549 | 4.13 | 20% | 438 | upper bound |
| `bucket_tokens_kernel` | 40 | 8.83 | 353 | ~0 | **0%** | **353** | high |
| vision `gemm` (Tensile) | 27 lyr | 14.81 | 400 | 3.74 | 25% | 299 | high |
| `gather_tokens` + `scatter_add` | 40 blk | 9.80 | 392 | 4.77 | 49% | 201 | medium |
| `cast_f16_to_f32` (596672) | 60 | 1.23 | 74 | 0.01 | 1% | 73 | high |

Rows marked `blk` are per MoE block and `lyr` per vision layer, aggregating the
several kernel launches of that kind inside one block or layer; every other row is
per individual dispatch.

Recoverable totals: **3,563 ms high confidence** (all pure data movement),
880 ms medium, 2,198 ms upper-bound-only.

Two rows carry a floor that omits real work and so must not be ranked against the
rest. Both attention kernels do softmax, whose transcendentals and reductions are
absent from `max(bytes/BW, FLOP/peak)`; for the v8 kernel, ablation already showed
softmax is 41.1 ms of its 203 ms, so its true floor is far above 47.95 ms.

The three linear-attention passes are reported measured-only, because a defensible
FLOP count for the chunked scan could not be derived:

| kernel | calls | ms/call | total ms |
|---|---:|---:|---:|
| `la_pf_pass3_wmma` | 1,110 | 0.829 | 920 |
| `la_pf_scan` | 1,110 | 0.430 | 477 |
| `la_pf_pass1_local` | 1,110 | 0.274 | 304 |
| **total** | | | **1,700** |

## Block model against measured TTFT

| block | count | ms each | total |
|---|---:|---:|---:|
| MoE blocks | 40 | 58.49 | 2,340 |
| linear-attention blocks | 30 | 148.40 | 4,452 |
| full-attention, `gqa` kernel only | 10 | 224.00 | 2,240 |
| vision encoder layers | 27 | 47.80 | 1,291 |
| **sum of captured blocks** | | | **10,322** |
| clean uninstrumented TTFT | | | 14,455 |
| not covered by a capture window | | | 4,133 |

The 4,133 ms residual is honest, not hidden time. The full-attention window
filled the SQTT buffer after 12 dispatches, so everything in those ten layers
except the GQA kernel itself — the QKV and o_proj matmuls, two large transposes
per layer, rotary, KV append — is uncaptured. RGP also pegs clocks, so per-kernel
times are systematically optimistic against production.

## Conclusions

**GQA is no longer the top target, and data movement is.** `transpose_tiled`,
`causal_conv_prefill` and `T5LayernormFwdContiguous` are pure data movement
running at 8-16% of achievable bandwidth, and together they are 3,153 ms of
measured time with 2,839 ms recoverable against a floor that genuinely applies to
them. That is more recoverable time than the entire GQA kernel now costs, and
unlike GQA it needs no algorithmic insight: `causal_conv_prefill` moves 611 MB in
31.5 ms, which is 19 GB/s out of 256.

**`bucket_tokens_kernel` remains the most clear-cut defect** and is now confirmed
at 8.83 ms per MoE block in a single 256-thread workgroup, using 1 of 40 CUs, for
353 ms of the prefill that is almost entirely recoverable.
`origin/perf/qwen-bucket-tokens` already carries a chunked rewrite.

**The MoE block is 41% plumbing.** Of 58.49 ms, real GEMM work is 32.8 ms; the
other 24 ms is `bucket_tokens` 8.8, `scatter_add` 4.9, `gather_tokens` 4.9,
`add_bias` 2.9, `swiglu` 1.3 and `topk_routing` 1.2.

**Sequence-dependent work is 94% of TTFT** (940 ms fixed, 0.729 ms/token), so
there is no startup cost worth attacking.

## Reproducing

```powershell
. C:\Users\zyq\hipep-env.ps1
$env:HIPEP_PROMPT_FILE = 'C:\Users\zyq\prompt_16k.txt'

# clean baseline -- verify no HIPDNN_EP_PERF/DEBUG/TRACE_FILE is set first
.\bench\bench_ttft.ps1 -Tag p0-16k-tight -Driver vlm -SeqLen 16384 `
  -MaxLength 18900 -Reps 3 -Warmup 1 -ExecutionProvider AMDGPU

# per-session decomposition (instrumented: proportions only, never a TTFT)
$env:HIPDNN_EP_TRACE_FILE = 'C:\Users\zyq\logs\eptrace\qwen16k.json'

# per-kernel timing (clean). SPM must be off: with -Counters the 16K decoder
# dies in the allocator ("memrefCopy failed") and no .rgp is written.
.\capture\rgp_capture.ps1 -Op qmoe -Skip 45 -Driver vlm -Reps 3 `
  -PromptFile C:\Users\zyq\prompt_16k.txt -MaxLength 18900 `
  -ExecutionProvider AMDGPU -MaxTokens 1 -Buf default -ArmTimeoutSec 900
```

`-Skip` must land in the second iteration: the first decoder inference takes
~236 s because of the autotune sweeps, and is not steady state. Fence on
`multi_head_attention` for vision (`conv` does not exist in this export, despite
being the obvious guess), on `causal_conv` or `qmoe` for the decoder, and note
that fencing on `gqa` wastes the window because that one kernel fills the buffer
by itself.
