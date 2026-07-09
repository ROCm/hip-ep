# AMDMLSS gfx1151 (Strix Halo) kernel port assessment

Status: **assessed, not merged.** This document records which AMDMLSS kernels are
portable to `onnx-hipdnn-ep` on gfx1151, what was actually validated, and why none
of them is worth enabling for the target workload (LLM **decode**) right now.

## TL;DR

- The complete set of gfx1151-eligible AMDMLSS kernels is small: **MHA, GQA, MVN,
  Conv1x1, Conv-dilated**. Everything else (RMSNorm, SiLU, GEMMGEMM, ConvMxN,
  GEMM, QGEMM) is RDNA4-only (`gfx1200/1201`) or an empty scaffold.
- All AMDMLSS attention kernels for gfx1151 are **CK WMMA throughput kernels**.
  They are efficient only for large query tiles (prefill), and only for
  `head_dim <= 80`. They are the wrong tool for **decode** (`q_seq = 1`).
- Real target models are all **GQA with `head_dim = 128`** (or 256). That combination
  falls on the **untuned fallback tile** and never hits the tuned kernels.
- Net: **zero (or negative) value for decode, and ~zero for prefill on real models.**
  Only the MHA decode shim was wired (behind an `m==1` gate) and its win is marginal.

## Kernels available on gfx1151

| Op | Available | Backend | Notes |
|---|---|---|---|
| MHA | Yes | CK WMMA fp16 | reuses gfx1150 blob |
| GQA | Yes | CK WMMA fp16 (relocatable) | reuses gfx1150 blob |
| MVN / InstanceNorm | Yes | HIP `mvn2` | vision/CNN op |
| Conv 1x1 | Yes | HIP WMMA (Gemm2d) | vision/CNN op |
| Conv dilated | Yes | HIP WMMA (grouped conv) | vision/CNN op |
| Conv MxN Winograd | No | — | excludes gfx115x |
| RMSNorm, SiLU, GEMMGEMM | No | — | `isGfx120x` only (RDNA4) |
| GEMM, QGEMM | No | — | scaffold, no code objects |

MVN / Conv1x1 / Conv-dilated are CNN/vision operators and do not appear in transformer
decode graphs, so they contribute nothing to the LLM workload regardless of whether
they are ported.

## GQA ABI (now fully decoded)

The blocker in prior rounds was the exact host ABI. It is recovered from
`amd-mlss-tester` (`lib/include/common/misc.hpp::calcStrides` +
`unit-tests/gqa_test.cpp`):

- Physical tensor layout **BSHD**; stride arrays passed in **`[batch, head, seq, dim]`** order.
- `Q`, `K`, `Out` use `make_query_strides(head, seq) = {seq*head*hd, hd, head*hd, 1}`.
- **`V` uses `make_value_strides(kvhead, kvseq) = {kvseq*kvhead*hd, hd, 1, kvhead*hd}`**
  i.e. `V` has **seq and dim strides swapped** vs `Q`. This single detail is what
  earlier stride brute-forcing never tried (it reused the `K` stride array for `V`),
  and is why validation failed for > 2 heads.
- Kernel arg order (28 args): `Q**, K**, V**, Out**`, then `int` `M=q_seq, N=kv_seq,
  K=head_dim, O=d_v, G0=batch, G1=q_heads, G1kv=kv_heads`, `float scale`, then
  `q/k/v/out` strides (4 each). Grid/block come from the binary metadata.

With this, standalone validation **passes bit-approximately (`maxAbsDiff ~6e-5`, fp16
tolerance 5e-2)** across all real decode shapes, including `head_dim = 128`:

```
GQA B=1 QH=32 KVH=8 QS=1 KVS=128..4096 HD=128  -> PASS (maxAbsDiff ~6e-5)
```

So **correctness is no longer the blocker. Performance is.**

## Why it is not worth enabling (measured on gfx1151)

### Decode (`q_seq = 1`) is memory-bound; the WMMA kernel wastes the machine

Peak: 59.4 TFLOPS fp16 (WMMA), 256 GB/s LPDDR5x. Decode attention is bandwidth-bound.

| KVS | kernel | achieved BW | vs mem-floor |
|---|---|---|---|
| 512 | 49 us | 43 GB/s (17% peak) | 6.0x |
| 4096 | 395 us | 43 GB/s (17% peak) | 6.0x |

Root cause — the WMMA path always processes a 16-row query tile:

| q_seq | kernel (KVS=1024, HD=128) | us / query |
|---|---|---|
| 1 | 100 us | 100 |
| 16 | 103 us | 6.5 |
| 64 | 172 us | 2.7 |

`q_seq=1` and `q_seq=16` cost the same: decode uses 1 of 16 rows and throws away 15/16.
Routing decode GQA here **regresses** vs any decode-tuned GEMV path.

### Prefill is only OK for `head_dim <= 80`; real models (`head_dim = 128`) fall to the fallback tile

| head_dim | q_seq | TFLOP/s | % of 59.4 peak | tile |
|---|---|---|---|---|
| 48 | 1024 | 12.4 | 21% | tuned |
| 80 | 1024 | 13.6 | 23% | tuned |
| 128 | 512 | 5.0 | 8% | fallback |
| 128 | 1024 | 2.6 | 4% | fallback |

The tuned GQA tiles only exist for `head_dim <= 80`. Every target model is
`head_dim = 128` (Llama-3.1, Mistral, Qwen2.5/3, DeepSeek) or 256 (gemma3, Qwen3.5),
so prefill hits the fallback at **4-8% of peak and degrades with prompt length**.
Even the best tuned case (~20% of peak) is well under CK/hipBLASLt efficiency (~60%).

## Recommendation

- **Do not enable GQA (or the other ops) for decode or prefill on gfx1151.** No target
  model benefits.
- Keep the MHA decode shim gated behind `m==1` (already the case); its win is marginal
  and no target model even emits plain MHA nodes, so it is effectively dormant.
- The value delivered here is **enablement, not speed**: the GQA ABI is decoded and a
  validator exists. If AMD ships (a) a decode-tuned (GEMV / flash-decode) attention
  blob for gfx1151, or (b) `head_dim = 128` tuned tiles, wiring + validating it is now
  a ~1-day task instead of weeks of ABI reverse-engineering.
