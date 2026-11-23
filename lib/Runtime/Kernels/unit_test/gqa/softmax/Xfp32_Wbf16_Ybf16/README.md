# gqa/softmax / Xfp32_Wbf16_Ybf16

Unit test for the softmax step of the decomposed GroupQueryAttention path
(`hip/gqa_kernel.hip`) when it writes bf16 probabilities: fp32 scores (`X`),
a bf16 additive mask (`W`) and bf16 probabilities (`Y`). It calls
`hip_gqa_softmax_f32_to_bf16` and `hip_gqa_softmax_f32_to_out_biased`, the
entries `real/gqa.cpp` uses for a bf16 query.

## Why this name and these cases

A bf16 GQA reaches this kernel only on the decomposed path, and only the
softmax step changes with the dtype: the GEMMs are hipBLASLt. So the leaf
tests that step on its own, and its X / W / Y are the step's tensors (scores,
mask, probabilities) rather than the GQA's Q / KV / O used by
`gqa/decode` and `gqa/prefill`.

The leaf also runs the biased entry with an fp16 mask and fp16 output, and
with an fp32 mask and fp32 output. Those are regression cases, not separate
dtype leaves: this change replaced the entry's element-size arguments with
dtype arguments, and the fp16 and fp32 GQA paths call the same entry, so a
wrong dtype dispatch would break them there.

## What it checks

Scores are `[B*H, queries, keys]` fp32; each query row is one softmax over its
keys. The reference is computed in double: the bias (read back from its own
rounded dtype, so both sides add the same value) is added before the max, as
the kernel does, then `exp(x - max) / sum`, with `+ exp(-max)` in the sum for
smooth softmax. Every output must be within one unit in the last place of the
output type of the reference, plus 4e-6 relative for the kernel's fp32 `exp2f`
and sum (it matters only for fp32 output); an exact match is not expected.
The bf16 <-> fp32 conversions are integer bit operations, not the HIP
intrinsics the kernel uses.

- Unbiased bf16 output, with and without smooth softmax.
- Biased bf16 mask and bf16 output: a full `[B, H, q, k]` mask; a broadcast
  one-row mask `[1, H, 1, k + 3]` read at column offset 3 (a sliding window);
  and a `[B, 1, 2q, k]` mask read from row offset `q` (a later prefill chunk).
  Every 7th mask element is a large negative value, so masked keys are covered.
- The fp16/fp16 and fp32/fp32 regression cases above.
- Every element is written and nothing else: the output has a padded per-head
  stride and 64 guard elements on each side, and each case runs twice with two
  different fill bytes.
- An unsupported output dtype (u8), an unsupported mask dtype (u8) and a null
  mask return -1 and leave the output untouched. The kernel prints an
  "unsupported dtype" line for the first two; that is expected.

A row whose scores are all `-inf` gives NaN, as in the fp16 path; real masks
always keep at least one key, so the leaf does not test it.

## Coverage

Keys 1 and 7 fit in one wave, 33 needs a partial second wave, 256 fills a
256-thread block and 1000 makes each thread loop over several keys.
`--coverage N` (or `HIPDNN_UT_COVERAGE`, default 3) picks the shapes; each
shape runs 7 cases, and the rejected cases run at every tier.

| Tier | Shapes (B, H, queries, keys) |
|---|---|
| 1 | (1, 2, 3, 7), (2, 3, 4, 33), (1, 4, 2, 1000) |
| 2 | + (1, 1, 1, 1), (2, 2, 5, 256) |
| 3 | + (1, 8, 16, 1000) |

## Targets

```
make test        HIP_SDK=<sdk> [OFFLOAD=--offload-arch=<gfx>] [COVERAGE=1|2|3]
make test_custom HIP_SDK=<sdk> [BATCH=..] [HEADS=..] [QUERIES=..] [KEYS=..]
                 [MASK=..] [MASK_DTYPE=..] [OUT_DTYPE=..] [SMOOTH=..] [SEED=..]
make build ...
make clean
```

`test_custom` runs one case through the same check and timer as the suite;
any exe flag other than `--coverage` selects it:

| Make | Exe flag | Default | Meaning |
|---|---|---|---|
| `BATCH` | `--batch` | 1 | Batch B, >= 1. |
| `HEADS` | `--heads` | 8 | Query heads H, >= 1. |
| `QUERIES` | `--queries` | 16 | Query rows, >= 1; one softmax per (head, query). |
| `KEYS` | `--keys` | 1000 | Keys, the softmax length, >= 1. |
| `MASK` | `--mask` | `full` | `none` (unbiased entry), `full`, `key_row` or `chunk` (the three layouts above). |
| `MASK_DTYPE` | `--mask-dtype` | `bf16` | `bf16`, `f16` or `f32`; ignored for `none`. |
| `OUT_DTYPE` | `--out-dtype` | `bf16` | `bf16`, `f16` or `f32`; `none` writes `bf16` only. |
| `SMOOTH` | `--smooth` | 0 | 1 adds the smooth-softmax term. |
| `SEED` | `--seed` | 1 | Seed of the scores and the mask, >= 0. |

`B * H * 2 * queries * (keys + 3)` must not exceed 2^26 (the double reference
is held in memory). Unknown or out-of-range arguments exit 2.

## Timing

Every passing case is timed with the shared UT timer
(`common/hip_adaptive_timing.h`) and writes `UT_TIMING` /
`UT_TIMING_BINDING` records; `time_ms` in `out/results.csv` is its mean.
`TIMING=screen|standard|confirm` (default `standard`) selects the preset and
`ADLX_INCLUDE=<dir>` optionally adds GPU clock sampling. The rejected cases
are not timed.

## Architectures and MODE

`OFFLOAD` defaults to `--offload-arch=gfx11-generic` (gfx115x); pass
`--offload-arch=gfx1151`, `gfx1170` or `gfx1171` for a chip build.

The softmax has no autotune table and no tuner; `MODE` only labels
`out/results.csv`, and every row's `config` is `fixed:gqa_softmax`.
