# gqa/softmax / Xfp32_Wbf16_Ybf16

Unit test for the softmax step of the decomposed GroupQueryAttention path
(`hip/gqa_kernel.hip`) when it writes bf16 probabilities: fp32 scores (`X`),
a bf16 additive mask (`W`) and bf16 probabilities (`Y`). It calls
`hip_gqa_softmax_f32_to_bf16` and `hip_gqa_softmax_f32_to_out_biased`, the
entries `real/gqa.cpp` uses for a bf16 query.

## What it checks

Scores are `[B*H, queries, keys]` fp32; each query row is one softmax over its
keys. The reference is computed in double: the bias (read back from its own
rounded dtype, so both sides add the same value) is added before the max, as
the kernel does, then `exp(x - max) / sum`, with `+ exp(-max)` in the sum for
smooth softmax. Every output must be within one unit in the last place of the
output type of the reference, plus 4e-6 relative for the kernel's fp32 `exp2f`
and sum (it matters only for fp32 output); an exact match is not expected.
The bf16 <-> fp32 conversions are integer bit operations, not
the HIP intrinsics the kernel uses.

- Unbiased bf16 output, with and without smooth softmax.
- Biased bf16 mask and bf16 output: a full `[B, H, q, k]` mask; a broadcast
  one-row mask `[1, H, 1, k + 3]` read at column offset 3 (a sliding window);
  and a `[B, 1, 2q, k]` mask read from row offset `q` (a later prefill chunk).
  Every 7th mask element is a large negative value, so masked keys are covered.
- The same biased entry with an fp16 mask and fp16 output, and with an fp32
  mask and fp32 output: its dtype arguments replaced the element-size
  arguments those paths used, so they are regression cases.
- Every element is written and nothing else: the output has a padded per-head
  stride and 64 guard elements on each side, and each case runs twice with two
  different fill bytes.
- An unsupported output dtype (u8), an unsupported mask dtype (u8) and a null
  mask return -1 and leave the output untouched. The kernel prints an
  "unsupported dtype" line for the first two; that is expected.

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
make test_custom HIP_SDK=<sdk> ARGS="--coverage 1"
make build ...
make clean
```

The softmax has no autotune table and no tuner; `MODE` only labels
`out/results.csv`, and every row's `config` is `fixed:gqa_softmax`.
