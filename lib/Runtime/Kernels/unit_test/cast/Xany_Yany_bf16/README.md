# cast / Xany_Yany_bf16

Unit test for `hip_cast` (`hip/cast_kernel.hip`) when one side of the cast is
bf16: bf16 <-> {f32, f16, f64, i64, i32, i16, i8, u8}, 16 pairs.

## Why one leaf for many dtypes

The unit-test README asks for one leaf per tensor-dtype combo. Cast has one
input and one output and the bf16 paths share a single templated kernel
(`cast_via_float_kernel`), so 16 near-identical leaves would add no coverage.
This leaf runs every pair as its own case instead (`X` = any source, `Y` = any
destination, `_bf16` = bf16 on one side). Each case has its own
`out/results.csv` row, so a failure still names the exact pair.

## What it checks

Every output is compared **bit-exactly** with a CPU reference written in the
test with plain integer bit operations (it does not use the HIP intrinsics):

- fp32 -> bf16 and fp32 -> fp16 round to nearest even; bf16 -> fp32 is exact;
  float -> integer truncates toward zero. f64 and i64 sources go through fp32
  first (two roundings), the same as the kernel and PyTorch.
- Special values: +-0, +-inf, NaN (any NaN matches a NaN reference), rounding
  ties, fp32/fp16/bf16 subnormals, the largest finite bf16, overflow to inf.
  16-bit sources enumerate every bit pattern at the 70001-element size.
- Every element is written and nothing outside the output: each case runs
  twice with two different fill bytes and 64 guard elements on each side.
- Regression: the pre-existing f32 -> f16, f16 -> f32, i64 -> i32 and
  u8 -> f32 paths give the same results.
- Unsupported pairs (bf16 <-> u16, f64 -> i8) return -1 and leave the output
  untouched (the kernel prints an "unsupported conversion" line, expected).
- `num_elements == 0` returns 0 and writes nothing.

bf16 -> i8 / u8 / i16 / i32 / i64 is only checked with values that fit the
destination: a float -> integer cast of an out-of-range value or NaN is
undefined in C++, as it is for the pre-existing fp32/fp16 -> integer kernels.

## Coverage

Sizes straddle the 256-thread block. `--coverage N` (or `HIPDNN_UT_COVERAGE`,
default 3) picks the sizes; every dtype pair and edge case runs at every tier.

| Tier | Sizes |
|---|---|
| 1 | 1, 257, 70001 |
| 2 | + 255, 256, 4099 |
| 3 | + 1048579 |

## Targets

```
make test        HIP_SDK=<sdk> [OFFLOAD=--offload-arch=<gfx>] [COVERAGE=1|2|3]
make test_custom HIP_SDK=<sdk> ARGS="--coverage 1"
make build ...
make clean
```

The op has no autotune table and no tuner; `MODE` only labels
`out/results.csv`, and every row's `config` is `fixed:elementwise`.
