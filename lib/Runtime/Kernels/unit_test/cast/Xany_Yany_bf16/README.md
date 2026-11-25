# cast / Xany_Yany_bf16

Unit test for `hip_cast` (`hip/cast_kernel.hip`) when one side of the cast is
bf16: bf16 <-> {f32, f16, f64, i64, i32, i16, i8, u8} (16 pairs), plus
f64 <-> {f32, f16, i64, i32, i16, i8, u8} (14 pairs). 30 pairs in all.

## Why one leaf for many dtypes

The unit-test README asks for one leaf per tensor-dtype combo. Cast has one
input and one output and the bf16 paths share a single templated kernel
(`cast_via_float_kernel`), so one near-identical leaf per pair would add no coverage.
This leaf runs every pair as its own case instead (`X` = any source, `Y` = any
destination, `_bf16` = bf16 on one side). Each case has its own
`out/results.csv` row, so a failure still names the exact pair.

## What it checks

Every output is compared **bit-exactly** with a CPU reference written in the
test with plain integer bit operations (it does not use the HIP intrinsics):

- fp32 -> bf16 and fp32 -> fp16 round to nearest even; bf16 -> fp32 is exact;
  float -> integer truncates toward zero. An f64 or i64 source to bf16 goes
  through fp32 first (two roundings), the same as the kernel and PyTorch.
- f64 -> f16 is one correct rounding, not two. The reference is computed in
  double (`f64_to_f16_direct`). A two-step f64 -> f32 -> f16 differs by one
  f16 ulp on values just off a tie, and the specials pin that case.
- f64 <-> integer is a direct static_cast of the double (truncation toward
  zero), and integer -> f64 is one rounding to double. Neither goes through
  fp32, so integers wider than 24 bits keep their value.
- Special values: +-0, +-inf, NaN (any NaN matches a NaN reference), rounding
  ties, fp32/fp16/bf16 subnormals, the largest finite bf16, overflow to inf.
  16-bit sources enumerate every bit pattern at the 70001-element size.
- Every element is written and nothing outside the output: each case runs
  twice with two different fill bytes and 64 guard elements on each side.
- Unsupported pairs return -1 and leave the output untouched, one per exit of
  the bf16 dispatch: bf16 -> u16 (output dtype), u16 -> bf16 (input dtype).
  The f64 path adds f64 -> u16 and u16 -> f64 (u16 is not an f64 integer
  endpoint). The kernel prints an "unsupported conversion" line for each;
  that is expected.

bf16 -> i8 / u8 / i16 / i32 / i64 is only checked with values that fit the
destination: a float -> integer cast of an out-of-range value or NaN is
undefined in C++, as it is for the pre-existing fp32/fp16 -> integer kernels.

## Coverage

257 ends in a partial 256-thread block, 256 is exactly one block, 70001 holds
every 16-bit source pattern. `--coverage N` (or `HIPDNN_UT_COVERAGE`,
default 3) picks the sizes; every dtype pair and unsupported case runs at
every tier.

| Tier | Sizes |
|---|---|
| 1 | 257, 70001 |
| 2 | + 256 |
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
