# div / Xbf16_Wbf16_Ybf16

Unit test for `hip_elementwise_div` (`hip/elementwise_binary_kernel.hip`) with
bf16 lhs (`X`), rhs (`W`) and output (`Y`).

## What it checks

Every output is compared **bit-exactly** with a CPU reference written in the
test: both operands are widened to fp32 (exact), divided in fp32 and rounded
once to bf16 with round to nearest even, the same as the kernel and PyTorch.
The bf16 <-> fp32 conversions use plain integer bit operations, not the HIP
intrinsics. A NaN result matches any NaN.

- Special values: every pair of +-0, +-inf, NaN, +-1, 2, 3, 7, 10, 0.1, the
  largest finite bf16 and the smallest normal bf16 (both signs), so x / 0,
  0 / 0, inf / inf, overflow to inf and results that need rounding are all
  covered. The rest of each tensor is seeded random values.
- Quotients that are fp32 subnormals are left out: whether the GPU keeps or
  flushes them depends on the device's denormal mode, not on this kernel.
  Random operands use exponents in [-60, 60] so their quotients stay normal.
- Every element is written and nothing outside the output: each case runs
  twice with two different fill bytes and 64 guard elements on each side.
- An unsupported dtype (u8) returns -1 and leaves the output untouched. The
  kernel prints an "unsupported dtype" line for it; that is expected.

Broadcasting is not part of this kernel: `wrap_div` materialises a broadcast
operand with `hip_expand` first.

## Coverage

257 ends in a partial 256-thread block and holds every special pair, 256 is
exactly one block. `--coverage N` (or `HIPDNN_UT_COVERAGE`, default 3) picks
the sizes; the unsupported case runs at every tier.

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
