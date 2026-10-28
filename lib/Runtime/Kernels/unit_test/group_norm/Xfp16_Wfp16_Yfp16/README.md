# group_norm Xfp16/Wfp16/Yfp16 standalone test

Standalone `hipcc`-only test for `hip_group_norm` (com.microsoft GroupNorm)
with fp16 input, scale (gamma), bias (beta) and output. It does not configure
or build hip-ep, ONNX Runtime, LLVM, or MLIR. `make` compiles
`group_norm_kernel.hip` into the leaf; the CMake package build links
`custom_kernels_<arch>.dll` instead. Inputs and the CPU reference are
generated in-process -- no python, no data files.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... ARGS="--coverage 1"
make clean
```

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | Runs the exe with `ARGS=` (e.g. `--coverage 1`). |
| `clean` | Removes `out/`. |

## What it checks

Against an fp64 CPU reference computed from the rounded fp16 inputs (two-pass
mean and variance, then the affine and the optional SiLU), within a per-dtype
absolute plus relative tolerance:

- channels_last 1 (NHWC) and 0 (NCHW);
- SiLU on or off (alternating by shape);
- the workspace the size query asks for (the split reduction), no workspace
  (the fused one-block-per-group kernel), a workspace 7x too small (a shrunk
  split or the fused kernel), and tensors one element off their allocation's
  alignment (scalar access);
- bitwise-identical output over two launches with the full workspace;
- output guard regions and a fill pattern that fails any element left
  unwritten.

Every shape also runs with inputs near a dtype-scaled offset (|mean| / stddev
up to about 1e4 for fp32 and 1e8 for fp64) and with sparse +-32 outliers on
top, so a variance computed from raw sums of squares fails on cancellation.
The reference sums in long double relative to each group's first value, and
the tolerance adds 16 compute-type ulps of `|mean| * |gamma| / stddev`: an
exact algorithm still rounds the mean to the compute type.

Further inputs cover values across most of the dtype's range (overflow of
intermediate sums), a group of one repeated large value (the output must be
exactly beta, so the kernels subtract the mean before the affine instead of
folding it into the bias), and a NaN in group 0 with +Inf as group 1's first
value (both groups must be NaN and the others unaffected). Zero N, C and
spatial extents must return success without touching memory.

`ARGS="--large"` adds a 2^31 + 96-element tensor in both layouts, so 32-bit
offset arithmetic would wrap. It needs about 9 GB of device memory for fp16
and bf16, twice that for fp32, and is not part of `make test`.

## Shape coverage (3 tiers)

Every tier runs all 8 layout x workspace combinations on each selected shape
(the largest tier3 shapes run only with the full workspace). `COVERAGE` only
thins the shape list (N x C x spatial, groups): tier1 covers the small fused
shapes, odd channels per group (C 40, G 8), G = 1 with prime spatial, G = C and
spatial 1; tier2 adds N = 3, groups wider than one block (C 4096 and 3000,
G 1) and the 512 x 96x96 VAE shape; tier3 (default) adds the 512 x 192x192 and
256 x 384x384 VAE shapes. A successful run prints one `PASS relL2=...` line per
case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `N2_C40_HW1000_G8`, `config` like `fp16:nhwc:split:silu1` and
`time_ms` 0: the leaf checks correctness only.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
