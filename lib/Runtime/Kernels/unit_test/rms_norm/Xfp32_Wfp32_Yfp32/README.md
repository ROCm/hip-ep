# rms_norm Xfp32_Wfp32_Yfp32 standalone test

Standalone `hipcc`-only test for `hip_rms_norm` with fp32 input, scale and
output. It does not configure or build hip-ep, ONNX Runtime, LLVM, or MLIR.
`make` compiles `rms_norm_kernel.hip` into the leaf; the CMake package build
links `custom_kernels_<arch>.dll` instead. Inputs and the CPU reference are
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

fp32 rows take the float4 body when the row is whole float4s with 16-byte
aligned pointers, else the scalar one, and pick their block by width: from
2048 a 1024-thread block, below that a 256-thread block, and below 512 a
32-thread block when the row takes the float4 body. Against a double-precision
CPU reference (every element within `2e-5` relative plus `1e-6` absolute,
since the kernel accumulates in fp32 in its own order):

- row widths 64, 511, 512, 1030, 2047, 2048, 4099 and 5120, which take every
  block x body pair, sit on both width boundaries and loop more than once per
  thread at 5120;
- each width with aligned pointers and with pointers one element off 16-byte
  alignment (a row of whole float4s then takes the scalar body);
- one row (a decode step, where the row's block is the whole grid) and six,
  with one shared gain vector and with three grouped ones (`scale_rows = 3`);
- a one-element row;
- `outer = 0` (returns 0), `scale_rows = 0` and a non-float dtype (return -1),
  none of which may write;
- output guard regions, and a NaN fill that fails any element left unwritten.

## Shape coverage (3 tiers)

Every tier runs all of the above (49 cases plus the 3 edge arguments).
`COVERAGE` only thins the prefill-sized shapes (rows x width, one gain
vector): tier1 adds `64x4096`, tier2 adds `512x2048` and `128x5120`, tier3
(default) adds `2048x4096` and `512x8192`. At startup the exe prints
`rms_norm fp32 coverage=N cases=<n>` (53 / 55 / 57); a successful run prints
one `PASS` line per case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `R6_D2048_G3_misaligned` (rows, width, gain vectors), `config`
the block and body the launcher picks (e.g. `block1024_vec4`), `relL2` against
the reference, and `time_ms` the mean of 50 back-to-back launches after 3
warm-up launches, on warm caches. A launch-to-launch gap is included, so for
the smallest rows it is an upper bound on the kernel time rather than the
kernel time itself.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
