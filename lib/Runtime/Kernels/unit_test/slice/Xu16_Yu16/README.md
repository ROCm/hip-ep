# slice Xu16_Yu16 standalone test

Standalone `hipcc`-only test for `hip_slice` on 16-bit elements
(f16, bf16, i16, u16). It does not configure or build hip-ep, ONNX Runtime, LLVM, or
MLIR. `make` compiles `slice_kernel.hip` into the leaf; the CMake package build
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

A slice only moves elements, so `hip_slice` runs one unsigned instantiation
per element size and every dtype of that size maps to it. This leaf launches
each of f16, bf16, i16, u16, so both the dtype dispatch and the 16-bit copy are
checked. Output must equal the in-process ONNX Slice reference byte for byte
(inputs are random bit patterns, NaNs and denormals of the float dtypes
included):

- contiguous, strided (step 2 on every axis), reversed (step -1), mixed
  starts and steps, rank 8 (the maximum), and an empty output;
- an over-allocated output (`logical_extent` below the physical shape), whose
  tail must be zero; every other case passes a null `logical_extent`;
- rank 0, rank 9 and an unknown dtype, which must return -1 and write nothing;
- a launch made while an earlier failed HIP call's error is still pending,
  which must not be reported as hip_slice's own;
- output guard regions, and a fill that fails any element left unwritten.

## Shape coverage (3 tiers)

Every tier runs the categorical situations above for every dtype of the leaf.
`COVERAGE` only thins the typical sizes: tier1 adds `512x1024 -> 256x512`,
tier2 adds a reversed `64x128x96 -> 64x64x96` and an over-allocated
`1024x768 -> 512x768`, tier3 (default) adds `4096x2048 -> 2048x1024`. At
startup the exe prints `slice u16 coverage=N cases=<n>`; a successful run
prints one `PASS` line per case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `strided_in33x70_out16x34`, `config` the dtype, `relL2` 0 on a
pass and 1 otherwise, and `time_ms` 0: the leaf checks correctness only.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
