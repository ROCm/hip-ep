# qmoe_ragged_mm Xfp16/Wu4/Yfp16 standalone test

Standalone `hipcc`-only test for the persistent LDS-staged FP16-WMMA ragged
MatMul primitive (`hip_qmoe_ragged_matmul_nbits`) and the routing kernels
around it. It does not configure or build hip-ep, ONNX Runtime, LLVM, or MLIR.
`make` compiles `qmoe_ragged_mm_kernel.hip` and `qmoe_kernel.hip` (token
bucketing, SwiGLU) into the leaf; the CMake package build links
`custom_kernels_<arch>.dll` instead. Inputs and the CPU reference
are generated in-process -- no python, no data files.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... ARGS="--launcher-grid-only"
make clean
```

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | Runs the exe with `ARGS=` (e.g. `--coverage 1`, `--launcher-grid-only`). |
| `clean` | Removes `out/`. |

## What it checks

Against the in-process CPU reference:

- balanced, empty-expert, and highly skewed routing;
- FC1-style indirect/gather input and FC2-style contiguous input;
- bias disabled and enabled;
- UINT4 weights with symmetric zero point 8 and unpacked asymmetric UINT8
  per-group zero points;
- row-group coverage and output guard regions;
- adaptive task selection: vectorized GEMV for 1..7 rows, WMMA16 for 8..31
  rows, and WMMA64 for 32..64 rows, including partial M/N/K tails;
- device bucket counts/offsets, `pair_to_sorted`, four-field adaptive row-group
  descriptors, and queue initialization;
- occupancy-capped persistent launch grids, including task-bound and
  occupancy-bound cases;
- one ragged FC1 launch, one SwiGLU launch, one ragged FC2 launch, followed by
  deterministic sorted reduction (20 byte-identical repetitions).

## Shape coverage (3 tiers)

Every tier runs all 24 routing x input layout x bias x zero-point
combinations on each selected shape, plus the row counts on both sides of
each GEMV / WMMA16 / WMMA64 boundary (`task_m1` .. `task_m65`), the bucket
descriptors, the task-policy and launch-grid selectors and the pipeline.
`COVERAGE` only thins the shape list: tier1 = `tiny`, tier2 adds
`multi_group`, tier3 (default) adds `tail_k_and_n`, `wider` and the exact
GPT-OSS-20B FC1 (`K=2880,N=5760`) and FC2 (`K=2880,N=2880`) dimensions. At
startup the exe prints `qmoe_ragged_mm coverage=N cases=<n>`; a successful run
prints one `PASS relL2=...` line per case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

This primitive has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`).
`config` uses the `fixed:` prefix and is read back through the production
selectors (`hip_qmoe_ragged_matmul_nbits_get_task_kind` / `_get_launch_config`):
for a MatMul case, the stage's task policy, the queue policy, the grid and the
task kinds its row groups ran, e.g. `fixed:p0-current queue=atomic1 grid=20
tasks=gemv+wmma16`. The leaf checks correctness only, so `time_ms` is 0.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
