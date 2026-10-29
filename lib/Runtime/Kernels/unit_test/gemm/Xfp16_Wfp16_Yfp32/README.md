# gemm Xfp16/Wfp16/Yfp32 standalone test

Standalone `hipcc`-only test for the CK GEMM entry points with fp16 operands
and fp32 output (`hip_ck_gemm_run`, `hip_ck_gemm_lut_candidates`,
`hip_ck_gemm_instance_name`, `hip_ref_gemm_run`). It does not configure or
build hip-ep, ONNX Runtime, LLVM, or MLIR. `make` compiles `ck_gemm.hip` and
`ref_gemm.hip` into the leaf against the pinned Composable Kernel source; the
CMake package build links `custom_kernels_<arch>.dll` instead. Inputs and the
CPU reference are generated in-process -- no python, no data files.

```
make test        HIP_SDK=%HIP_SDK% CK_SRC=<ck checkout> CK_GEN=<build>/ck_generated [OFFLOAD=--offload-arch=%GFX%] [COVERAGE=1|2|3]
make test_custom HIP_SDK=... CK_SRC=... CK_GEN=... ARGS="--coverage 1"
make clean
```

`MODE=lookup` (default) also needs `FLATC` and `FLATBUFFERS_INC`, as the
matmul_nbits leaves do.

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | Runs the exe with `ARGS=` (e.g. `--coverage 1`). |
| `clean` | Removes `out/`. |

## What it checks

Routing, the reference row, the batch layouts, the alignments and the
untouched-padding checks are those of `gemm/Xfp16_Wfp16_Yfp16` (see its
README), for the one fp16 -> fp32 combo: TN without bias, whose epilogue
applies alpha. Every categorical case runs with alpha 1 and alpha 0.375, and
the attention-score typical shapes with alpha `1/sqrt(k)`.

Declined cases (`config` `declined`) pass when the entry point returns non-zero
and leaves its output untouched:

- `hip_ck_gemm_run` with NN, and with a bias: every instance refuses, while
  some instance accepts the same problem as TN without bias;
- `hip_ref_gemm_run` with a non-packed leading dimension, with `m * k` past
  int32, and with `ceil(m / 16)` past the 65535 grid-y limit (sizes only, with
  a 4 KB buffer: the limits are checked before any memory access).

Not covered: the per-launch 4 GiB operand window and its batch splitting,
which need more than 4 GiB of device memory.

## Tolerance

relL2 <= 1e-6 + 1e-8 * k. The WMMA instances' fp32 accumulation error grows
linearly with k: measured relL2 is 5.3e-7 at k = 168, 1.4e-6 at k = 448,
4.0e-6 at k = 1280 and 1.6e-5 at k = 5120 on gfx1151, against 1.3e-6 at
k = 5120 for the FMA-based reference GEMM. The bound leaves at least 3x
headroom at every k.

## Shape coverage (3 tiers)

Every tier runs the 64 categorical cases (8 alignments of a 200 x 136 x 168
problem x 4 batch layouts x alpha {1, 0.375}) and 6 registry / rejection
cases. `COVERAGE` thins the typical shapes exactly as in
`gemm/Xfp16_Wfp16_Yfp16` (2 / 4 / 8 shapes). At startup the exe prints
`gemm coverage=N mode=... -> ... = <n> cases` (72 / 74 / 78) and ends with a
`summary` line and `ALL PASS`.

## `MODE` and `out/results.csv`

As in `gemm/Xfp16_Wfp16_Yfp16`: `config` is `lookup:<instance>`,
`autotune:<instance>`, `fixed:ref`, `fixed:registry` or `declined`, a
reference row of a case that ran CK has `_ref` appended to its `shape`, and
`time_ms` is the mean of 10 launches rotating over 256 MB of input copies.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
