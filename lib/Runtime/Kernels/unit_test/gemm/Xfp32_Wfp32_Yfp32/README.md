# gemm Xfp32/Wfp32/Yfp32 standalone test

Standalone `hipcc`-only test for the CK GEMM entry points with fp32 operands
and fp32 output (`hip_ck_gemm_run`, `hip_ck_gemm_lut_candidates`,
`hip_ck_gemm_instance_name`, `hip_ref_gemm_run`). The fp32 combos run on CK's
FMA-based DL pipeline. It does not configure or build hip-ep, ONNX Runtime,
LLVM, or MLIR. `make` compiles `ck_gemm.hip` and `ref_gemm.hip` into the leaf
against the pinned Composable Kernel source; the CMake package build links
`custom_kernels_<arch>.dll` instead. Inputs and the CPU reference are
generated in-process -- no python, no data files.

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
README), for the two fp32 -> fp32 combos, NN and TN, whose epilogue applies
alpha. Every categorical case runs with alpha 1 and alpha 0.375, and the
attention-score typical shapes with alpha `1/sqrt(k)`.

Declined cases (`config` `declined`) pass when the entry point returns non-zero
and leaves its output untouched:

- `NN_F32:DLKN_1` and `TN_F32AB:DLNK_1` (4-wide output stores) with m = 30,
  and with m = 32 but ldd = 34: the instance refuses, while it accepts the
  aligned twin (m = 32, ldd = 32 or the non-packed 40);
- `hip_ck_gemm_run` with a bias: every instance refuses, while some instance
  accepts the same problem without it;
- `hip_ref_gemm_run` with a non-packed leading dimension, with `m * k` past
  int32, and with `ceil(m / 16)` past the 65535 grid-y limit (sizes only, with
  a 4 KB buffer: the limits are checked before any memory access).

Not covered: the per-launch 2 GB operand window of the DL instances and its
batch splitting, which need more than 2 GB of device memory.

## Tolerance

relL2 <= 1e-5. fp32 FMA accumulation gives relL2 of 1.3e-6 at k = 5120 on
gfx1151, identical for the DL instances and the reference GEMM, and below
that at every smaller k.

## Shape coverage (3 tiers)

Every tier runs the 128 categorical cases (2 combos x 8 alignments of a
200 x 136 x 168 problem x 4 batch layouts x alpha {1, 0.375}) and 9
registry / rejection cases. `COVERAGE` thins the typical shapes exactly as in
`gemm/Xfp16_Wfp16_Yfp16` (2 / 4 / 8 shapes, each on both combos). At startup
the exe prints `gemm coverage=N mode=... -> ... = <n> cases`
(141 / 145 / 153) and ends with a `summary` line and `ALL PASS`.

## `MODE` and `out/results.csv`

As in `gemm/Xfp16_Wfp16_Yfp16`: `config` is `lookup:<instance>`,
`autotune:<instance>`, `fixed:ref`, `fixed:registry` or `declined`, a
reference row of a case that ran CK has `_ref` appended to its `shape`, and
`time_ms` is the mean of 10 launches rotating over 256 MB of input copies.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
