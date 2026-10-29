# gemm Xfp16/Wfp16/Yfp16 standalone test

Standalone `hipcc`-only test for the CK GEMM entry points with fp16 operands
and fp16 output (`hip_ck_gemm_run`, `hip_ck_gemm_lut_candidates`,
`hip_ck_gemm_instance_name`, `hip_ref_gemm_run`) and for `hip_gemv_fp16`. It
does not configure or build hip-ep, ONNX Runtime, LLVM, or MLIR. `make`
compiles `ck_gemm.hip`, `ref_gemm.hip` and `gemv_kernel.hip` into the leaf
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

Each GEMM case picks its route the way `real/ck_gemm_select.h` orders it: the
table's proposals from `hip_ck_gemm_lut_candidates` first, then every instance
in registry order; the first one `hip_ck_gemm_run` accepts runs (production
times the accepted ones instead). With no accepting instance the case runs
`hip_ref_gemm_run`. Every case also runs `hip_ref_gemm_run` on packed copies of
the same operands as a separate row, since it is the fallback for every
refused shape; it has no bias, so that row is checked without one.

Against a double-precision CPU reference:

- the four fp16 -> fp16 combos: NN, TN, NN + bias, TN + bias;
- batch 1, and batch 3 with packed batch strides, with padded leading
  dimensions and batch gaps, and with B broadcast (`strideB = 0`);
- m, n and k that are multiples of 8, of 4 only, of 2 only (m) and odd;
- every element outside the output matrices (leading-dimension padding, batch
  gaps, a 64-element tail) must keep its NaN fill, and input padding is NaN,
  so a kernel that reads or writes outside the matrices fails;
- `hip_gemv_fp16` (`M == 1`) on its wide kernel with and without 4-wide
  vectors (`n % 4`, and `b` offset by one element), its narrow K-split kernel
  with and without them (`n % 4`, and `y` offset by one element), and the two
  shapes it declines (`n < 64`, `k < 256`). The path label in `config` is
  derived from the device's CU count with the kernel's own rule.

Declined cases (`config` `declined`) pass when the entry point returns non-zero
and leaves its output untouched:

- `hip_ck_gemm_run` with alpha 0.5 on each combo: every instance refuses,
  while some instance accepts the same problem at alpha 1;
- `hip_ref_gemm_run` with a non-packed leading dimension, with `m * k` past
  int32 (`M65536_K65536`), and with `ceil(m / 16)` past the 65535 grid-y limit
  (`M1048576`). The last two pass sizes only, with a 4 KB buffer: the limits
  are checked before any memory access.

A registry row checks that instance names are unique and that out-of-range
indices are refused.

Not covered: the per-launch 4 GiB operand window and its batch splitting,
which need more than 4 GiB of device memory.

## Tolerance

relL2 <= 1e-3. Rounding the output to fp16 alone gives relL2 about 2.1e-4
(half an fp16 ulp, uniformly distributed); the measured maximum over every
row, CK and reference alike, is 2.2e-4 on gfx1151.

## Shape coverage (3 tiers)

Every tier runs the 128 categorical cases (4 combos x 8 alignments of a
200 x 136 x 168 problem x 4 batch layouts), the 9 GEMV cases and 8
registry / rejection cases. `COVERAGE` thins the typical shapes, each run on
all 4 combos (column-major: m = keys, n = query rows, k = head dim for an
attention score; m = out features, n = tokens, k = in features for a linear
layer):

| Tier | Adds |
|---|---|
| 1 | linear 1280 x 100 x 1280; score 448 x 448 x 64, batch 20 |
| 2 | value 64 x 448 x 448, batch 20; linear 5120 x 100 x 1280 |
| 3 | linear 1280 x 1500 x 5120; decode score 1500 x 1 x 64 (batch 20) and 777 x 1 x 128 (batch 32); decode value 128 x 1 x 777 (batch 32) |

At startup the exe prints `gemm coverage=N mode=... -> ... = <n> cases`
(153 / 161 / 177) and ends with a `summary` line and `ALL PASS`.

## `MODE` and `out/results.csv`

`lookup` asks `hip_ck_gemm_lut_candidates` first; `autotune`
(`HIPDNN_KERNEL_UT_MODE=autotune`, which `hipdnn-kernel-ut.exe --mode autotune`
sets) also sets `HIPDNN_CK_GEMM_AUTOTUNE_MODE=online`, so the table is not
consulted. `config` is `lookup:<instance>`, `autotune:<instance>`,
`fixed:ref`, `fixed:gemv-<wide|narrow>-v<4|1>`, `fixed:registry` or
`declined`. A reference row of a case that ran CK has `_ref` appended to its
`shape`, e.g. `TN_bias_M199_N135_K167_B3_pad_ref`.

`time_ms` is the mean of 10 launches rotating over copies of every input (A,
B, bias) totalling 256 MB, as in the matmul_nbits leaves;
`HIPDNN_KERNEL_UT_WARM=1` times one copy on warm caches.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
