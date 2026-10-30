# gemv_transposed_b Xbf16_Wbf16_Yfp32 standalone test

Standalone `hipcc`-only test for the bf16-input, fp32-output GEMV
`C[1,N] = A[1,K] * transpose(B[N,K])`, `hip_gemv_bf16_f32_transposed_b`, which
the runtime runs for one-row bf16 -> fp32 projections with `N >= 1024`,
`K >= 256` and even `K` on gfx1151. It does not configure or build hip-ep, ONNX
Runtime, LLVM, or MLIR. `make` compiles `gemm_wmma_kernel.hip` into the leaf;
the CMake package build links `custom_kernels_<arch>.dll` instead. Inputs and
the CPU reference are generated in-process -- no python, no data files.

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

Against a double-precision CPU reference on the bf16 values (every output
within `(ceil(K / 16) + 32)` units of fp32 rounding of the sum of the
products' magnitudes, a bound on the error of the kernel's fp32 sums):

- the smallest shape the runtime sends to this GEMV (`1024x256`), outputs that
  do not fill a whole number of blocks, and a reduction shorter than one wave
  (`K = 2`);
- both kernel bodies (128-thread blocks below `K = 8192`, 512-thread blocks
  from it), each with and without a remainder after the unrolled loop;
- a launch right after a failed HIP call, on each body, which must not report
  that call's error as its own;
- `N = 0`, `K = 0`, a negative `N` or `K`, an odd `K`, a null stream, input or
  output pointer, and an input or weight pointer one element off 4-byte
  alignment, each of which must return -1 without writing;
- output guard regions, and a NaN fill that fails any output left unwritten.

## Shape coverage (3 tiers)

Every tier runs all of the above (8 shape cases, 2 stale-error cases and 11
edge arguments). `COVERAGE` only thins the typical decode projection sizes
(`NxK`): tier1 adds `3072x3072`, tier2 adds `8192x3072` and `3072x8192`, tier3
(default) adds `14336x4096` and `4096x14336`. At startup the exe prints
`gemv_transposed_b bf16 coverage=N cases=<n>` (22 / 24 / 26); a successful
run prints one `PASS` line per case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `N3072_K8192`, `config` the body the launcher picks
(`block128_acc4` or `block512_acc2`), `relL2` against the reference, and
`time_ms` the mean of 50 back-to-back launches after 3 warm-up launches, on
warm caches. A launch-to-launch gap is included, so for the smallest shapes it
is an upper bound on the kernel time rather than the kernel time itself.

Built for gfx1151, the kernel computes each pair with the bf16 dot
instruction (`v_dot2_f32_bf16`), which does not round to nearest even: its
result can differ from the correctly rounded one by an ulp or two even when
every operand and the exact result are small integers. The test therefore
never compares outputs bit for bit. Built for any other ISA (for example
`gfx11-generic`, which the packaged tests use on gfx1151) the kernel uses two
fp32 FMAs per pair instead; both stay within the bound above, with about the
same error against the reference.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
