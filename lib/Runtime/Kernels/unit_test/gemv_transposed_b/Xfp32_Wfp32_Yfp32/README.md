# gemv_transposed_b Xfp32_Wfp32_Yfp32 standalone test

Standalone `hipcc`-only test for the fp32 GEMVs
`C[1,N] = A[1,K] * transpose(B[N,K])`: `hip_gemv_f32_transposed_b` (one wave
per output) and `hip_gemv_f32_transposed_b_multiwave_gfx1151` (one 256-thread
block per output, float4 or scalar body), which the runtime runs for one-row
fp32 projections with `N <= 256` on gfx1151. It does not configure or build
hip-ep, ONNX Runtime, LLVM, or MLIR. `make` compiles `gemm_wmma_kernel.hip`
into the leaf; the CMake package build links `custom_kernels_<arch>.dll`
instead. Inputs and the CPU reference are generated in-process -- no python,
no data files.

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

For each launcher, against a double-precision CPU reference (every output
within `(ceil(K / 32) + 16)` units of fp32 rounding of the sum of the
products' magnitudes, a bound on the error of any of the kernels' fp32 sums):

- the shapes the runtime sends to a GEMV (`1 <= N <= 256`, `K >= 256`): one
  output and several, `K` a multiple of 4 and not, up to `256x4096`;
- reductions shorter than one wave (`K = 1` and `31`), which both launchers
  also accept;
- each shape with aligned pointers and with pointers one element off 16-byte
  alignment, so the multi-wave launcher takes both its float4 and its scalar
  body;
- a launch right after a failed HIP call, which must not report that call's
  error as its own;
- `N = 0`, `K = 0`, a negative `N` and a null stream, input or output
  pointer, and for the multi-wave launcher `N = 257`, each of which must
  return -1 without writing;
- output guard regions, and a NaN fill that fails any output left unwritten.

## Shape coverage (3 tiers)

Every tier runs all of the above (32 shape cases, 2 stale-error cases and 15
edge arguments). `COVERAGE` only thins the typical decode projection sizes
(`NxK`): tier1 adds `128x2048`, tier2 adds `256x5120` and `32x14336`, tier3
(default) adds `256x14336` and, on the one-wave launcher only, `1024x4096`,
each on both launchers unless noted. At startup the exe prints
`gemv_transposed_b fp32 coverage=N cases=<n>` (51 / 55 / 58); a successful
run prints one `PASS` line per case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

These kernels have no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `multiwave_N256_K4096_misaligned`, `config` the kernel the
launcher picks (`one_wave`, `multiwave_vec4` or `multiwave_scalar`), `relL2`
against the reference, and `time_ms` the mean of 50 back-to-back launches after
3 warm-up launches, on warm caches. A launch-to-launch gap is included, so for
the smallest shapes it is an upper bound on the kernel time rather than the
kernel time itself.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
