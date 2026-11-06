# gemm_transposed_b Xbf16_Wbf16_Yfp32 standalone test

Standalone `hipcc`-only test for the bf16-input, fp32-output WMMA GEMM
`C[M,N] = A[M,K] * transpose(B[N,K])`, `hip_gemm_bf16_f32_transposed_b`, which
the runtime runs for bf16 -> fp32 projections with more than one row on
gfx1151. It does not configure or build hip-ep, ONNX Runtime, LLVM, or MLIR.
`make` compiles `gemm_wmma_kernel.hip` into the leaf; the CMake package build
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

Against a double-precision CPU reference on the bf16 values (every checked
output within `(K / 16 + 32)` units of fp32 rounding of the sum of the
products' magnitudes; on gfx1151 the largest error over these cases is 24
units):

- both tile shapes (64x64 below `M = 128`, 128x64 from it), the smallest `M`
  the runtime sends to this GEMM (2), outputs that do not fill a whole number
  of tiles in `M` or `N` (down to `N = 1`), `K` a multiple of the 64-wide
  slice and not, and the smallest `K` (8);
- a launch right after a failed HIP call, on each tile shape, which must not
  report that call's error as its own;
- `M`, `N` or `K` = 0, a negative `M` or `N`, `K` not a multiple of 8, a null
  stream, input or output pointer, and an input or weight pointer 8 bytes off
  16-byte alignment, each of which must return -1 without writing;
- output guard regions, and a NaN fill that fails any output left unwritten.

Cases with more than `4e8` multiply-adds compare a sample of the rows (the
first and last row of every 64-row tile and every 7th row, each in full) with
the reference, and check that every other output was written.

## Shape coverage (3 tiers)

Every tier runs all of the above (8 shape cases, 2 stale-error cases and 12
edge arguments). `COVERAGE` only thins the typical prefill projection sizes
(`MxNxK`): tier1 adds `512x3072x3072`, tier2 adds `512x1024x3072` and
`128x8192x3072`, tier3 (default) adds `512x3072x8192` and `512x14336x4096`.
At startup the exe prints `gemm_transposed_b bf16 coverage=N cases=<n>`
(23 / 25 / 27); a successful run prints one `PASS` line per case and ends with
`ALL PASS`.

The kernel implements only the gfx11 WMMA encoding. On gfx1170 and gfx12
(wave32, gfx12-style WMMA only) it would trap, and the launcher refuses wave64
devices, so on those the exe prints `SKIP` and exits 0.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `M512_N3072_K3072`, `config` the tile shape the launcher picks
(`tile64x64` or `tile128x64`), `relL2` against the reference over the checked
outputs, and `time_ms` the mean of 20 back-to-back launches after 3 warm-up
launches, on warm caches.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
