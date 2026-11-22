# gather_block_quantized Xi64/Wu4/Yfp16 standalone test

Standalone `hipcc`-only test for `hip_gather_block_quantized` with int64
indices, 4-bit weights (two nibbles per byte, low nibble first, packed along
the last axis) and fp16 scales / output. It does not configure or build hip-ep,
ONNX Runtime, LLVM, or MLIR. `make` compiles `gather_block_quantized_kernel.hip`
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

Bit-exact against the in-process ONNX GatherBlockQuantized reference
(`(q - zp) * scale` in fp32, rounded once to fp16; every product is exact in
fp32, so both sides round the same value):

- full-table data and compact rows (`compact_rows=1`: only the gathered rows,
  in row-major index order, with garbage bytes standing in for out-of-range
  indices), on the same inputs;
- zero points absent (default 8 unsigned, 0 signed) and present (packed like
  the data);
- uint4 storage and, at no extra cost, int4;
- rank-1 and rank-2 indices that include the first and last rows, negative
  indices, and out-of-range indices on both sides (which must write 0);
- output guard regions, and a NaN fill that fails any element left unwritten.

## Shape coverage (3 tiers)

Every tier runs all 16 compact x zero-point x signedness x indices-rank
combinations on each selected shape. `COVERAGE` only thins the shape list
(rows x row length x block size x index count): tier1 = `1024x64 bs16`
with one index and `1024x160 bs16` with 16, tier2 adds `4096x2048 bs32` (16
indices) and `2048x960 bs32` (2048 indices), tier3 (default) adds
`8192x3840 bs128` (16 indices) and `4096x2048 bs64` (2048 indices). At startup
the exe prints `gather_block_quantized coverage=N cases=<n>` (32 / 64 / 96); a
successful run prints one `PASS relL2=...` line per case and ends with
`ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `R1024_C160_bs16_I2x8_compact_zp_signed`, `config` always
`fixed:elementwise` (one thread per output element) and `time_ms` 0: the leaf
checks correctness only.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
