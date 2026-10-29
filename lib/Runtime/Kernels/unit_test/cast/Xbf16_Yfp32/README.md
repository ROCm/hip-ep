# cast Xbf16_Yfp32 standalone test

Standalone `hipcc`-only test for `hip_cast` from bf16 to fp32. It does not
configure or build hip-ep, ONNX Runtime, LLVM, or MLIR. `make` compiles
`cast_kernel.hip` into the leaf; the CMake package build links
`custom_kernels_<arch>.dll` instead. Inputs and the CPU reference are
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

bf16 to fp32 is exact, so every output is compared bit-exactly with the
bf16 bits widened to the upper half of an fp32, NaN payloads included:

- every bf16 bit pattern (zeros, subnormals, normals, infinities and NaNs of
  both signs);
- random bit patterns at 1, 255, 256, 257 and 65536 elements, around the
  256-thread block;
- each input set with aligned pointers and with pointers one element off
  16-byte alignment;
- a launch right after a failed HIP call, which must not report that call's
  error as its own;
- `num_elements` 0 and negative (return 0), and fp64 and fp16 outputs
  (unsupported pairs, return -1), none of which may write;
- output guard regions, and a fill pattern no bf16 widens to, which fails any
  output left unwritten.

## Shape coverage (3 tiers)

Every tier runs all of the above (12 input cases, 1 stale-error case and 4
edge calls). `COVERAGE` only adds larger random tensors: tier2 adds
`4194307` elements, tier3 (default) also `16777217`, each aligned and not.
At startup the exe prints `cast bf16->f32 coverage=N cases=<n>`
(17 / 19 / 21); a successful run prints one `PASS` line per case and ends with
`ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `every_pattern_misaligned` or `random_n257`, `config`
`bf16_to_f32`, `relL2` 0 on a bit-exact pass and 1 otherwise, and `time_ms`
the mean of 50 back-to-back launches after 3 warm-up launches, on warm
caches. A launch-to-launch gap is included, so for the smallest tensors it is
an upper bound on the kernel time rather than the kernel time itself.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
