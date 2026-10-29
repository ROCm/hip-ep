# cast Xfp32_Ybf16 standalone test

Standalone `hipcc`-only test for `hip_cast` from fp32 to bf16. It does not
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

Every output is compared bit-exactly with a round-to-nearest-even reference;
a NaN input must give a bf16 NaN of the same sign:

- special values: signed zeros, ties that round down and up to even, the
  largest floats (which round to infinity) and the largest that stays finite,
  infinities, quiet and signaling NaNs of both signs, and subnormals;
- every sign, exponent and upper mantissa (the 65536 bf16 patterns), each
  with six low halves: exact, one ulp above, just below, at and just above
  the halfway point, and the top of the interval (393216 values);
- random bit patterns at 1, 255, 256, 257 and 65536 elements, around the
  256-thread block;
- each input set with aligned pointers and with pointers one element off
  16-byte alignment;
- a launch right after a failed HIP call, which must not report that call's
  error as its own;
- `num_elements` 0 and negative (return 0), and fp64 and int32 inputs
  (unsupported pairs, return -1), none of which may write;
- output guard regions, and a NaN fill that fails any output left unwritten.

## Shape coverage (3 tiers)

Every tier runs all of the above (14 input cases, 1 stale-error case and 4
edge calls). `COVERAGE` only adds larger random tensors: tier2 adds
`4194307` elements, tier3 (default) also `16777217`, each aligned and not.
At startup the exe prints `cast f32->bf16 coverage=N cases=<n>`
(19 / 21 / 23); a successful run prints one `PASS` line per case and ends with
`ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `exponent_sweep_misaligned` or `random_n257`, `config`
`f32_to_bf16`, `relL2` 0 on a bit-exact pass and 1 otherwise, and `time_ms`
the mean of 50 back-to-back launches after 3 warm-up launches, on warm
caches. A launch-to-launch gap is included, so for the smallest tensors it is
an upper bound on the kernel time rather than the kernel time itself.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
