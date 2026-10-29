# qmoe_decode Xfp16_Wu4_Yfp16 standalone test

Standalone `hipcc`-only test for the fused single-token QMoE decode,
`hip_qmoe_decode_fused` and its W4A8 variant `hip_qmoe_decode_fused_dp4a`:
fp16 input, 4-bit expert weights with fp16 scales, fp16 output. It does not
configure or build hip-ep, ONNX Runtime, LLVM, or MLIR. `make` compiles
`qmoe_kernel.hip` into the leaf; the CMake package build links
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

Every case runs FC1 + SwiGLU, FC2 and the weighted sum over k distinct
experts, and checks:

- the first call against a double-precision reference that rounds to fp16
  where the kernels store (SwiGLU output, weighted slot, output) and, for
  dp4a, quantizes the activations to per-group int8 the way the kernels do.
  Each element must be within `4e-3` of itself plus `2e-3` of the largest
  output (the k slots are fp16 before they are summed, so an element where
  they cancel may be off by an ulp of the slots), and relL2 within `1e-3`;
- 128 more calls on the same inputs, each bit-identical to the first. The
  first call of a shape picks the GEMV tiling for the process, and the block
  size sets the reduction order, so a tiling that changed after the first
  call would show here;
- output guard regions, and a NaN fill that fails any element left unwritten.

Categorical situations, at every tier: fp and dp4a, block size 32, 64 and
128, zero points absent (8) and present, bias absent and present. Only block
sizes that are multiples of 32 are run: both GEMVs read one group's scale and
zero point per 32 values.

Also at every tier:

- the first call of an untimed shape made while an earlier failed HIP call's
  error is pending, for fp and dp4a, must return 0 with a correct, repeatable
  result. Without the decode clearing the error on entry, dp4a reports it as
  its own; fp's first tiling launch consumes it, so fp still succeeds but
  keeps the default tiling, which this test cannot see;
- an fp32 element size, an odd block size, a dp4a block size of 48, a hidden
  size of 48 and k = 0 must return -1 and write nothing.

## Shape coverage (3 tiers)

`COVERAGE` only thins the shapes (hidden x inter, experts, k): tier1 =
`64 x 64`, 4 experts, k 2; tier2 adds GPT-OSS-20B (`2880 x 2880`, 32, 4);
tier3 (default) adds `2048 x 768`, 128, 8. Each shape runs all 24 categories.
At startup the exe prints `qmoe_decode coverage=N cases=<n>` (32 / 56 / 80);
a successful run prints one `PASS` line per case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

The decode has no autotune table: it times its GEMV tiling candidates on the
first call of each shape. `MODE` is accepted for consistency with other
leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `H2880_I2880_E32_k4`, `config` like `dp4a_bs32_zp_bias`, `relL2`
against the reference, and `time_ms` the mean of 50 back-to-back calls after
the repeat check, on warm caches (the same experts every call).

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
