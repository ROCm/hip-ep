# qmoe_ragged_mm Xfp16/Wu4/Yfp16 standalone test

This leaf directly compiles `qmoe_ragged_mm_kernel.hip` and a self-contained
C++ driver. It does not configure or build hip-ep, ONNX Runtime, LLVM, or MLIR.

The current coverage validates the persistent LDS-staged FP16-WMMA ragged
MatMul primitive against an in-process CPU reference:

- balanced, empty-expert, and highly skewed routing;
- FC1-style indirect/gather input and FC2-style contiguous input;
- bias disabled and enabled;
- UINT4 weights with symmetric zero point 8 and unpacked asymmetric UINT8
  per-group zero points;
- row-group coverage and output guard regions;
- adaptive task selection: vectorized GEMV for 1..7 rows, WMMA16 for 8..31
  rows, and WMMA64 for 32..64 rows, including partial M/N/K tails;
- device bucket counts/offsets, `pair_to_sorted`, four-field adaptive row-group
  descriptors, and queue initialization;
- occupancy-capped persistent launch grids, including task-bound and
  occupancy-bound cases;
- one ragged FC1 launch, one SwiGLU launch, one ragged FC2 launch, followed by
  deterministic sorted reduction (20 byte-identical repetitions).

`COVERAGE=1` runs one small shape across every categorical case. Higher tiers
add shapes while preserving all combinations of routing, input layout, bias,
and zero-point mode. `COVERAGE=3` additionally runs exact GPT-OSS-20B FC1
(`K=2880,N=5760`) and FC2 (`K=2880,N=2880`) dimensions.

```text
make test HIP_SDK=C:/ROCm_workspace/therock-dist \
  OFFLOAD=--offload-arch=gfx1151 COVERAGE=1

make test_custom HIP_SDK=C:/ROCm_workspace/therock-dist \
  OFFLOAD=--offload-arch=gfx1151 ARGS="--coverage 3"

make clean
```

`MODE` is accepted for consistency with other leaves but this primitive has no
autotune LUT. A successful run prints one `PASS relL2=...` line per case and
ends with `ALL PASS`.
