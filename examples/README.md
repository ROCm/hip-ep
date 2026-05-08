<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Examples

Self-contained demo programs and sample workloads referenced by the
project documentation. New examples should each live in their own
subdirectory under this folder, alongside a short `README.md` that
explains what the example does, how to run it, and which doc section
references it.

## Index

| Example | Path | Topic |
|---------|------|-------|
| Quick start toy model | [`quickstart/`](./quickstart/) | Generates a small Conv ONNX model used by [`docs/quick_start.md`](../docs/quick_start.md) for the end-to-end smoke test. |

## Legacy flat layout

The HIP-dialect / MLIR test programs at the top level of this directory
(`*.hip.mlir` plus matching `main_*.cpp` drivers) pre-date the
"one-subdirectory-per-example" convention above. Each `.mlir` is a small
program exercising one HIP-dialect operator end-to-end, and the matching
`main_*.cpp` driver loads and runs the compiled artifact:

| Operator family | MLIR | Driver |
|-----------------|------|--------|
| Element-wise add | [`add.hip.mlir`](./add.hip.mlir) | [`main_add.cpp`](./main_add.cpp) |
| Element-wise mul | [`mul.hip.mlir`](./mul.hip.mlir) | [`main_mul.cpp`](./main_mul.cpp) |
| GEMM | [`gemm.hip.mlir`](./gemm.hip.mlir) | [`main_gemm.cpp`](./main_gemm.cpp) |
| Softmax | [`softmax.hip.mlir`](./softmax.hip.mlir) | [`main_softmax.cpp`](./main_softmax.cpp) |
| RMS normalization | [`rms_norm.hip.mlir`](./rms_norm.hip.mlir) | [`main_rms_norm.cpp`](./main_rms_norm.cpp) |
| Attention block | [`attention.hip.mlir`](./attention.hip.mlir) | [`main_attention.cpp`](./main_attention.cpp) |
| Full transformer (E2E) | (sourced from `test/e2e/`) | [`main_e2e.cpp`](./main_e2e.cpp) |

These run as part of the broader test infrastructure; see
[`test/e2e/README.md`](../test/e2e/README.md) for build and run details.
A future cleanup may relocate them into a dedicated subdirectory (for
example `examples/hip_dialect/`) so the top level here matches the
convention used by other MLIR-family projects.
