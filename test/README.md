<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# HIP Compiler Tests

Quick reference for running all test suites in the hip-compiler.

**IMPORTANT**: All commands assume you're at the project root directory.

## Quick Start

```bash
ctest --test-dir ../build/$(basename $PWD) --verbose
```

## Running Tests

### All Tests (Recommended)

```bash
# Via CTest
ctest --test-dir ../build/$(basename $PWD) --verbose

# LIT tests via CMake target
cmake --build ../build/$(basename $PWD) --target check-hip-mlir-lit
```

### By Category

```bash
# LIT tests only (MLIR pass validation)
ctest --test-dir ../build/$(basename $PWD) -R MorphizenMLIRLitTests --verbose

# E2E tests only (hip-compiler + hip-test on test/lit/e2e/*.mlir)
ctest --test-dir ../build/$(basename $PWD) -R "^E2E_" --verbose
```

### Specific Tests

```bash
# Single LIT test via llvm-lit (point at the build tree, filter by path)
llvm-lit -v ../build/$(basename $PWD)/test/lit/ --filter onnx-to-hip/test_gemm
```

## Prerequisites

See [docs/quick_start.md](../docs/quick_start.md) for build prerequisites.

Additional test-specific dependencies:
- **Python + lit**: `pip install lit` (for LIT tests)

## Build Configuration

For build prerequisites and full build instructions, see
[docs/quick_start.md](../docs/quick_start.md).

Test-specific CMake flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `BUILD_HIP_UNIT_TESTS` | ON | Register LIT and E2E tests with CTest |
| `BUILD_MOCK_RUNTIME` | ON in CMake; `build.py` passes OFF unless `--mock` | Build the mock runtime (no GPU required) instead of the real one |

To disable HIP unit tests (LIT + E2E) during the build:

```bash
cmake -DBUILD_HIP_UNIT_TESTS=OFF ...
```

## Test Coverage

### LIT Tests (`test/lit/`)
- **Conversion passes**: ONNX → HIP, HIP → LLVM / TOSA lowering (default path), plus the opt-in hipDNN (when a hipDNN handle is available) and HIPSR (`HIPDNN_EP_HIPSR=1`) paths
- **Transform passes**: C-ABI wrapper generation, op-state slots, HIP fusion
- **Pipelines and plugins**: Multi-pass pipelines and compiler plugin hooks

See [test/lit/README.md](lit/) for detailed test writing guide and debugging tips.

### E2E Tests (`test/e2e/`)
- **Compile + execute**: Each `test/lit/e2e/*.mlir` is compiled with `hip-compiler` and run with `hip-test --validate`

See [test/e2e/README.md](e2e/) for running the suite and GPU setup.

### Numeric Tests (`test/numeric/`)
- **Per-operator correctness**: Single-op ONNX models (Sigmoid, MatMul, ...) run on the MorphiZen EP and compared against an ORT CPU reference.
- **Pluggable backends**: New backends register in `conftest.py::_BACKENDS`.
- **Reference cache**: Expensive CPU references (Llama Q/O / gate / up projections) are cached on disk, keyed by sanitised test name with a sha256 drift tripwire.

See [test/numeric/README.md](numeric/) for backends, CLI options, and the "Bring your own ONNX" pattern.

### Kernel compile-time guards (`test/kernel_guards/`)
- **GPU-free checks on device-kernel source**: `LdsBarrierArchGate` compiles the `HIPDNN_LDS_BARRIER` gate in `lib/Runtime/Kernels/include/hip_arch_compat.h` for gfx9 to gfx12 and checks the barrier each architecture emits.

The per-op kernel unit tests, which need a GPU, live in [lib/Runtime/Kernels/unit_test/](../lib/Runtime/Kernels/unit_test/README.md).

## Common Issues

### LIT Tests

**Missing llvm-lit**:
```bash
pip install lit
```

**Tests fail to find hip-mlir-opt**:
```bash
# Ensure hip-mlir-opt is built
cmake --build ../build/$(basename $PWD) --target hip-mlir-opt --config Debug
```

## References

- [LIT Tests Detailed Guide](lit/README.md) - Test writing, FileCheck patterns, debugging
- [E2E Tests Quick Guide](e2e/README.md) - Running the suite, GPU setup, troubleshooting
- [LLVM Testing Infrastructure](https://llvm.org/docs/TestingGuide.html)
- [FileCheck Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
