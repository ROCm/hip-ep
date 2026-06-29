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

# Via CMake target
cmake --build ../build/$(basename $PWD) --target check-onnx-hip-lit
```

### By Category

```bash
# LIT tests only (MLIR pass validation)
ctest --test-dir ../build/$(basename $PWD) -R LitTests --verbose

# E2E tests only (ORT integration)
ctest --test-dir ../build/$(basename $PWD) -R OrtIntegration --verbose
```

### Specific Tests

```bash
# Single LIT test via llvm-lit
llvm-lit -v onnx-hip-ep/test/lit/Conversion/onnx-to-hip/test_gemm_basic.mlir
```

## Prerequisites

See [docs/quick_start.md](../docs/quick_start.md) for build prerequisites.

Additional test-specific dependencies:
- **Python + lit**: `pip install lit` (for LIT tests)
- **Test Models**: Generated via Python scripts (see [test/e2e/README.md](e2e/))

## Build Configuration

For build prerequisites and full build instructions, see
[docs/quick_start.md](../docs/quick_start.md).

Test-specific CMake flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `BUILD_HIP_UNIT_TESTS` | ON | Register LIT and E2E tests with CTest |
| `ONNX_HIP_INCLUDE_LIT_TESTS` | ON | Enable LIT tests |
| `BUILD_MOCK_RUNTIME` | ON | Enable E2E tests with mock runtime |

To disable HIP unit tests (LIT + E2E) during the build:

```bash
cmake -DBUILD_HIP_UNIT_TESTS=OFF ...
```

## Test Coverage

### LIT Tests (`test/lit/`)
- **Conversion passes**: ONNX→HIP, HIP→LLVM dialect lowering
- **Transform passes**: Buffer deallocation, C-ABI wrapper generation
- **Integration**: Multi-pass compilation pipelines

See [test/lit/README.md](lit/) for detailed test writing guide and debugging tips.

### E2E Tests (`test/e2e/`)
- **ORT Integration**: EP registration, session creation
- **Model Execution**: Conv and Conv+Gemm models with CPU device

See [test/e2e/README.md](e2e/) for model generation and environment setup.

### Numeric Tests (`test/numeric/`)
- **Per-operator correctness**: Single-op ONNX models (Sigmoid, MatMul, ...) run on the MorphiZen EP and compared against an ORT CPU reference.
- **Pluggable backends**: New backends register in `conftest.py::_BACKENDS`.
- **Reference cache**: Expensive CPU references (Llama Q/O / gate / up projections) are cached on disk, keyed by sanitised test name with a sha256 drift tripwire.

See [test/numeric/README.md](numeric/) for backends, CLI options, and the "Bring your own ONNX" pattern.

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

### E2E Tests

**Test skips with "V2 API not implemented"**:
```bash
# Set environment variable
export MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1  # Linux/Mac
set MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1     # Windows CMD
```

**Missing test models**:
```bash
# Generate models (see test/e2e/README.md)
cd test/e2e
python gen_conv_model.py --two-layer --output /path/to/build/Debug/bin/conv_model.onnx
```

## References

- [LIT Tests Detailed Guide](lit/README.md) - Test writing, FileCheck patterns, debugging
- [E2E Tests Quick Guide](e2e/README.md) - Model generation, environment setup, troubleshooting
- [LLVM Testing Infrastructure](https://llvm.org/docs/TestingGuide.html)
- [FileCheck Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
