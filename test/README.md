<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# UDNA Compiler Tests

Quick reference for running all test suites in the udna-compiler.

**IMPORTANT**: All commands assume you're at the project root directory.

## Quick Start

```bash
ctest --test-dir ../build/$(basename $PWD)/udna-compiler --verbose
```

## Running Tests

### All Tests (Recommended)

```bash
# Via CTest
ctest --test-dir ../build/$(basename $PWD)/udna-compiler --verbose

# Via CMake target
cmake --build ../build/$(basename $PWD) --target check-onnx-hip-lit
```

### By Category

```bash
# LIT tests only (MLIR pass validation)
ctest --test-dir ../build/$(basename $PWD)/udna-compiler -R LitTests --verbose

# E2E tests only (ORT integration)
ctest --test-dir ../build/$(basename $PWD)/udna-compiler -R OrtIntegration --verbose
```

### Specific Tests

```bash
# Single LIT test via llvm-lit
llvm-lit -v udna-compiler/test/lit/Conversion/onnx-to-hip/test_gemm_basic.mlir
```

## Prerequisites

### LIT Tests
- **LLVM/MLIR**: Auto-fetched by CMake if not found
- **Python + lit**: `pip install lit`
- **FileCheck**: Provided by LLVM
- **udna-opt**: Built by this project

### E2E Tests
- **ONNX Runtime**: Pre-built in `../../local/`
- **GTest**: Auto-fetched by CMake if not found
- **Test Models**: Generated via Python scripts (see [test/e2e/README.md](e2e/))

## Build Configuration

Enable tests during CMake configuration:

```bash
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../build/$(basename $PWD) \
  -Dmorphizen_ENABLE_MLIR_COMPILER=ON \
  -DONNX_HIP_INCLUDE_LIT_TESTS=ON \
  -DBUILD_MOCK_RUNTIME=ON \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTABLEHLO_BUILD_TESTS=OFF \
  --fresh

# Build
cmake --build ../build/$(basename $PWD) --config Debug --parallel
```

**Key Options**:
- `morphizen_ENABLE_MLIR_COMPILER=ON` - **REQUIRED**: Enable udna-compiler build
- `ONNX_HIP_INCLUDE_LIT_TESTS=ON` - Enable LIT tests (default: ON)
- `BUILD_MOCK_RUNTIME=ON` - Enable E2E tests with mock runtime

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

## Common Issues

### LIT Tests

**Missing llvm-lit**:
```bash
pip install lit
```

**Tests fail to find udna-opt**:
```bash
# Ensure udna-opt is built
cmake --build ../build/$(basename $PWD) --target udna-opt --config Debug
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
- [CLAUDE.md](../CLAUDE.md) - Build system and project conventions
- [LLVM Testing Infrastructure](https://llvm.org/docs/TestingGuide.html)
- [FileCheck Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
