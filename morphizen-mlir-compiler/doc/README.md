<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Compiler Backend

MLIR-based ahead-of-time (AOT) compilation backend for onnx-hipdnn-ep.

## Architecture

MLIR dialects → mlir-hip-compiler → HIP backend → DLL generation → ONNX Runtime EP

## Documentation

- **[Building & Testing](BUILDING.md)** - Build instructions and testing guide
- **[Architecture](design/ARCHITECTURE.md)** - System architecture and design decisions
- **[Why GenerateInterfacePass](WHY-GENERATEINTERFACEPASS.md)** - Explanation of interface generation
- **[ONNX-MLIR Fork Summary](ONNX_MLIR_FORK_SUMMARY.md)** - ONNX-MLIR integration details

## Building

```bash
cmake -S . -B ../build/onnx-hipdnn-ep -DBUILD_MLIR_BACKEND=ON
cmake --build ../build/onnx-hipdnn-ep --config Debug --parallel
```

## Testing

```bash
# LIT tests
ctest --test-dir ../../build/$(basename $PWD) -R LitTests --verbose

# E2E tests
ctest --test-dir ../../build/$(basename $PWD) -R "CompileDemoConvDLL|TestDemoConvDLL" --verbose
```
