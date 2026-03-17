<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# udna-opt - MLIR Pass Testing and Debugging Tool

## Purpose

`udna-opt` is a specialized MLIR transformation tool for **development, debugging, and learning**. It allows you to run individual MLIR passes or the complete pipeline and inspect intermediate results.

## Use Cases

### 1. MLIR Pass Development
Test individual passes in isolation:
```bash
udna-opt input.mlir --convert-onnx-to-hip -o stage1.mlir
udna-opt stage1.mlir --convert-hip-to-llvm -o stage2.mlir
udna-opt stage2.mlir --generate-interface -o stage3.mlir
```

### 2. Learning MLIR Transformations
Understand what each pass does by inspecting outputs:
```bash
# See how ONNX operations are converted to HIP dialect
udna-opt demo.mlir --convert-onnx-to-hip

# See how HIP dialect is lowered to LLVM dialect
udna-opt demo.mlir --convert-onnx-to-hip --convert-hip-to-llvm

# Or use the complete pipeline
udna-opt demo.mlir --morphizen-pipeline
```

### 3. Debugging Compilation Issues
Identify which pass is failing:
```bash
udna-opt failing.mlir --convert-onnx-to-hip         # Works?
udna-opt failing.mlir --convert-onnx-to-hip \
                           --convert-hip-to-llvm          # Fails here?
```

### 4. Inspecting Intermediate MLIR
Save and examine intermediate representations:
```bash
udna-opt input.mlir --convert-onnx-to-hip > after_onnx_to_hip.mlir
# Manually inspect the file
udna-opt after_onnx_to_hip.mlir --convert-hip-to-llvm > after_hip_to_llvm.mlir
```

## Available Passes

- `--convert-onnx-to-hip` - Convert ONNX operations to HIP dialect
- `--convert-hip-to-llvm` - Convert HIP operations to LLVM dialect
- `--generate-interface` - Generate C-ABI interface functions

## Registered Dialects

- `builtin` - MLIR builtin operations
- `arith` - Arithmetic operations
- `func` - Function operations
- `memref` - Memory reference operations
- `hip` - Custom HIP dialect
- `onnx` - ONNX dialect (from onnx-mlir)

## Comparison with Other Tools

**udna-opt:**
- Purpose: Development and debugging
- Input: ONNX-MLIR or HIP dialect MLIR
- Output: Transformed MLIR (text)
- Use when: Learning, debugging, testing passes

**udna-compile:**
- Purpose: Production compilation
- Input: ONNX-MLIR (with `--from-onnx-mlir`) or LLVM dialect MLIR
- Output: Native DLL
- Use when: Generating production artifacts

**test-model-dll:**
- Purpose: DLL testing and validation
- Input: Compiled DLL (from udna-compile)
- Output: Test results (PASSED/FAILED)
- Use when: Verifying compiled models, CI/CD testing

## Example Workflow

```bash
# Development workflow (step-by-step debugging)
udna-opt demo.mlir --convert-onnx-to-hip -o stage1.mlir
# Inspect stage1.mlir...
udna-opt stage1.mlir --convert-hip-to-llvm -o stage2.mlir
# Inspect stage2.mlir...
udna-opt stage2.mlir --generate-interface -o stage3.mlir
# Inspect stage3.mlir...

# Or use the complete pipeline
udna-opt demo.mlir --morphizen-pipeline -o output.mlir

# Production workflow (compile and test)
udna-compile demo.mlir -o output.dll --from-onnx-mlir
test-model-dll output.dll --verbose --validate
```

## Building

```bash
cmake -S . -B build -DBUILD_STANDALONE_TOOLS=ON
cmake --build build --target udna-opt
```

## See Also

- [DEMO.md](../../doc/guides/DEMO.md) - Complete demo of MLIR compilation pipeline
- [udna-compile](../udna-compile/) - Production DLL compilation tool
- [test-model-dll](../test-model-dll/) - DLL testing and validation tool
