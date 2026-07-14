<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# hip-mlir-opt - MLIR Pass Testing and Debugging Tool

## Purpose

`hip-mlir-opt` is a specialized MLIR transformation tool for **development, debugging, and learning**. It allows you to run individual MLIR passes or the complete pipeline and inspect intermediate results.

## Use Cases

### 1. MLIR Pass Development
Test individual passes in isolation:
```bash
hip-mlir-opt input.mlir --convert-onnx-to-hip -o stage1.mlir
hip-mlir-opt stage1.mlir --convert-hip-to-llvm -o stage2.mlir
hip-mlir-opt stage2.mlir --generate-interface -o stage3.mlir
```

### 2. Learning MLIR Transformations
Understand what each pass does by inspecting outputs:
```bash
# See how ONNX operations are converted to HIP dialect
hip-mlir-opt demo.mlir --convert-onnx-to-hip

# See how HIP dialect is lowered to LLVM dialect
hip-mlir-opt demo.mlir --convert-onnx-to-hip --convert-hip-to-llvm

# Or use the complete pipelines
hip-mlir-opt demo.mlir --onnx-to-hip-pipeline --hip-to-llvm-pipeline
```

### 3. Debugging Compilation Issues
Identify which pass is failing:
```bash
hip-mlir-opt failing.mlir --convert-onnx-to-hip         # Works?
hip-mlir-opt failing.mlir --convert-onnx-to-hip \
                           --convert-hip-to-llvm          # Fails here?
```

### 4. Inspecting Intermediate MLIR
Save and examine intermediate representations:
```bash
hip-mlir-opt input.mlir --convert-onnx-to-hip > after_onnx_to_hip.mlir
# Manually inspect the file
hip-mlir-opt after_onnx_to_hip.mlir --convert-hip-to-llvm > after_hip_to_llvm.mlir
```

## Available Passes

- `--convert-onnx-to-hip` - Convert ONNX operations to HIP dialect
- `--convert-hip-to-llvm` - Convert HIP operations to LLVM dialect
- `--generate-interface` - Generate C-ABI interface functions
- `--hip-add-context-arg` - Add runtime context argument
- `--hip-pool-allocs` - Pack allocations into a single byte pool
- `--hip-lower-allocs` - Lower allocation operations
- `--hip-resolve-extern-constants` - Resolve external constant references

## Pipelines

- `--onnx-to-hip-pipeline` - Lower ONNX IR to bufferized HIP memref IR
- `--hip-to-llvm-pipeline` - Lower HIP memref IR to LLVM dialect and generate C interface

## Registered Dialects

- `builtin` - MLIR builtin operations
- `arith` - Arithmetic operations
- `func` - Function operations
- `memref` - Memory reference operations
- `scf` - Structured control flow operations
- `cf` - Control flow operations
- `tensor` - Tensor operations
- `bufferization` - Bufferization operations
- `llvm` - LLVM dialect
- `hip` - Custom HIP dialect
- `onnx` (stub) - ONNX dialect stub for testing

## Comparison with Other Tools

**hip-mlir-opt:**
- Purpose: Development and debugging
- Input: ONNX-MLIR or HIP dialect MLIR
- Output: Transformed MLIR (text)
- Use when: Learning, debugging, testing passes

**hip-compiler:**
- Purpose: Production compilation
- Input: ONNX-MLIR or LLVM dialect MLIR
- Output: LLVM bitcode (`.bc`) -- JITted in-process by the EP DLL
- Use when: Generating production artifacts

**hip-test:**
- Purpose: Run and validate a compiled per-model artifact end-to-end
- Input: LLVM bitcode (`.bc`) or native `.dll`/`.so` (detected by file extension)
- Output: Inference run + optional NaN/Inf output validation
- Use when: Smoke-testing an artifact through the same loader the EP uses

## Example Workflow

```bash
# Development workflow (step-by-step debugging)
hip-mlir-opt demo.mlir --convert-onnx-to-hip -o stage1.mlir
# Inspect stage1.mlir...
hip-mlir-opt stage1.mlir --convert-hip-to-llvm -o stage2.mlir
# Inspect stage2.mlir...
hip-mlir-opt stage2.mlir --generate-interface -o stage3.mlir
# Inspect stage3.mlir...

# Or use the complete pipelines
hip-mlir-opt demo.mlir --onnx-to-hip-pipeline --hip-to-llvm-pipeline -o output.mlir

# Production workflow (compile to bitcode)
hip-compiler demo.mlir -o output.bc
# Execution happens via the EP DLL's LlvmIrJit loader.
```

## Building

For build instructions, see
[docs/quick_start.md](../../docs/quick_start.md).

## See Also

- [Quick Start Guide](../../docs/quick_start.md) - Build and test instructions
- [hip-compiler](../hip-compiler/) - Production bitcode compilation tool
- [hip-test](../hip-test/) - Run/validate a compiled artifact (`.bc` or `.dll`/`.so`)
