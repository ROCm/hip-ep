<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Lowering Pipeline

**Date:** 2026-03-02
**Document Type:** Design
**Status:** Draft
**Related:** [MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md)

---

## Overview

Complete MLIR lowering pipeline in execution order, from ONNX input to executable shared library output.

For detailed implementation of each pass, click the links in the "Document" column below.

---

## Complete Pipeline

Implemented in `populateCompleteOnnxToLLVMPipeline()`.

| Stage | Pass/Transformation | Input Dialect | Output Dialect | Scope | Document | Purpose |
|-------|---------------------|---------------|----------------|-------|----------|---------|
| **0** | **ONNX-MLIR (External)** | `.onnx` file | ONNX dialect | — | MorphiZen | Frontend conversion |
| **1** | **hip-add-context-arg** | ONNX dialect | ONNX dialect | Module | [01-HipAddContextArg.md](passes/01-HipAddContextArg.md) | Prepend `!hip.context` arg to all functions |
| **2** | **convert-onnx-to-hip** | ONNX dialect | HIP dialect (tensor) | Module | [02-OnnxToHip.md](passes/02-OnnxToHip.md) | Tensor-first dialect conversion, constant extraction |
| **3** | **one-shot-bufferize** | HIP dialect (tensor) | HIP dialect (memref) | Module | [02b-OneShotBufferize.md](passes/02b-OneShotBufferize.md) | Tensor → memref, insert `memref.alloc` with ownership |
| **4** | **buffer-results-to-out-params** | HIP dialect (memref) | HIP dialect (memref) | Module | [02b-OneShotBufferize.md](passes/02b-OneShotBufferize.md) | Convert return memrefs to output arguments |
| **5** | **canonicalize** | HIP dialect (memref) | HIP dialect (memref) | Module | [03-Canonicalization.md](passes/03-Canonicalization.md) | Constant folding, dead memref.alloc elimination |
| **6** | **memory-pooling** | HIP dialect (memref) | HIP dialect (memref) | Module | [04-MemoryPooling.md](passes/04-MemoryPooling.md) | Strip-packing: replace `memref.alloc` with `memref.view` of pool |
| **7** | **convert-hip-to-llvm** | HIP dialect (memref) | LLVM dialect | Module | [05-HipToLLVM.md](passes/05-HipToLLVM.md) | Dialect lowering, wrapper generation |
| **8** | **generate-interface** | LLVM dialect | LLVM dialect | Module | [06-GenerateInterfacePass.md](passes/06-GenerateInterfacePass.md) | C interface generation |
| **9** | **LLVM Backend (External)** | LLVM dialect | Object code | — | LLVM toolchain | IR → object → shared library |

**Key dialect transformations:**
- **ONNX → HIP (tensor)** (Stage 2): `onnx.Conv` → `hip.conv(ins..., outs tensor.empty())`
- **HIP (tensor) → HIP (memref)** (Stage 3): `one-shot-bufferize` converts tensor DPS ops to memref
- **HIP (memref) → LLVM** (Stage 7): `!hip.context` → `!llvm.ptr`, `hip.get_pool` → runtime call, memrefs → structs

**Scope meanings:**
- **Module**: Pass operates on entire MLIR module
- **func.func**: Pass operates on individual function operations (nested)
