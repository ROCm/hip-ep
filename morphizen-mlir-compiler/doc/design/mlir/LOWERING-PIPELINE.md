<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Lowering Pipeline

**Date:** 2026-02-20
**Document Type:** Design
**Status:** Draft
**Related:** [MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md)

---

## Overview

This document shows the complete MLIR lowering pipeline: all passes in execution order, from ONNX input to executable shared library output.

For detailed implementation of each pass, click the links in the "Document" column below.

---

## Complete Pipeline

Implemented in `populateCompleteOnnxToLLVMPipeline()`.

| Stage | Pass/Transformation | Input Dialect | Output Dialect | Scope | Document | Purpose |
|-------|---------------------|---------------|----------------|-------|----------|---------|
| **1** | **ONNX-MLIR (External)** | `.onnx` file | ONNX dialect | — | MorphiZen | Frontend conversion |
| **2** | **OnnxToHip** | ONNX dialect | HIP dialect | Module | [01-OnnxToHip.md](passes/01-OnnxToHip.md) | Dialect conversion, constant extraction |
| **3** | **Buffer Management** | HIP dialect | HIP dialect | func.func | [02-BufferDeallocation.md](passes/02-BufferDeallocation.md) | Automatic memory management |
| 3.1 | ├─ BufferLoopHoisting | HIP | HIP | func.func | [02-BufferDeallocation.md](passes/02-BufferDeallocation.md) | Move allocations out of loops |
| 3.2 | ├─ OwnershipBasedBufferDeallocation | HIP | HIP | func.func | [02-BufferDeallocation.md](passes/02-BufferDeallocation.md) | Insert hip.free operations |
| 3.3 | └─ OptimizeAllocationLiveness | HIP | HIP | func.func | [02-BufferDeallocation.md](passes/02-BufferDeallocation.md) | Optimize buffer lifetimes |
| **4** | **Canonicalization** | HIP dialect | HIP dialect | Module | [03-Canonicalization.md](passes/03-Canonicalization.md) | Constant folding, copy elimination |
| **5** | **MemoryPooling** | HIP dialect | HIP dialect | Module | [04-MemoryPooling.md](passes/04-MemoryPooling.md) | Graph coloring (60% memory savings) |
| **6** | **HipToLLVM** | HIP dialect | LLVM dialect | Module | [05-HipToLLVM.md](passes/05-HipToLLVM.md) | Dialect lowering, wrapper generation |
| **7** | **GenerateInterface** | LLVM dialect | LLVM dialect | Module | [06-GenerateInterfacePass.md](passes/06-GenerateInterfacePass.md) | C interface generation |
| **8** | **LLVM Backend (External)** | LLVM dialect | Object code | — | LLVM toolchain | IR → object → shared library |

**Key dialect transformations:**
- **ONNX → HIP** (Stage 2): `tensor<...>` → `memref<...>`, functional → imperative
- **HIP → LLVM** (Stage 6): `!hip.context` → `!llvm.ptr`, memref → structs, operations → wrapper calls
- **LLVM → Object** (Stage 8): LLVM IR → machine code, link with HIP/MIOpen/hipBLAS

**Scope meanings:**
- **Module**: Pass operates on entire MLIR module
- **func.func**: Pass operates on individual function operations (nested)
