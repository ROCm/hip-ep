<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Compilation Overview

**Date:** 2026-02-14
**Document Type:** Design Document
**Review Status:** Self-Reviewed
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [mlir/](mlir/)

---

## Purpose

This document provides a high-level overview of how ONNX models are compiled through MLIR to native DLL code for AMD ROCm GPUs. For detailed designs, see the documents in `doc/mlir/`.

---

## Compilation Pipeline Overview

```
ONNX Model (.onnx)
  ↓
┌──────────────────────────────────────────────────┐
│ MorphiZen Framework (existing)                   │
│ - Parse ONNX to MLIR (func, arith, ONNX dialect) │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Pass 1: OnnxToHip                                │
│ - Convert ONNX operations → HIP operations       │
│ - Extract constants to globals                   │
│ - Generate @main and constant helpers            │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ BufferDeallocation (MLIR standard)               │
│ - Insert hip.free operations                     │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ MemoryPoolingPass                                │
│ - Graph coloring assigns pool offsets            │
│ - 60% memory savings for demo model              │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Pass 2: HipToLLVM                                │
│ - Lower HIP operations → MIOpen/hipBLAS calls    │
│ - Generate wrapper functions                     │
│ - Convert types to LLVM                          │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Pass 3: GenerateInterfacePass                    │
│ - Generate C interface wrappers                  │
│ - inference_init/compute/cleanup                 │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ LLVM Backend                                     │
│ - Translate to LLVM IR                           │
│ - Compile to native object file                  │
│ - Link to DLL                                    │
└──────────────────────────────────────────────────┘
  ↓
Native DLL (inference.dll / inference.so)
  ↓
Embedded in ONNX EPContext
```

For detailed transformation through each stage, see [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md).

---

## Output: 3-Function Interface

The compiled DLL exports exactly 3 functions. For complete interface specification including data structures, error codes, and design rationale, see [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md).

### 1. inference_init
```c
int inference_init(void** out_state);
```
- Allocates runtime state (context)
- Creates GPU handles (stream, MIOpen, hipBLAS)
- Uploads constants (weights) to GPU
- Returns state pointer via out parameter

### 2. inference_compute
```c
int inference_compute(void* state, span_t* inputs, span_t* outputs);
```
- Parses input/output tensors from span_t
- Executes GPU operations (all inline)
- Returns status code (0 = success)

### 3. inference_cleanup
```c
int inference_cleanup(void* state);
```
- Frees GPU constant memory
- Destroys GPU handles
- Frees state structure

---

### Parameter Details

**About the `state` parameter:** An opaque pointer (`void*`) representing the execution context. Internally contains GPU handles (stream, MIOpen, hipBLAS) and pre-uploaded constant pointers. Allocated once in `init`, used throughout execution, freed in `cleanup`. See [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) for details.

**About `span_t` and tensor interface:**

The C interface uses `tensor_t` and `span_t` structs to pass tensors between CustomOp and compiled DLL. For complete struct definitions, see [mlir/INTERFACE-DESIGN.md#prerequisite-5-tensor-interface](mlir/INTERFACE-DESIGN.md#prerequisite-5-tensor-interface).

---

## Two-Layer Architecture

The compiled DLL has two layers: C interface (public API) and internal MLIR functions (private implementation). For complete architectural details and rationale, see [mlir/INTERFACE-DESIGN.md#two-layer-architecture](mlir/INTERFACE-DESIGN.md#two-layer-architecture).

---

## Interface Design for Runtime Shapes

Interface designed to accept runtime dimension values:

- **Rank**: Compile-time known (e.g., 4D tensor)
- **Dimensions**: Loaded from tensor_t.shape pointer
- **Strides**: Calculated from dimension values

**Status:** Dynamic shapes not yet implemented (memory pooling incompatibility).

See [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) for implementation challenges.

---

## Compilation to DLL

**TODO**: Document LLVM IR generation, native compilation, linking, and EPContext embedding details.

---

## Detailed Design Documents

| Document | Description |
|----------|-------------|
| [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) | Pass pipeline and transformation stages |
| [mlir/passes/04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) | Memory pooling optimization (60% savings) |
| [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) | C interface and GenerateInterfacePass prerequisites |
| [mlir/HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) | HIP context, types, and wrapper functions |
| [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) | Constant handling (globals, upload, retrieval) |
| [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) | Runtime state/context structure, static library design |
| [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) | Dynamic shape support |

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) - Runtime state structure and lifecycle
- [DEMO.md](../guides/DEMO.md) - End-to-end demo walkthrough
