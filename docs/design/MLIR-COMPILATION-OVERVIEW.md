<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Compilation Overview

**Date:** 2026-03-02
**Document Type:** Design Document
**Review Status:** Draft
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [mlir/](mlir/)

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
│ Stage 1: hip-add-context-arg                     │
│ - Prepend !hip.context as arg 0 to all funcs     │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Stage 2: convert-onnx-to-hip                     │
│ - Convert ONNX ops → HIP ops (tensor mode)       │
│ - Extract constants to globals                   │
│ - Generate constant registry                     │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Stage 3: one-shot-bufferize                      │
│ - Tensor → memref, insert memref.alloc           │
│ - In-place for relu/cast (no extra alloc)        │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Stage 4: buffer-results-to-out-params            │
│ - Return memrefs → output arguments              │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Stage 5: canonicalize                            │
│ - Remove dead memref.alloc, fold constants       │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Stage 6: memory-pooling (optional)               │
│ - Strip packing: replace memref.alloc with       │
│   memref.view of hip.get_pool                    │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Stage 7: convert-hip-to-llvm                     │
│ - Lower HIP ops → MIOpen/hipBLAS calls           │
│ - hip.get_pool → hipdnn_ep_get_pool_base         │
│ - Convert types to LLVM                          │
└──────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────┐
│ Stage 8: generate-interface                      │
│ - Generate inference_init/compute/cleanup        │
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

For detailed transformation at each stage, see [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md).

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
- Allocates intermediate buffer pool (if memory-pooling enabled)
- Returns state pointer via out parameter

### 2. inference_compute
```c
int inference_compute(void* state, span_t* inputs, span_t* outputs);
```
- Parses input/output tensors from `span_t`
- Executes GPU operations
- Returns status code (0 = success)

### 3. inference_cleanup
```c
int inference_cleanup(void* state);
```
- Frees GPU constant memory
- Frees buffer pool
- Destroys GPU handles
- Frees state structure

---

### Parameter Details

**About the `state` parameter:** An opaque pointer (`void*`) representing the execution context. Internally contains GPU handles (stream, MIOpen, hipBLAS), pre-uploaded constant pointers, and pool base pointer. Allocated once in `init`, used throughout execution, freed in `cleanup`. See [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) for details.

**About `span_t` and tensor interface:** See [mlir/INTERFACE-DESIGN.md#prerequisite-5-tensor-interface](mlir/INTERFACE-DESIGN.md#prerequisite-5-tensor-interface).

---

## Detailed Design Documents

| Document | Description |
|----------|-------------|
| [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) | Pass pipeline and transformation stages |
| [02b-OneShotBufferize.md](mlir/passes/02b-OneShotBufferize.md) | Bufferization pipeline (one-shot-bufferize) |
| [mlir/passes/04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) | Memory pooling (strip packing) |
| [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) | C interface and GenerateInterfacePass prerequisites |
| [mlir/HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) | HIP dialect types and operations |
| [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) | Constant handling (globals, upload, retrieval) |
| [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) | Runtime state/context structure |
| [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) | Dynamic shape support |

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) - Runtime state structure and lifecycle
- [DEMO.md](../guides/DEMO.md) - End-to-end demo walkthrough
