<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Memory Management Strategy

**Date:** 2026-02-14
**Document Type:** Design
**Review Status:** Draft

---

## Overview

GPU memory allocation is expensive (~35ms per GB). Current design uses memory pooling to minimize allocation overhead and memory footprint.

---

## Destination-Passing Style Design

All levels use destination-passing style: caller allocates output buffers, callee writes to them.

- **HIP dialect operations**: Take output buffer as argument
  - Example: `hip.conv(%ctx, %input, %weights, %bias, %output)`
- **HIP dialect functions**: Take output pointers as arguments, return status code
  - Example: `func.func @inference_compute(%state: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32`
- **C interface**: Take output pointers as arguments, return status code
  - Example: `int inference_compute(void* state, span_t* inputs, span_t* outputs)`

No memory is returned from functions - all outputs are written to caller-provided buffers.

**Note:** This design works for static shapes. Dynamic shape support requires resolving memory pooling incompatibility. See [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md).

---

## Memory Categories

| Type | Lifetime | Managed By | Example |
|------|----------|------------|---------|
| **Constants** | Session | Compiled code (weights embedded in DLL) | Conv weights: 37KB |
| **Intermediates** | Per-layer | Compiled code (pooled memory) | Activation: 3MB |
| **Workspace** | Session | Compiled code (shared scratch space) | MIOpen: 8MB |
| **Input/Output** | Per-call | Caller (CustomOp) | Model input: 600KB |

---

## Buffer Allocation Strategy

Single memory pool allocated in `inference_init()`, accessed via offsets in `inference_compute()`.

Graph coloring assigns offsets to buffers with non-overlapping lifetimes (60% memory savings: 12.8MB vs 32.1MB for demo model).

See [04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) for algorithm and [BUFFER-LIFETIME-DESIGN.md](BUFFER-LIFETIME-DESIGN.md) for deallocation placement.

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system design
- [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) - MLIR lowering pipeline
