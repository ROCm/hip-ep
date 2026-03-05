<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Memory Management Strategy

**Date:** 2026-03-02
**Document Type:** Design
**Status:** Draft

---

## Overview

GPU memory allocation is expensive. The design minimizes allocation overhead and memory footprint by using a single pool allocated once per session.

---

## Destination-Passing Style

All levels use destination-passing style: caller allocates output buffers, callee writes to them.

- **HIP dialect ops** (memref mode): take output buffer as `outs` operand
  - Example: `hip.conv ins(%ctx, %input, %weights, %bias : ...) outs(%output : memref<...>)`
- **HIP dialect functions**: take output pointers as out-parameters (after `buffer-results-to-out-params`)
  - Example: `func.func @main_graph(%ctx, %input, %output)`
- **C interface**: take output pointers as arguments, return status code
  - Example: `int inference_compute(void* state, span_t* inputs, span_t* outputs)`

No memory is returned from functions — all outputs are written to caller-provided buffers.

---

## Memory Categories

| Type | Lifetime | Managed By | Example |
|------|----------|------------|---------|
| **Constants** | Session | DLL (weights embedded in `.data`) | Conv weights |
| **Intermediates** | Per-compute | Compiled code (pooled, one `hipMalloc`) | Activation buffers |
| **Input/Output** | Per-call | Caller (CustomOp) | Model input/output tensors |

---

## Buffer Allocation Strategy

`one-shot-bufferize` inserts `memref.alloc` for intermediate tensors. The optional `memory-pooling` pass replaces these with `memref.view` slices of a single pool buffer obtained via `hip.get_pool(%ctx)`.

Pool lifecycle:
- `inference_init`: single `hipMalloc(pool_size)` allocates the pool
- `inference_compute`: `hipdnn_ep_get_pool_base(state)` + static offsets locate each buffer
- `inference_cleanup`: single `hipFree(pool_base)` frees the pool

Strip packing assigns offsets so buffers with non-overlapping lifetimes share the same pool location.

See [mlir/passes/04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) for the algorithm and [mlir/passes/02b-OneShotBufferize.md](mlir/passes/02b-OneShotBufferize.md) for the bufferization pipeline.

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system design
- [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) - MLIR lowering pipeline
- [02b-OneShotBufferize.md](mlir/passes/02b-OneShotBufferize.md) - Bufferization pipeline
- [mlir/passes/04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) - Pool assignment algorithm
