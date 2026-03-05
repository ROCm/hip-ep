<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Memory Pooling Pass

**Date:** 2026-03-02
**Document Type:** Implementation
**Status:** Draft
**Related:** [04a-MemoryPoolingAlgorithm.md](04a-MemoryPoolingAlgorithm.md), [02b-OneShotBufferize.md](02b-OneShotBufferize.md), [MEMORY-MANAGEMENT.md](../../MEMORY-MANAGEMENT.md)

---

## Overview

`memory-pooling` enables spatial reuse of GPU memory. Without pooling, each `memref.alloc` (inserted by `one-shot-bufferize`) becomes a separate `hipMalloc()` call at runtime. With pooling, buffers with non-overlapping lifetimes share the same physical memory location.

**Input:** HIP dialect (memref mode) with `memref.alloc` ops produced by `one-shot-bufferize`
**Output:** HIP dialect with `memref.alloc` replaced by `memref.view` of a single pool

---

## Algorithm

Greedy best-fit strip packing. For algorithm detail see [04a-MemoryPoolingAlgorithm.md](04a-MemoryPoolingAlgorithm.md).

Steps:
1. Collect all `memref.alloc` ops with static shapes in each function
2. For each alloc, compute liveness end via transitive use-def traversal
3. Sort buffers largest-first; for each buffer find the smallest non-conflicting gap, or append at high-water mark
4. Alignment: 256 bytes (`kPoolAlignment`)
5. Replace each `memref.alloc` with:
   - `hip.get_pool(%ctx) : memref<?xi8, 1>` (emitted once per function)
   - `memref.view %pool[%offset][] : memref<..., 1>` at the assigned byte offset

---

## IR Transformation

### Before (memref.alloc from one-shot-bufferize)

```mlir
func.func @main_graph(%ctx: !hip.context,
                      %input: memref<1x3x224x224xf32>) {
  %buf0 = memref.alloc() : memref<1x64x224x224xf32, 1>
  hip.conv ins(%ctx, %input, %weights, %bias : ...)
           outs(%buf0 : memref<1x64x224x224xf32, 1>)

  %buf1 = memref.alloc() : memref<1x64x224x224xf32, 1>
  hip.relu ins(%ctx, %buf0 : ...)
           outs(%buf1 : memref<1x64x224x224xf32, 1>)
  return
}
```

### After (memref.view of pool)

```mlir
func.func @main_graph(%ctx: !hip.context,
                      %input: memref<1x3x224x224xf32>) {
  %pool = hip.get_pool(%ctx) : memref<?xi8, 1>

  %off0 = arith.constant 0 : index
  %buf0 = memref.view %pool[%off0][] : memref<1x64x224x224xf32, 1>
  hip.conv ins(%ctx, %input, %weights, %bias : ...)
           outs(%buf0 : memref<1x64x224x224xf32, 1>)

  %off1 = arith.constant 12845056 : index   // offset for buf1 (non-overlapping)
  %buf1 = memref.view %pool[%off1][] : memref<1x64x224x224xf32, 1>
  hip.relu ins(%ctx, %buf0 : ...)
           outs(%buf1 : memref<1x64x224x224xf32, 1>)
  return
}
```

Pool size is stored as a module attribute for diagnostics:

```mlir
module attributes {"hipdnn.pool_size" = 12845056 : i64} { ... }
```

---

## Runtime Integration

`hip.get_pool(%ctx)` lowers to `hipdnn_ep_get_pool_base(state)` in [05-HipToLLVM.md](05-HipToLLVM.md), which returns the base pointer of the pool allocated by `inference_init`.

`memref.view` is lowered by standard `finalize-memref-to-llvm` to pointer arithmetic.

Pool lifecycle:
- `inference_init`: `hipMalloc(pool_size)` — allocates the pool
- `inference_compute`: `get_pool_base(state)` → pointer arithmetic for each buffer
- `inference_cleanup`: single `hipFree(pool_base)` — frees entire pool

See [06-GenerateInterfacePass.md](06-GenerateInterfacePass.md) for pool initialization and cleanup generation.

---

## Comparison to Previous Design

| Aspect | Old design | New design |
|--------|-----------|-----------|
| Input ops | `hip.alloc` | `memref.alloc` (from one-shot-bufferize) |
| Liveness source | MLIR `Liveness` class | Transitive use-def traversal |
| Algorithm | Chaitin graph coloring | Greedy best-fit strip packing |
| Alignment | 4096 bytes | 256 bytes |
| Output mechanism | Module metadata side-channel | Direct IR replacement (`memref.view`) |
| Runtime query | `hipdnn_ep_get_buffer_from_pool(state, index)` | `hipdnn_ep_get_pool_base(state)` + offset constant |

---

## Limitations

- **Static shapes only**: Dynamic dimensions are skipped (pass leaves dynamic `memref.alloc` ops unchanged)
- **Single pool per function**: All static buffers in one contiguous allocation

---

## Related Documents

- [04a-MemoryPoolingAlgorithm.md](04a-MemoryPoolingAlgorithm.md) - Strip packing algorithm detail
- [02b-OneShotBufferize.md](02b-OneShotBufferize.md) - one-shot-bufferize (produces memref.alloc input)
- [05-HipToLLVM.md](05-HipToLLVM.md) - Lowers hip.get_pool and memref.view
- [MEMORY-MANAGEMENT.md](../../MEMORY-MANAGEMENT.md) - Memory allocation strategy
- [../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md) - Pipeline integration
