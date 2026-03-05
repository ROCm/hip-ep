<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# tensor.empty() vs memref.alloc()

**Date:** 2026-03-03
**Document Type:** Tech Note
**Status:** Draft
**Related:** [02b-OneShotBufferize.md](../design/mlir/passes/02b-OneShotBufferize.md), [02-OnnxToHip.md](../design/mlir/passes/02-OnnxToHip.md)

## Summary

`tensor.empty()` and `memref.alloc()` both represent "I need a buffer here",
but they live at different abstraction levels and have very different
optimization properties.

## tensor.empty()

```mlir
%init = tensor.empty() : tensor<1x64x224x224xf32>
```

- Operates in **tensor semantics** — no concrete memory, no address.
- A placeholder that tells one-shot-bufferize: "a result buffer of this shape
  is needed here."
- **May not allocate at all.** If the result is consumed by an op that can
  write in-place (e.g. `hip.relu`, `hip.cast`), the bufferizer eliminates the
  allocation entirely and reuses the input buffer.
- When allocation is necessary, one-shot-bufferize emits `memref.alloc()` at
  the optimal point — which the memory pooling pass then replaces with a
  `memref.view` into the pool.

## memref.alloc()

```mlir
%buf = memref.alloc() : memref<1x64x224x224xf32, 1>
```

- Operates in **memref semantics** — concrete memory with a fixed address.
- Allocation site is fixed in the IR; the optimizer cannot move or eliminate it
  without explicit analysis.
- Requires a paired deallocation or a pool-based replacement.
- Produced by one-shot-bufferize from `tensor.empty()` when in-place reuse is
  not possible.

## Why OnnxToHip uses tensor.empty()

`convert-onnx-to-hip` runs in **tensor mode**: all op results are tensors, and
`tensor.empty()` is used for DPS output slots. This defers all allocation
decisions to one-shot-bufferize (Stage 2 of the pipeline), which:

1. **Eliminates allocations** for element-wise ops (`hip.relu`, `hip.cast`)
   via in-place aliasing — the output reuses the input buffer.
2. **Emits `memref.alloc()`** only where a fresh buffer is genuinely needed.
3. The memory pooling pass then **replaces `memref.alloc()`** with
   `memref.view` into the pool returned by `hip.get_pool(%ctx)`.

If `convert-onnx-to-hip` emitted `memref.alloc()` directly, none of steps 1–3
would be possible: allocations would be fixed in place, in-place reuse would
require explicit alias analysis after the fact, and pool integration would be
more complex.

## Pipeline view

```
convert-onnx-to-hip   → tensor.empty()  (placeholder, no allocation)
one-shot-bufferize    → memref.alloc()  (concrete, where needed)
memory-pooling        → memref.view()   (into hip.get_pool(%ctx) blob)
convert-hip-to-llvm   → llvm.call       (hipdnn_ep_get_pool_base)
```
