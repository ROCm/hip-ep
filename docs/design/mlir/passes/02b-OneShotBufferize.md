<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# One-Shot Bufferization Pass

**Date:** 2026-03-03
**Document Type:** Design
**Status:** Draft
**Related:** [02-OnnxToHip.md](02-OnnxToHip.md), [03-Canonicalization.md](03-Canonicalization.md), [04-MemoryPooling.md](04-MemoryPooling.md), [MEMORY-MANAGEMENT.md](../../MEMORY-MANAGEMENT.md)

**Prerequisite:** [02-OnnxToHip.md](02-OnnxToHip.md) — must run first to produce tensor-mode HIP ops
**Input:** HIP dialect module (tensor mode)
**Output:** HIP dialect module (memref mode)

---

## Overview

Stage 3 and 4 of the pipeline convert the tensor-mode HIP module produced by
`convert-onnx-to-hip` into memref mode. Two passes run back-to-back:

1. **`one-shot-bufferize`** — converts tensor DPS ops to memref ops, inserting
   `memref.alloc` with explicit ownership where a fresh buffer is required.
2. **`buffer-results-to-out-params`** — converts functions that return memrefs
   into functions that write into caller-provided output arguments.

This is the only place in the pipeline where `memref.alloc` is introduced.
The memory pooling pass (Stage 6) subsequently replaces those allocs with
`memref.view` of the pool.

---

## Configuration

```cpp
mlir::bufferization::OneShotBufferizationOptions opts;
opts.bufferizeFunctionBoundaries = true;
opts.defaultMemorySpaceFn = [](TensorType) -> std::optional<Attribute> {
  // Address space 1 = GPU memory
  return IntegerAttr::get(IntegerType::get(ctx, 64), 1);
};
pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
```

Key options:
- `bufferizeFunctionBoundaries = true` — bufferizes function signatures, not
  just op bodies. Required so `@main_graph` transitions from returning a tensor
  to accepting an output `memref` argument.
- `defaultMemorySpaceFn` → address space 1 — ensures all `memref.alloc` ops
  produced by bufferization target GPU memory (address space 1), matching the
  memory space used throughout the HIP dialect.

---

## Bufferization Models

HIP ops implement `BufferizableOpInterface` via external models registered at
dialect load time (`registerHipBufferizableOpInterfaceModels`):

| Model | Ops | Behavior |
|-------|-----|----------|
| `HipDstBufferizableModel` | All 16 compute ops | Allocates a fresh `memref.alloc` for each DPS output |
| `HipElementwiseBufferizableModel` | `hip.relu`, `hip.cast` | In-place: output aliases the input buffer when input has a single use |

### HipDstBufferizableModel

Standard DPS bufferization: for each tensor output, `one-shot-bufferize`
emits a `memref.alloc` and passes it as the `outs` operand.

### HipElementwiseBufferizableModel

Overrides `getAliasingValues` to return `BufferRelation::Equivalent` for the
output operand. When the input tensor has a single use, `one-shot-bufferize`
reuses the input buffer in-place and eliminates the `tensor.empty()` + `memref.alloc`
entirely. The op writes its result directly into the input buffer.

This is safe for element-wise ops where output shape == input shape and there
are no read-after-write hazards.

---

## Input / Output

**Input (tensor mode, from `convert-onnx-to-hip`):**
```mlir
func.func @main_graph(%ctx: !hip.context,
                      %input: tensor<1x3x224x224xf32>)
                      -> tensor<1x64x224x224xf32> {
  %init0 = tensor.empty() : tensor<1x64x224x224xf32>
  %0 = hip.conv ins(%ctx, %input, %weights, %bias : ...)
                outs(%init0 : tensor<1x64x224x224xf32>)
                -> tensor<1x64x224x224xf32>

  %init1 = tensor.empty() : tensor<1x64x224x224xf32>
  %1 = hip.relu ins(%ctx, %0 : ...)
                outs(%init1 : tensor<1x64x224x224xf32>)
                -> tensor<1x64x224x224xf32>
  return %1 : tensor<1x64x224x224xf32>
}
```

**Output (memref mode, after both passes):**
```mlir
func.func @main_graph(%ctx: !hip.context,
                      %input: memref<1x3x224x224xf32>,
                      %output: memref<1x64x224x224xf32>) {
  // HipDstBufferizableModel: fresh alloc for hip.conv output
  %buf = memref.alloc() : memref<1x64x224x224xf32, 1>
  hip.conv ins(%ctx, %input, %weights, %bias : ...)
           outs(%buf : memref<1x64x224x224xf32, 1>)

  // HipElementwiseBufferizableModel: relu writes into %buf in-place
  hip.relu ins(%ctx, %buf : ...)
           outs(%buf : memref<1x64x224x224xf32, 1>)

  memref.copy %buf, %output
}
```

`tensor.empty()` ops are removed; `memref.alloc` is emitted only for
`hip.conv` (not for `hip.relu` — in-place reuse eliminates it).
`buffer-results-to-out-params` adds `%output` and removes the `return`.

---

## Pipeline Position

```
convert-onnx-to-hip       → tensor.empty()   (Stage 2, no allocation)
one-shot-bufferize         → memref.alloc()   (Stage 3, concrete where needed)
buffer-results-to-out-params → out-param ABI  (Stage 4)
canonicalize               → dead alloc cleanup (Stage 5)
memory-pooling             → memref.view()    (Stage 6, replaces alloc with pool)
convert-hip-to-llvm        → llvm.call        (Stage 7)
```

---

## Related Documents

- **[02-OnnxToHip.md](02-OnnxToHip.md)** — produces the tensor-mode input
- **[03-Canonicalization.md](03-Canonicalization.md)** — cleans up dead `memref.alloc` after in-place bufferization
- **[04-MemoryPooling.md](04-MemoryPooling.md)** — replaces `memref.alloc` with pool views
- **[MEMORY-MANAGEMENT.md](../../MEMORY-MANAGEMENT.md)** — overall memory allocation strategy
- **[../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** — pipeline overview
- **[../../tech-notes/2026-03-03-tensor-empty-vs-memref-alloc.md](../../../tech-notes/2026-03-03-tensor-empty-vs-memref-alloc.md)** — why `tensor.empty()` instead of `memref.alloc()`
