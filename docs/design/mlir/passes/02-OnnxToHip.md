<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OnnxToHip Pass

**Date:** 2026-03-02
**Document Type:** Design
**Status:** Draft
**Related:** [01-HipAddContextArg.md](01-HipAddContextArg.md), [05-HipToLLVM.md](05-HipToLLVM.md), [CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md), [INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md), [02b-OneShotBufferize.md](02b-OneShotBufferize.md)

**Prerequisite:** [01-HipAddContextArg.md](01-HipAddContextArg.md) — must run first to prepend `%ctx`
**Input:** ONNX-MLIR module (after [`hip-add-context-arg`](01-HipAddContextArg.md))
**Output:** HIP dialect module (tensor mode)

---

## Overview

`convert-onnx-to-hip` performs three transformations:

1. **[Constant extraction](../../CONSTANT-HANDLING-DESIGN.md)** — Writes all
   constant data to the configured constants file (default: `constants.bin`,
   see [CompilationOptions](../../COMPILATION-OPTIONS.md))
   via `DiskFileSystem`.

2. **Op conversion** — Converts ONNX operations to HIP operations in tensor
   mode. Replaces `onnx.Constant` ops with `hip.get_constant` calls indexed by
   position.

3. **Module annotation** — Attaches metadata attributes to the module for use
   by downstream passes ([GenerateInterfacePass](06-GenerateInterfacePass.md),
   [HipToLLVM](05-HipToLLVM.md)):
   - `hipdnn.constant_sizes` — byte size of each constant (indexed by position)
   - `hipdnn.constant_offsets` — byte offset of each constant in the file
   - `hipdnn.input_count`, `hipdnn.input_shapes`,
     `hipdnn.input_element_sizes` — model input metadata
   - `hipdnn.output_count`, `hipdnn.output_shapes`,
     `hipdnn.output_element_sizes` — model output metadata

Bufferization (tensor→memref) happens downstream in
[one-shot-bufferize](02b-OneShotBufferize.md), not here.

---

## Input Format

```mlir
module {
  // %ctx prepended by hip-add-context-arg (prerequisite)
  func.func @main_graph(%ctx: !hip.context,
                        %arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32> {
    %weights = "onnx.Constant"() {value = dense<[...]> : tensor<64x3x3x3xf32>}
      : () -> tensor<64x3x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<[...]> : tensor<64xf32>}
      : () -> tensor<64xf32>

    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      kernel_shape = [3, 3], strides = [1, 1],
      pads = [1, 1, 1, 1], dilations = [1, 1], group = 1
    } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
        -> tensor<1x64x224x224xf32>

    %1 = "onnx.Relu"(%0) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

    onnx.Return %1 : tensor<1x64x224x224xf32>
  }
}
```

---

## Key Transformations

### 1. Add Module Metadata Attributes

Computed from the original `@main_graph` argument types and emitted before any
op conversion.

```mlir
module attributes {
  // Constant metadata (see CONSTANT-HANDLING-DESIGN.md)
  hipdnn.constant_sizes = array<i64: 6912, 256>,      // → GenerateInterface
  hipdnn.constant_offsets = array<i64: 0, 6912>,      // → GenerateInterface

  // Input tensor metadata
  hipdnn.input_count = 1 : i64,                       // → HipToLLVM, GenerateInterface
  hipdnn.input_shapes = [dense<[1,3,224,224]> : tensor<4xi64>],  // → GenerateInterface
  hipdnn.input_element_sizes = array<i64: 4>,         // → GenerateInterface

  // Output tensor metadata
  hipdnn.output_count = 1 : i64,                      // → HipToLLVM, GenerateInterface
  hipdnn.output_shapes = [dense<[1,64,224,224]> : tensor<4xi64>], // → GenerateInterface
  hipdnn.output_element_sizes = array<i64: 4>         // → GenerateInterface
}
```

### 2. Extract Constants to File

Processes all `onnx.Constant` ops in walk order:

1. **Index** — assign each constant a unique index 0, 1, 2, …
2. **Layout** — compute byte offset for each constant using the bin-packing strategy described in [CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md) (`kConstantAlignment = 256`)
3. **Write** — write raw bytes of each constant to the [constants file](../../CONSTANT-HANDLING-DESIGN.md) at its computed offset; zero-fill alignment gaps
4. **Annotate** — emit `hipdnn.constant_sizes` and `hipdnn.constant_offsets` on the module
5. **Replace** — remove the `onnx.Constant` op; replace each use with `hip.get_constant(%ctx, index)`

**Before:**
```mlir
%weights = "onnx.Constant"() {value = dense<[...]> : tensor<64x3x3x3xf32>}
  : () -> tensor<64x3x3x3xf32>
```

**After:** *(op removed — 6912 bytes written to constants.bin at offset 0)*
```mlir
%weights = hip.get_constant(%ctx, 0) : memref<64x3x3x3xf32, 1>
```

### 3. Convert Operations (tensor mode)

Patterns are `OpRewritePattern` (not conversion patterns). Each op:
1. Allocates a [`tensor.empty()`](../../../tech-notes/2026-03-03-tensor-empty-vs-memref-alloc.md) destination
2. Emits the HIP op with `ins(...) outs(tensor.empty())` format
3. Returns the tensor result

| ONNX Operation | HIP Operation |
|----------------|---------------|
| `onnx.Conv` | `hip.conv` |
| `onnx.Relu` | `hip.relu` |
| `onnx.MaxPool` | `hip.maxpool` |
| `onnx.AveragePool` | `hip.avgpool` |
| `onnx.Gemm` | `hip.gemm` |
| `onnx.MatMul` | `hip.matmul` |
| `onnx.Sigmoid` | `hip.sigmoid` |
| `onnx.Mul` | `hip.mul` |
| `onnx.Sub` | `hip.sub` |
| `onnx.Gather` | `hip.gather` |
| `onnx.ReduceSum` | `hip.reduce_sum` |
| `onnx.Cast` | `hip.cast` |
| `onnx.RotaryEmbedding` | `hip.rotary_embedding` |
| `onnx.SimplifiedLayerNorm` | `hip.simplified_layer_norm` |
| `onnx.SkipSimplifiedLayerNorm` | `hip.skip_simplified_layer_norm` |
| `onnx.GroupQueryAttention` | `hip.group_query_attention` |
| `onnx.Constant` | `hip.get_constant` (data extracted to file, see §2) |

**Example:**
```mlir
// Before (ONNX)
%0 = "onnx.Conv"(%input, %weights, %bias) {kernel_shape = [3, 3], ...}
  : (tensor<...>, tensor<...>, tensor<...>) -> tensor<...>

// After (HIP, tensor mode)
%init = tensor.empty() : tensor<1x64x224x224xf32>
%0 = hip.conv ins(%ctx, %input, %weights, %bias : ...)
              outs(%init : tensor<1x64x224x224xf32>) {...}
              -> tensor<1x64x224x224xf32>
```

No `hip.alloc`, no `hip.copy`, no `-> i32` return. Bufferization happens downstream in [one-shot-bufferize](02b-OneShotBufferize.md).


---

## Related Documents

- **[02b-OneShotBufferize.md](02b-OneShotBufferize.md)** - one-shot-bufferize (converts tensor ops to memref)
- **[03-Canonicalization.md](03-Canonicalization.md)** - Cleanup after bufferization
- **[04-MemoryPooling.md](04-MemoryPooling.md)** - Pool assignment after bufferization
- **[05-HipToLLVM.md](05-HipToLLVM.md)** - Final lowering pass
- **[06-GenerateInterfacePass.md](06-GenerateInterfacePass.md)** - C interface generation
- **[../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md)** - Constant extraction strategy
- **[../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md)** - HIP operation definitions
- **[../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md)** - Interface prerequisites this pass satisfies for GenerateInterfacePass
- **[../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** - Pipeline overview

---

## Appendix: Output IR Example

Full HIP dialect module produced from the conv+relu example in [Input Format](#input-format).

```mlir
module attributes {
  // Constant metadata — data written to constants.bin by this pass
  hipdnn.constant_sizes = array<i64: 6912, 256>,    // weights=64*3*3*3*4, bias=64*4
  hipdnn.constant_offsets = array<i64: 0, 6912>,
  // I/O metadata consumed by HipToLLVM and GenerateInterfacePass
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [dense<[1, 3, 224, 224]> : tensor<4xi64>],
  hipdnn.input_element_sizes = array<i64: 4>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [dense<[1, 64, 224, 224]> : tensor<4xi64>],
  hipdnn.output_element_sizes = array<i64: 4>
} {
  // @main_graph in tensor mode — bufferization happens downstream
  func.func @main_graph(%ctx: !hip.context,
                        %arg0: tensor<1x3x224x224xf32>)
                        -> tensor<1x64x224x224xf32> {

    // Constants retrieved from GPU memory (uploaded by inference_init from constants.bin)
    %weights = hip.get_constant(%ctx, 0) : memref<64x3x3x3xf32, 1>
    %bias    = hip.get_constant(%ctx, 1) : memref<64xf32, 1>

    // ONNX ops → HIP ops (tensor mode, DPS with tensor.empty init)
    %init0 = tensor.empty() : tensor<1x64x224x224xf32>
    %0 = hip.conv ins(%ctx, %arg0, %weights, %bias : !hip.context,
                      tensor<1x3x224x224xf32>, memref<64x3x3x3xf32, 1>,
                      memref<64xf32, 1>)
                  outs(%init0 : tensor<1x64x224x224xf32>) {
      kernel_shape = [3, 3], strides = [1, 1],
      pads = [1, 1, 1, 1], dilations = [1, 1], group = 1
    } -> tensor<1x64x224x224xf32>

    %init1 = tensor.empty() : tensor<1x64x224x224xf32>
    %1 = hip.relu ins(%ctx, %0 : !hip.context, tensor<1x64x224x224xf32>)
                  outs(%init1 : tensor<1x64x224x224xf32>)
                  -> tensor<1x64x224x224xf32>

    return %1 : tensor<1x64x224x224xf32>
  }
}
```
