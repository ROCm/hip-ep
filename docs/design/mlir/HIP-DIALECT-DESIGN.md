<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP Dialect

**Date:** 2026-03-02
**Document Type:** Design
**Status:** Draft
**Related:** [../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md), [passes/02-OnnxToHip.md](passes/02-OnnxToHip.md), [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md)

---

## Overview

The HIP dialect provides MLIR operations for AMD GPU execution via MIOpen (convolution, pooling, normalization) and hipBLAS (matrix operations). It represents the intermediate stage between high-level ONNX operations and low-level LLVM IR.

**Pipeline position:** ONNX dialect → **HIP dialect** → LLVM dialect

**Design principle:** Destination-passing style with opaque context for GPU runtime state. Operations are defined with `ins`/`outs` operand lists using `Hip_TensorOrMemRef` types so the same op definition works in both tensor mode (before bufferization) and memref mode (after bufferization).

---

## Operations Summary

| Operation | Mnemonic | Description | Backend |
|-----------|----------|-------------|---------|
| [hip.get_pool](#memory-pool) | `get_pool` | Get raw GPU pool buffer | Runtime |
| [hip.get_constant](#constant-access) | `get_constant` | Retrieve pre-uploaded constant | Runtime |
| [hip.conv](#convolution-and-pooling-miopen) | `conv` | 2D convolution | MIOpen |
| [hip.maxpool](#convolution-and-pooling-miopen) | `maxpool` | Max pooling | MIOpen |
| [hip.avgpool](#convolution-and-pooling-miopen) | `avgpool` | Average pooling | MIOpen |
| [hip.gemm](#linear-algebra-hipblas) | `gemm` | Matrix multiplication (GEMM) | hipBLAS |
| [hip.matmul](#linear-algebra-hipblas) | `matmul` | Matrix multiplication | hipBLAS |
| [hip.relu](#activation-and-elementwise) | `relu` | ReLU activation | MIOpen |
| [hip.sigmoid](#activation-and-elementwise) | `sigmoid` | Sigmoid activation | MIOpen |
| [hip.cast](#activation-and-elementwise) | `cast` | Type cast | Runtime |
| [hip.mul](#activation-and-elementwise) | `mul` | Element-wise multiply | Runtime |
| [hip.sub](#activation-and-elementwise) | `sub` | Element-wise subtract | Runtime |
| [hip.gather](#data-operations) | `gather` | Gather indices | Runtime |
| [hip.reduce_sum](#data-operations) | `reduce_sum` | Reduction sum | Runtime |
| [hip.rotary_embedding](#transformer-ops) | `rotary_embedding` | Rotary position embedding | Runtime |
| [hip.simplified_layer_norm](#transformer-ops) | `simplified_layer_norm` | Simplified layer norm | Runtime |
| [hip.skip_simplified_layer_norm](#transformer-ops) | `skip_simplified_layer_norm` | Skip + layer norm | Runtime |
| [hip.group_query_attention](#transformer-ops) | `group_query_attention` | Group query attention | Runtime |

---

## Type System

### !hip.context

Opaque pointer to runtime execution state. See [../RUNTIME-ARCHITECTURE.md](../RUNTIME-ARCHITECTURE.md) for `RuntimeState` implementation.

**Properties:**
- Passed as first argument to all HIP operations (prepended by `hip-add-context-arg`)
- Lowered to `!llvm.ptr` by [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md)
- Thread-safe (one context per inference session)
- Runtime extracts GPU stream, library handles, constant pointers internally

### Hip_TensorOrMemRef

Type alias covering both `AnyRankedTensor` and `AnyMemRef`. Used for compute op data operands so the same TableGen op definition works in tensor mode (after `convert-onnx-to-hip`) and memref mode (after `one-shot-bufferize`).

---

## Operation Categories

### Memory Pool

**hip.get_pool** — Get raw GPU pool buffer
```mlir
%pool = hip.get_pool(%ctx) : memref<?xi8, 1>
```
- Emitted once per function by `memory-pooling` pass
- Returns a 1-D raw byte buffer in GPU address space (address space 1)
- Lowered to `hipdnn_ep_get_pool_base(state)` runtime call
- `memref.view` ops slice it at static byte offsets for each intermediate buffer

### Constant Access

**hip.get_constant** — Retrieve pre-uploaded constant
```mlir
%weights = hip.get_constant(%ctx, 0) : memref<64x3x3x3xf32, 1>
```
- Index refers to `RuntimeState.gpu_constants` array
- Constants uploaded during `inference_init`
- See [../CONSTANT-HANDLING-DESIGN.md](../CONSTANT-HANDLING-DESIGN.md) for upload flow

### Convolution and Pooling (MIOpen)

**hip.conv** — 2D convolution
```mlir
%0 = hip.conv ins(%ctx, %input, %weights, %bias :
                  !hip.context, tensor<1x3x224x224xf32>,
                  memref<64x3x3x3xf32, 1>, memref<64xf32, 1>)
              outs(%init : tensor<1x64x224x224xf32>) {
  kernel_shape = [3, 3], strides = [1, 1],
  pads = [1, 1, 1, 1], dilations = [1, 1], group = 1
} -> tensor<1x64x224x224xf32>
```
- Bias operand optional
- Lowered to `wrap_miopenConvolutionForward` runtime call

**hip.maxpool / hip.avgpool** — Pooling operations
```mlir
%0 = hip.maxpool ins(%ctx, %input : !hip.context, tensor<1x64x224x224xf32>)
                 outs(%init : tensor<1x64x112x112xf32>) {
  kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]
} -> tensor<1x64x112x112xf32>
```

### Linear Algebra (hipBLAS)

**hip.gemm** — General matrix multiply
```mlir
%0 = hip.gemm ins(%ctx, %A, %B : !hip.context,
                   tensor<MxKxf32>, tensor<KxNxf32>)
              outs(%C : tensor<MxNxf32>) {
  transA = 0, transB = 0, alpha = 1.0, beta = 0.0
} -> tensor<MxNxf32>
```
- Computes: C = alpha × A × B + beta × C

### Activation and Elementwise

**hip.relu, hip.sigmoid, hip.cast, hip.mul, hip.sub** — Element-wise operations
```mlir
%0 = hip.relu ins(%ctx, %input : !hip.context, tensor<1x64x224x224xf32>)
              outs(%init : tensor<1x64x224x224xf32>)
              -> tensor<1x64x224x224xf32>
```

`hip.relu` and `hip.cast` support in-place bufferization via `HipElementwiseBufferizableModel`. See [passes/02b-OneShotBufferize.md](passes/02b-OneShotBufferize.md).

### Data Operations

**hip.gather, hip.reduce_sum** — Data manipulation
```mlir
%0 = hip.gather ins(%ctx, %data, %indices : ...)
                outs(%init : tensor<...>) {...} -> tensor<...>
```

### Transformer Ops

**hip.rotary_embedding, hip.simplified_layer_norm, hip.skip_simplified_layer_norm, hip.group_query_attention**

Large-kernel transformer operations lowered to runtime wrapper functions.

---

## Ins/Outs Format

All compute ops use TableGen `ins`/`outs` format with `DestinationStyleOpInterface`:

```tablegen
def Hip_ConvOp : Hip_Op<"conv", [
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$weights,
    Optional<Hip_TensorOrMemRef>:$bias,
    Hip_TensorOrMemRef:$output,
    // ... attributes
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
}
```

`getDpsInitsMutable()` returns the mutable range of the `output` (and `result` for gemm/matmul) operand.

---

## Lowering Strategy

HIP operations lower to calls to external C++ runtime wrappers. See [passes/WHY-HIP-WRAPPERS.md](passes/WHY-HIP-WRAPPERS.md) for rationale.

**Example transformation:**
```mlir
// HIP dialect (memref mode, after bufferization)
hip.conv ins(%ctx, %input, %weights, %bias : ...)
         outs(%output : memref<...>) {...}

// LLVM dialect (after convert-hip-to-llvm)
llvm.call @wrap_miopenConvolutionForward(
  %ctx,
  %input_ptr, %N, %C, %H, %W,
  %weights_ptr, %K,
  %bias_ptr,
  %output_ptr, %H_out, %W_out,
  3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1
) : (!llvm.ptr, ...) -> i32
```

See [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md) for complete lowering details.

---

## Related Documents

**Dialect Usage:**
- [passes/02-OnnxToHip.md](passes/02-OnnxToHip.md) - Generates HIP operations from ONNX
- [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md) - Lowers HIP to LLVM dialect
- [passes/04-MemoryPooling.md](passes/04-MemoryPooling.md) - Emits hip.get_pool, replaces memref.alloc

**Runtime Integration:**
- [../RUNTIME-ARCHITECTURE.md](../RUNTIME-ARCHITECTURE.md) - RuntimeState implementation
- [../CONSTANT-HANDLING-DESIGN.md](../CONSTANT-HANDLING-DESIGN.md) - Constant upload flow
- [passes/WHY-HIP-WRAPPERS.md](passes/WHY-HIP-WRAPPERS.md) - Wrapper function rationale

**Pipeline Context:**
- [../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md) - Full compilation pipeline
- [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - Complete transformation stages
- [passes/02b-OneShotBufferize.md](passes/02b-OneShotBufferize.md) - Bufferization models for HIP ops
