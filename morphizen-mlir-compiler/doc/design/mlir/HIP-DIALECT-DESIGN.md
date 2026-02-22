<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP Dialect

**Date:** 2026-02-15
**Document Type:** Design
**Status:** Self-Reviewed
**Related:** [../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md), [passes/01-OnnxToHip.md](passes/01-OnnxToHip.md), [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md)

---

## Overview

The HIP dialect provides MLIR operations for AMD GPU execution via MIOpen (convolution, pooling, batch norm) and hipBLAS (matrix operations). It represents the intermediate stage between high-level ONNX operations and low-level LLVM IR.

**Pipeline position:** ONNX dialect → **HIP dialect** → LLVM dialect

**Design principle:** Destination-passing style with opaque context for GPU runtime state.

---

## Operations Summary

| Operation | Mnemonic | Description | Backend |
|-----------|----------|-------------|---------|
| [hip.alloc](#memory-management) | `alloc` | Allocate GPU memory | HIP Runtime |
| [hip.free](#memory-management) | `free` | Free GPU memory | HIP Runtime |
| [hip.copy](#memory-management) | `copy` | GPU-to-GPU memory copy | HIP Runtime |
| [hip.conv](#convolution-and-pooling-miopen) | `conv` | 2D convolution | MIOpen |
| [hip.maxpool](#convolution-and-pooling-miopen) | `maxpool` | Max pooling | MIOpen |
| [hip.avgpool](#convolution-and-pooling-miopen) | `avgpool` | Average pooling | MIOpen |
| [hip.gemm](#linear-algebra-hipblas) | `gemm` | Matrix multiplication (GEMM) | hipBLAS |
| [hip.relu](#activation-functions-miopen) | `relu` | ReLU activation | MIOpen |
| [hip.get_constant](#constant-access) | `get_constant` | Retrieve pre-uploaded constant | Runtime |

---

## Type System

### !hip.context

Opaque pointer to runtime execution state. See [../RUNTIME-ARCHITECTURE.md](../RUNTIME-ARCHITECTURE.md) for RuntimeState implementation.

**Properties:**
- Passed as first argument to all HIP operations
- Lowered to !llvm.ptr by [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md)
- Thread-safe (one context per inference session)
- Runtime extracts GPU stream, library handles, constant pointers internally

**Rationale:** Opaque type decouples IR from runtime implementation details (handle offsets, struct layout). Adding new library support (rocFFT, rocBLAS) requires no IR changes.

---

## Operation Categories

### Memory Management

**hip.alloc** - GPU memory allocation
```mlir
%buf = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
```
- Address space 1 = GPU memory
- Implements `AllocationOpInterface` (automatic deallocation via hip.free)
- Supports dynamic dimensions: `hip.alloc(%ctx, %N) : memref<?x128xf32, 1>`

**hip.free** - GPU memory deallocation
```mlir
hip.free(%ctx, %buf) : memref<1x64x224x224xf32, 1>
```

### Convolution and Pooling (MIOpen)

**hip.conv** - 2D convolution
```mlir
hip.conv(%ctx, %input, %weights, %bias, %output) {
  kernel_shape = [3, 3], strides = [1, 1],
  pads = [1, 1, 1, 1], dilations = [1, 1], group = 1
} : (!hip.context, memref<1x3x224x224xf32, 1>, memref<64x3x3x3xf32, 1>,
     memref<64xf32, 1>, memref<1x64x224x224xf32, 1>)
```
- Bias operand optional
- Lowered to `wrap_miopenConvolutionForward` runtime call

**hip.maxpool / hip.avgpool** - Pooling operations
```mlir
hip.maxpool(%ctx, %input, %output) {
  kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]
} : (!hip.context, memref<1x64x112x112xf32, 1>, memref<1x64x56x56xf32, 1>)
```

### Linear Algebra (hipBLAS)

**hip.gemm** - Matrix multiplication
```mlir
hip.gemm(%ctx, %A, %B, %C) {transA = 0, transB = 0, alpha = 1.0, beta = 0.0}
  : (!hip.context, memref<MxKxf32, 1>, memref<KxNxf32, 1>, memref<MxNxf32, 1>)
```
- Computes: C = alpha * A * B + beta * C
- Lowered to `wrap_hipblasSgemm` runtime call

### Activation Functions (MIOpen)

**hip.relu** - ReLU activation
```mlir
hip.relu(%ctx, %input, %output)
  : (!hip.context, memref<1x64x224x224xf32, 1>, memref<1x64x224x224xf32, 1>)
```

### Constant Access

**hip.get_constant** - Retrieve pre-uploaded constant
```mlir
%weights = hip.get_constant(%ctx, 0) : memref<64x3x3x3xf32, 1>
```
- Index refers to RuntimeState.gpu_constants array
- Constants uploaded during initialization
- See [../CONSTANT-HANDLING-DESIGN.md](../CONSTANT-HANDLING-DESIGN.md) for upload flow

---

## Destination-Passing Style

All operations modify pre-allocated output buffers instead of returning values:

```mlir
// Allocate output buffer
%output = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>

// Operation writes result to %output (no return value)
hip.conv(%ctx, %input, %weights, %bias, %output) {...}
  : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)
```

**Rationale:**
- Matches GPU library APIs (MIOpen, hipBLAS use output pointers)
- Enables buffer reuse optimizations (see [passes/04-MemoryPooling.md](passes/04-MemoryPooling.md))
- Clear data flow for alias analysis

---

## Lowering Strategy

HIP operations lower to calls to external runtime wrappers implemented in C++. See [passes/WHY-HIP-WRAPPERS.md](passes/WHY-HIP-WRAPPERS.md) for rationale.

**Example transformation:**
```mlir
// HIP dialect
hip.conv(%ctx, %input, %weights, %bias, %output) {...}

// LLVM dialect (after HipToLLVM)
llvm.call @wrap_miopenConvolutionForward(
  %ctx,                                    // opaque RuntimeState*
  %input_ptr, %input_n, %input_c, %h, %w, // input data + dimensions
  %weights_ptr, %K,                        // weights data + output channels
  %bias_ptr,                               // bias data (nullable)
  %output_ptr, %output_h, %output_w,       // output data + dimensions
  3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1          // conv attributes (kernel, stride, pad, dilation, group)
) : (!llvm.ptr, ...) -> i32
```

Runtime wrapper handles:
- Extracting MIOpen handle and GPU stream from opaque RuntimeState
- Creating MIOpen descriptors from dimension values
- Calling `miopenConvolutionForward` with 13+ parameters
- Destroying descriptors and returning status

See [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md) for complete lowering details.

---

## Dynamic Shape Support

Dimensions flow as runtime values through memref size arrays:

```mlir
// Rank known at compile time (4D tensor)
// Dimension values extracted at runtime from memref struct
%input : memref<?x?x?x?xf32, 1>

// In lowering: extract runtime dimensions
%n = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64
%c = llvm.extractvalue %input[3, 1] : !llvm.struct<...> -> i64
// ... pass to MIOpen descriptor creation
```

See [../DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) for interface integration.

---

## Related Documents

**Dialect Usage:**
- [passes/01-OnnxToHip.md](passes/01-OnnxToHip.md) - Generates HIP operations from ONNX
- [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md) - Lowers HIP to LLVM dialect
- [passes/04-MemoryPooling.md](passes/04-MemoryPooling.md) - Optimizes hip.alloc calls

**Runtime Integration:**
- [../RUNTIME-ARCHITECTURE.md](../RUNTIME-ARCHITECTURE.md) - RuntimeState implementation
- [../CONSTANT-HANDLING-DESIGN.md](../CONSTANT-HANDLING-DESIGN.md) - Constant upload flow
- [passes/WHY-HIP-WRAPPERS.md](passes/WHY-HIP-WRAPPERS.md) - Wrapper function rationale

**Pipeline Context:**
- [../MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md) - Full compilation pipeline
- [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - Complete transformation examples
