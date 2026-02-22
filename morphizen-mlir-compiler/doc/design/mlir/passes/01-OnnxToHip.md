<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OnnxToHip Pass

**Date:** 2026-02-20
**Document Type:** Design
**Status:** Draft
**Related:** [05-HipToLLVM.md](05-HipToLLVM.md), [CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md), [INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md)

**Input:** ONNX-MLIR module
**Output:** HIP dialect module

---

## Table of Contents

- [Overview](#overview)
- [Input Format](#input-format)
- [Output Format (HIP Dialect)](#output-format-hip-dialect)
- [Key Transformations](#key-transformations)
  - [1. Add Module Metadata Attributes](#1-add-module-metadata-attributes)
  - [2. Extract Constants to Globals](#2-extract-constants-to-globals)
  - [3. Change @main_graph Signature](#3-change-main-signature)
  - [4. Convert Operations](#4-convert-operations)
  - [5. Generate Constant Registry](#5-generate-constant-registry)
- [Destination-Passing Style (DPS) Optimization](#destination-passing-style-dps-optimization)
- [Prerequisites Met](#prerequisites-met)
- [Related Documents](#related-documents)

---

## Overview

The OnnxToHip pass transforms ONNX operations into HIP dialect operations. This is the first major lowering step in the compilation pipeline.

**Key transformations:**
1. Extract constants to `llvm.mlir.global`
2. Convert ONNX operations to HIP operations
3. Change @main_graph signature (add context, output arg, i32 return)
4. Generate constant registry (ConstantInfo array + accessor function)
5. Add module I/O metadata attributes

---

## Input Format

```mlir
module {
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32> {
    %weights = "onnx.Constant"() {value = dense<[...]> : tensor<64x3x3x3xf32>}
      : () -> tensor<64x3x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<[...]> : tensor<64xf32>}
      : () -> tensor<64xf32>

    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1
    } : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
        -> tensor<1x64x224x224xf32>

    %1 = "onnx.Relu"(%0) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

    return %1 : tensor<1x64x224x224xf32>
  }
}
```

**Characteristics:**
- High-level ONNX operations
- Tensor types
- Constants as `onnx.Constant` operations
- Return value semantics

---

## Output Format (HIP Dialect)

```mlir
// Module with I/O metadata
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
} {
  // Constants extracted to globals
  llvm.mlir.global constant @constant_0(
    dense<[[[[1.0, ...]]]]> : tensor<64x3x3x3xf32>
  ) : !llvm.array<1728 x f32>

  llvm.mlir.global constant @constant_1(
    dense<[0.5, ...]> : tensor<64xf32>
  ) : !llvm.array<64 x f32>

  // Main function - signature changed
  func.func @main_graph(%ctx: !hip.context,
                   %input: memref<1x3x224x224xf32>,
                   %output: memref<1x64x224x224xf32>) -> i32 {

    // Retrieve constants from GPU state
    %weights = hip.get_constant(%ctx, 0) : (!hip.context, i64) -> memref<64x3x3x3xf32, 1>
    %bias = hip.get_constant(%ctx, 1) : (!hip.context, i64) -> memref<64xf32, 1>

    // ONNX operations → HIP operations (inline, in-place)
    %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    hip.conv(%ctx, %input, %weights, %bias, %temp) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1
    } : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)

    // Destination-passing: copy result to output argument
    memref.copy %temp, %output : memref<1x64x224x224xf32, 1> to memref<1x64x224x224xf32>

    // Return success status
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }

  // Constant registry - metadata for runtime
  llvm.mlir.global constant @constant_info_array() : !llvm.array<2 x !llvm.struct<(ptr, i64, i64, i64)>> {
    // ConstantInfo for constant_0: cpu_data=@constant_0, size_bytes=6912, elem_size=4, num_elems=1728
    // ConstantInfo for constant_1: cpu_data=@constant_1, size_bytes=256, elem_size=4, num_elems=64
  }

  llvm.mlir.global constant @constant_registry() : !llvm.struct<(ptr, i64)> {
    // constants=@constant_info_array, count=2
  }

  llvm.func @get_constant_registry() -> !llvm.ptr {
    %registry_ptr = llvm.mlir.addressof @constant_registry : !llvm.ptr
    llvm.return %registry_ptr : !llvm.ptr
  }
}
```

---

## Key Transformations

### 1. Add Module Metadata Attributes

**Critical for GenerateInterfacePass**

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,    // Input 0 is rank 4
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>    // Output 0 is rank 2
}
```

**Why?** GenerateInterfacePass needs to know:
- How many inputs/outputs to expect
- What rank each tensor has
- This drives validation and memref struct generation

**How to compute:**
- Count arguments to original @main
- Inspect tensor types to get ranks
- Store as module attributes

### 2. Extract Constants to Globals

**Before:**
```mlir
%weights = "onnx.Constant"() {value = dense<[...]> : tensor<64x3x3x3xf32>}
  : () -> tensor<64x3x3x3xf32>
```

**After:**
```mlir
llvm.mlir.global constant @constant_0(dense<[...]> : tensor<64x3x3x3xf32>)
  : !llvm.array<1728 x f32>
```

**Why?**
- Constants embedded in DLL `.data` section
- Avoid runtime parsing overhead
- Enable efficient CPU → GPU transfer

### 3. Change @main_graph Signature

**Before:**
```mlir
func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32>
```

**After:**
```mlir
func.func @main_graph(%ctx: !hip.context,
                %input: memref<1x3x224x224xf32>,
                %output: memref<1x64x224x224xf32>) -> i32
```

**Changes:**
1. **Add context parameter** - `%ctx: !hip.context` (first parameter)
2. **tensor → memref** - `tensor<...>` → `memref<...>`
3. **Add output argument** - Destination-passing style
4. **Return i32** - Status code instead of tensor value

### 4. Convert Operations

**Pattern-based conversion:**

| ONNX Operation | HIP Operation |
|----------------|---------------|
| `onnx.Conv` | `hip.conv` |
| `onnx.Relu` | `hip.relu` |
| `onnx.MaxPool` | `hip.maxpool` |
| `onnx.Gemm` | `hip.gemm` |
| `onnx.BatchNormalization` | `hip.batchnorm` |
| `onnx.Constant` | (extracted to global) |

**Example conversion:**
```mlir
// Before (ONNX)
%0 = "onnx.Conv"(%input, %weights, %bias) {kernel_shape = [3, 3], ...}
  : (tensor<...>, tensor<...>, tensor<...>) -> tensor<...>

// After (HIP)
%temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
hip.conv(%ctx, %input, %weights, %bias, %temp) {kernel_shape = [3, 3], ...}
  : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)
```

**Key differences:**
- HIP ops take context as first parameter
- HIP ops use in-place semantics (output as last argument)
- HIP ops don't return values (modify output in-place)

### 5. Generate Constant Registry

**Generated structures and function:**

1. **ConstantInfo array** (`@constant_info_array`)
   - Static array of metadata for each constant
   - Fields: `cpu_data` (pointer), `size_bytes`, `element_size`, `num_elements`

2. **ConstantRegistry struct** (`@constant_registry`)
   - Wraps array with count
   - Fields: `constants` (pointer to array), `count`

3. **get_constant_registry()** function
   - Returns pointer to ConstantRegistry
   - Runtime uses this to get metadata for uploading/freeing constants

**Ownership:**
- DLL owns: Constant data, metadata structures (static lifetime)
- Runtime owns: GPU memory allocation, upload strategy, cleanup strategy

See [../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md) for details.

---

## Destination-Passing Style (DPS) Optimization

OnnxToHip eliminates copies by detecting when operations can write directly to output arguments instead of temporary buffers.

### Pattern Detection

**Target pattern:**
```mlir
%temp = hip.alloc(%ctx) : memref<...>
hip.conv(%ctx, %input, %weights, %bias, %temp) {...}
memref.copy %temp, %output
return %output
```

**Optimized output:**
```mlir
hip.conv(%ctx, %input, %weights, %bias, %output) {...}
return %output
```

### Algorithm

1. Check if return value comes from `hip.alloc` (temporary allocation)
2. Find DPS writer operation (implements `DestinationStyleOpInterface`)
3. Verify operation writes to temporary buffer
4. Redirect DPS operation to write to output argument
5. Erase temporary allocation and copy

### Supported Operations

All HIP operations implementing `DestinationStyleOpInterface`:
- `hip.conv`
- `hip.relu`
- `hip.gemm`
- `hip.max_pool`
- `hip.avg_pool`

### Fallback: Copy Generation

When DPS optimization cannot apply (multi-use buffers, unsupported ops):

**Replace `memref.copy` (CPU) with `hip.copy` (GPU-aware):**
```mlir
// Before
memref.copy %src, %dst : memref<...> to memref<...>

// After
hip.copy(%ctx, %src, %dst) : (!hip.context, memref<...>, memref<...>)
```

Ensures GPU address space correctness. Later optimized by [03-Canonicalization.md](03-Canonicalization.md).

### Results

**Typical case:** Zero-copy semantics (most operations)
**Multi-use case:** `hip.copy` generated (cleaned up by later passes)

---

## Prerequisites Met

This pass satisfies prerequisites for [06-GenerateInterfacePass.md](06-GenerateInterfacePass.md):

✅ **Prerequisite 2:** Generates get_constant_registry() function returning constant metadata - see [06-GenerateInterfacePass.md - Prerequisite 2](06-GenerateInterfacePass.md#prerequisite-2-get_constant_registry-function)
✅ **Prerequisite 3:** Adds module metadata (hipdnn.input_count, hipdnn.input_ranks, etc.) - see [06-GenerateInterfacePass.md - Prerequisite 3](06-GenerateInterfacePass.md#prerequisite-3-module-metadata-attributes)
✅ Generates @main_graph with signature: `(context, input, output) -> i32`
✅ Uses memref types (ready for struct-by-value in later passes)

For complete interface design, see [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md).

---

## Related Documents

- **[02-BufferDeallocation.md](02-BufferDeallocation.md)** - Next pass (buffer deallocation)
- **[03-Canonicalization.md](03-Canonicalization.md)** - Copy elimination patterns (cleanup fallback cases)
- **[05-HipToLLVM.md](05-HipToLLVM.md)** - Later pass (HIP to LLVM lowering)
- **[06-GenerateInterfacePass.md](06-GenerateInterfacePass.md)** - Final pass (C interface generation)
- **[../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md)** - Constant extraction strategy
- **[../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md)** - HIP operation definitions
- **[../HIP-COPY-DESIGN.md](../HIP-COPY-DESIGN.md)** - Copy optimization strategy
- **[../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md)** - Prerequisites this pass must satisfy
- **[../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** - Pipeline overview
