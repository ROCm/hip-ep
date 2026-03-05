<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HipToLLVM Pass

**Date:** 2026-03-02
**Document Type:** Design
**Status:** Draft
**Related:** [02-OnnxToHip.md](02-OnnxToHip.md), [04-MemoryPooling.md](04-MemoryPooling.md), [06-GenerateInterfacePass.md](06-GenerateInterfacePass.md)

**Input:** HIP dialect module (memref mode, after memory-pooling)
**Output:** LLVM dialect module with external runtime function calls

---

## Overview

The HipToLLVM pass lowers HIP dialect operations to LLVM dialect, calling external C++ runtime functions that encapsulate MIOpen/hipBLAS complexity.

**Key transformations:**
1. Lower `!hip.context` → `!llvm.ptr`
2. Lower `hip.get_pool` → call to `hipdnn_ep_get_pool_base`
3. Lower `hip.get_constant` → call to `hipdnn_ep_get_constant`
4. Convert HIP compute ops to calls to external runtime wrappers
5. Transform `@main_graph` to 3-parameter array-based interface

`memref.view` (emitted by memory-pooling) is lowered by the standard `finalize-memref-to-llvm` pass, not by this pass.

---

## Transformations

### Input Format (HIP Dialect, memref mode)

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [dense<[1, 3, 224, 224]> : tensor<4xi64>],
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [dense<[1, 64, 224, 224]> : tensor<4xi64>],
  "hipdnn.pool_size" = 12845056 : i64
} {
  func.func @main_graph(%ctx: !hip.context,
                        %input: memref<1x3x224x224xf32>,
                        %output: memref<1x64x224x224xf32>) {
    %pool = hip.get_pool(%ctx) : memref<?xi8, 1>

    %off0 = arith.constant 0 : index
    %buf0 = memref.view %pool[%off0][] : memref<1x64x224x224xf32, 1>

    %weights = hip.get_constant(%ctx, 0) : memref<64x3x3x3xf32, 1>
    %bias    = hip.get_constant(%ctx, 1) : memref<64xf32, 1>

    hip.conv ins(%ctx, %input, %weights, %bias : ...)
             outs(%buf0 : memref<1x64x224x224xf32, 1>) {
      kernel_shape = [3, 3], strides = [1, 1],
      pads = [1, 1, 1, 1], dilations = [1, 1], group = 1
    }

    memref.copy %buf0, %output
  }
}
```

---

### Output Format (LLVM Dialect)

```mlir
module attributes { ... } {
  // External declarations to C++ runtime functions
  llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr) -> !llvm.ptr
  llvm.func @hipdnn_ep_get_constant(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @wrap_miopenConvolutionForward(
      !llvm.ptr,                      // state (RuntimeState* — opaque)
      !llvm.ptr, i64, i64, i64, i64,  // input ptr, N, C, H, W
      !llvm.ptr, i64,                 // weights ptr, K (output channels)
      !llvm.ptr,                      // bias ptr (nullable)
      !llvm.ptr, i64, i64,            // output ptr, H', W'
      i64, i64, i64, i64,             // kernel_h, kernel_w, stride_h, stride_w
      i64, i64, i64, i64,             // pad_top, pad_left, pad_bottom, pad_right
      i64, i64, i64                   // dilation_h, dilation_w, group
  ) -> i32

  // Transformed @main_graph — 3-parameter array-based interface
  llvm.func @main_graph(%context: !llvm.ptr,
                        %inputs: !llvm.ptr,
                        %outputs: !llvm.ptr) -> i32 {
    // Load input/output memref structs from arrays
    // Extract data pointers and shapes
    // Call @main_internal with unpacked params
  }
}
```

---

## Key Transformations

### 1. Lower hip.get_pool

```mlir
// Before
%pool = hip.get_pool(%ctx) : memref<?xi8, 1>

// After
%pool_ptr = llvm.call @hipdnn_ep_get_pool_base(%ctx) : (!llvm.ptr) -> !llvm.ptr
// (then reconstructed as memref descriptor for memref.view lowering)
```

Runtime implementation returns `RuntimeState.pool_base`.

### 2. Lower hip.get_constant

```mlir
// Before
%weights = hip.get_constant(%ctx, 0) : memref<64x3x3x3xf32, 1>

// After
%c0 = llvm.mlir.constant(0 : i64) : i64
%weights_ptr = llvm.call @hipdnn_ep_get_constant(%context, %c0)
    : (!llvm.ptr, i64) -> !llvm.ptr
```

Runtime implementation retrieves `RuntimeState.gpu_constants[index]`. See [../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md).

### 3. Lower HIP Compute Ops

**Pattern:** `hip.conv` → external declaration + `LLVM::CallOp`

```mlir
// Before
hip.conv ins(%ctx, %input, %weights, %bias : ...)
         outs(%output : memref<1x64x224x224xf32, 1>) {
  kernel_shape = [3, 3], strides = [1, 1], ...
}

// After
%ret = llvm.call @wrap_miopenConvolutionForward(
  %context,
  %input_ptr, %N, %C, %H, %W,
  %weights_ptr, %K,
  %bias_ptr,
  %output_ptr, %H_out, %W_out,
  3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1
) : (!llvm.ptr, ...) -> i32
```

**Type conversions:**
- `!hip.context` → `!llvm.ptr`
- `memref<...>` → `!llvm.struct<(ptr, ptr, i64, array<Nxi64>, array<Nxi64>)>`

### 4. Transform @main_graph Signature

The standard memref-to-llvm conversion unpacks memrefs into scalar components. For a rank-4 tensor this produces 11 parameters (2 pointers + 1 offset + 4 sizes + 4 strides). With 1 input and 1 output, `@main_graph` would have 23 parameters.

**Solution:** Two-function wrapper architecture:

1. **`@main_graph`** (3 parameters — public interface required by GenerateInterfacePass):
   - Signature: `(%context: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32`
   - Loads memref structs from arrays using GEP + load
   - Calls `@main_internal` with unpacked parameters

2. **`@main_internal`** (23+ parameters — private):
   - Original unpacked signature from standard conversion
   - Contains the actual computation

**Flow:**
```
Standard MLIR conversion
    ↓
@main_graph with N unpacked params
    ↓ transformMainFunction()
Rename to @main_internal (private)
    ↓
Create new @main_graph (3 params)
    ↓
@main_graph loads memref structs, extracts fields, calls @main_internal
```

**Metadata used:**
- `hipdnn.input_count` / `hipdnn.input_shapes` — number and shape of inputs (rank = shape array length)
- `hipdnn.output_count` / `hipdnn.output_shapes` — number and shape of outputs

For each rank-R tensor, unpacking extracts `2 + 1 + R + R` parameters.

---

## Implementation Strategy

The pass uses external function declarations rather than generating inline MIOpen code in MLIR. Runtime implementations handle all library complexity. See [WHY-HIP-WRAPPERS.md](WHY-HIP-WRAPPERS.md) for rationale.

**Lowering patterns registered:**
- `GetPoolOpLowering` — `hip.get_pool` → `@hipdnn_ep_get_pool_base`
- `GetConstantOpLowering` — `hip.get_constant` → `@hipdnn_ep_get_constant`
- `ConvOpLowering` — `hip.conv` → `@wrap_miopenConvolutionForward`
- `GemmOpLowering` — `hip.gemm` → `@wrap_hipblasSgemm`
- *(one pattern per HIP compute op)*

**Type conversion:**
```cpp
typeConverter.addConversion([](hip::ContextType) -> Type {
  return LLVM::LLVMPointerType::get(context);
});
```

`memref.view` and `memref.alloc` (without pooling) are lowered by `finalize-memref-to-llvm`.

---

## Related Documents

- **[02-OnnxToHip.md](02-OnnxToHip.md)** - Previous pass (generates HIP dialect)
- **[04-MemoryPooling.md](04-MemoryPooling.md)** - Emits hip.get_pool and memref.view
- **[06-GenerateInterfacePass.md](06-GenerateInterfacePass.md)** - Next pass (C interface generation, pool lifecycle)
- **[WHY-HIP-WRAPPERS.md](WHY-HIP-WRAPPERS.md)** - Wrapper architecture rationale
- **[../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md)** - Interface design
- **[../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** - Pipeline overview
- **[../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md)** - Constant management
- **[../../MEMORY-MANAGEMENT.md](../../MEMORY-MANAGEMENT.md)** - Overall memory management strategy
- **[../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md)** - Bitcode merging pipeline
- **[../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md)** - Dynamic shape flow
