<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HipToLLVM Pass

**Date:** 2026-02-20
**Document Type:** Design
**Status:** Draft
**Related:** [01-OnnxToHip.md](01-OnnxToHip.md), [04-MemoryPooling.md](04-MemoryPooling.md), [06-GenerateInterfacePass.md](06-GenerateInterfacePass.md)

**Input:** HIP dialect module
**Output:** LLVM dialect module with external runtime function calls

---

## Overview

The HipToLLVM pass lowers HIP dialect operations to LLVM dialect, calling external C++ runtime functions that encapsulate MIOpen/hipBLAS complexity.

**Key transformations:**
1. Lower !hip.context → !llvm.ptr
2. Transform @main_graph to array-based interface (3 parameters)
3. Convert HIP ops to calls to external runtime wrappers
4. Lower constant access helpers to LLVM

---

## Transformations

### Input Format (HIP Dialect)

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
} {
  func.func @main_graph(%ctx: !hip.context,
                   %input: memref<1x3x224x224xf32>,
                   %output: memref<1x64x224x224xf32>) -> i32 {

    %weights = hip.get_constant(%ctx, 0) : (!hip.context, i64) -> memref<64x3x3x3xf32, 1>
    %bias = hip.get_constant(%ctx, 1) : (!hip.context, i64) -> memref<64xf32, 1>

    %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    hip.conv(%ctx, %input, %weights, %bias, %temp) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1
    } : (!hip.context, memref<...>, memref<...>, memref<...>, memref<...>)

    memref.copy %temp, %output
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
```

---

### Output Format (LLVM Dialect)

The pass generates external function declarations and transforms the @main_graph function to use memref struct arrays.

```mlir
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_ranks = dense<[4]> : tensor<1xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_ranks = dense<[2]> : tensor<1xi64>
} {
  // 1. External declarations to C++ runtime functions
  //    Implementations in lib/Runtime/
  llvm.func @wrap_miopenConvolutionForward(
      !llvm.ptr,                      // state (RuntimeState* - opaque)
      !llvm.ptr, i64, i64, i64, i64,  // input ptr, N, C, H, W
      !llvm.ptr, i64,                 // weights ptr, K (output channels)
      !llvm.ptr,                      // bias ptr (nullable)
      !llvm.ptr, i64, i64,            // output ptr, H', W'
      i64, i64, i64, i64,             // kernel_h, kernel_w, stride_h, stride_w
      i64, i64, i64, i64,             // pad_top, pad_left, pad_bottom, pad_right
      i64, i64, i64                   // dilation_h, dilation_w, group
  ) -> i32

  llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr

  // 2. Transformed @main_graph - array-based interface (3 parameters)
  llvm.func @main_graph(%context: !llvm.ptr,
                  %inputs: !llvm.ptr,   // Array of memref structs
                  %outputs: !llvm.ptr)  // Array of memref structs
                  -> i32 {

    // Load input/output memref structs from arrays
    %input_0 = llvm.load %inputs[0] : !llvm.ptr
      -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
    %output_0 = llvm.load %outputs[0] : !llvm.ptr
      -> !llvm.struct<...>

    // Extract data pointers from memref structs
    %input_ptr = llvm.extractvalue %input_0[1] : !llvm.struct<...> -> !llvm.ptr<1>
    %output_ptr = llvm.extractvalue %output_0[1] : !llvm.struct<...> -> !llvm.ptr<1>

    // Extract shapes (compile-time constants from memref types)
    %input_n = llvm.mlir.constant(1 : i64) : i64   // batch size
    %input_c = llvm.mlir.constant(3 : i64) : i64   // input channels
    %input_h = llvm.mlir.constant(224 : i64) : i64 // height
    %input_w = llvm.mlir.constant(224 : i64) : i64 // width
    %output_h = llvm.mlir.constant(224 : i64) : i64
    %output_w = llvm.mlir.constant(224 : i64) : i64

    // Get constants from RuntimeState (opaque access)
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %weights_ptr = llvm.call @hipdnn_ep_constant_get(%context, %c0)
    %weights_k = llvm.mlir.constant(64 : i64) : i64  // output channels
    %bias_ptr = llvm.mlir.zero : !llvm.ptr  // no bias

    // Call C++ runtime wrapper (opaque state pointer!)
    %ret = llvm.call @wrap_miopenConvolutionForward(
      %context,                                // state (opaque!)
      %input_ptr, %input_n, %input_c, %input_h, %input_w,
      %weights_ptr, %weights_k,
      %bias_ptr,
      %output_ptr, %output_h, %output_w,
      3, 3, 1, 1,                             // kernel, stride
      1, 1, 1, 1,                             // pad
      1, 1, 1                                  // dilation, group
    )

    llvm.return %c0 : i32
  }
}
```

**Key points:**
- External declarations (no inline implementations) - runtime handles complexity
- @main_graph transformed to 3-parameter array-based interface
- Memref structs loaded from arrays, data pointers and shapes extracted
- Shapes passed as compile-time constants (i64 values)
- Wrapper receives opaque RuntimeState* and extracts handle/stream internally
- **No direct field access in generated code** - follows opaque pointer pattern (RUNTIME-ARCHITECTURE.md)

**Runtime implementations:**
- `wrap_miopenConvolutionForward()` in lib/Runtime/hipdnn_ep_runtime_miopen.cpp
- `hipdnn_ep_constant_get()` in lib/Runtime/hipdnn_ep_runtime_constants.cpp

The runtime functions handle all MIOpen complexity: descriptor creation, algorithm finding, workspace allocation, and cleanup.

---

### Memory Pooling Integration

HipToLLVM consumes metadata from [MemoryPooling](04-MemoryPooling.md) pass to generate pool-based allocation.

**Metadata consumed:**
- `hipdnn.pool_size` (module attribute) - Total pool bytes
- `hipdnn.buffer_offsets` (module attribute) - Array of offsets
- `hipdnn.buffer_index` (operation attribute on `hip.alloc`) - Index into offset array

**Input (hip.alloc with pool metadata):**
```mlir
module attributes {
  hipdnn.pool_size = 12845056 : i64,
  hipdnn.buffer_offsets = dense<[0, 8388608, ...]> : tensor<5xi64>
} {
  func @main_graph(...) {
    %temp = hip.alloc(%ctx) : memref<...> {hipdnn.buffer_index = 0 : i64}
  }
}
```

**Output (LLVM dialect with pool call):**
```mlir
%c0 = llvm.mlir.constant(0 : i64) : i64
%buf_ptr = llvm.call @hipdnn_ep_get_buffer_from_pool(%state, %c0)
    : (!llvm.ptr, i64) -> !llvm.ptr
```

**Runtime implementation:**
```cpp
void* hipdnn_ep_get_buffer_from_pool(RuntimeState* state, size_t index) {
    return (char*)state->pool_base + state->buffer_offsets[index];
}
```

**Free operation handling:**
- **With pooling:** `hip.free` → nop (pool deallocated in inference_cleanup)
- **Without pooling:** `hip.free` → `llvm.call @hipFree(...)`

**Pool lifecycle:**
- `inference_init`: Allocates pool via `hipMalloc` (size from `hipdnn.pool_size`)
- `inference_compute`: `get_buffer_from_pool` returns `pool_base + offset`
- `inference_cleanup`: Frees entire pool with single `hipFree` call

See [06-GenerateInterfacePass.md](06-GenerateInterfacePass.md) for pool initialization and cleanup.

**Performance:**
- **Memory savings:** 60% reduction (demo model: 12.8MB vs 32.1MB)
- **Allocation efficiency:** O(1) pointer arithmetic vs O(n) hipMalloc calls
- **Fragmentation:** Eliminated (single contiguous pool)

See [04-MemoryPooling.md](04-MemoryPooling.md) for algorithm details.

---

### Key Transformations

#### 1. Transform @main_graph Signature

**Critical transformation to satisfy GenerateInterfacePass Prerequisite 1.**

**Problem:** Standard MLIR memref-to-llvm conversion unpacks memrefs into scalar components. For a rank-4 tensor, this creates **11 parameters** (2 pointers + 1 offset + 4 sizes + 4 strides). With 1 input and 1 output, @main_graph gets **23 parameters** - barely readable and doesn't scale.

**Solution:** Two-function wrapper architecture:

1. **@main_graph** (3 parameters): Clean array-based interface
   - Signature: `(context: !llvm.ptr, inputs: !llvm.ptr, outputs: !llvm.ptr) -> i32`
   - Loads memref structs from arrays using GEP + load
   - Unpacks structs using extractvalue (11 extracts per rank-4 tensor)
   - Calls @main_internal with unpacked parameters
   - Private (not exported from DLL)

2. **@main_internal** (23+ parameters): Computation logic
   - Original unpacked signature from standard conversion
   - Contains actual computation (calls to runtime wrappers, etc.)
   - Private (not exported from DLL)

**Transformation Flow:**

```
Standard MLIR conversion
    ↓
@main_graph with 23 unpacked params (allocated, aligned, offset, sizes[4], strides[4] for each tensor)
    ↓ transformMainFunction()
Rename to @main_internal (private)
    ↓
Create new @main_graph (3 params)
    ↓
@main_graph loads memref structs from arrays
    ↓
@main_graph extracts 11 fields per tensor using llvm.extractvalue
    ↓
@main_graph calls @main_internal with 23 unpacked params
```

**Code Example:**

```mlir
// NEW: Clean 3-parameter wrapper
llvm.func private @main_graph(%ctx: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32 {
  // Load input memref struct from inputs[0]
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %input_ptr = llvm.getelementptr %inputs[%c0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
  %input = llvm.load %input_ptr : !llvm.ptr
    -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  // Extract 11 fields
  %allocated = llvm.extractvalue %input[0] : !llvm.struct<...> -> !llvm.ptr<1>
  %aligned = llvm.extractvalue %input[1] : !llvm.struct<...> -> !llvm.ptr<1>
  %offset = llvm.extractvalue %input[2] : !llvm.struct<...> -> i64
  %size0 = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64  // Runtime dimension!
  // ... extract remaining sizes and strides

  // Load output struct (similar)
  // ...

  // Call computation function with 23 unpacked params
  %result = llvm.call @main_internal(%ctx, %allocated, %aligned, %offset,
                                      %size0, %size1, %size2, %size3,
                                      %stride0, %stride1, %stride2, %stride3,
                                      %out_allocated, %out_aligned, ...)
  llvm.return %result : i32
}

// Internal computation (original body, 23 parameters)
llvm.func private @main_internal(%ctx: !llvm.ptr, %arg1: !llvm.ptr<1>, ..., %arg22: i64) -> i32 {
  // Rebuild memref descriptors from unpacked params
  %0 = llvm.mlir.poison : !llvm.struct<...>
  %1 = llvm.insertvalue %arg12, %0[0] : ...
  // ... (computation logic)
}
```

**Metadata-Driven:** Uses module attributes to determine structure:
- `hipdnn.input_count` - number of input tensors
- `hipdnn.input_ranks` - rank of each input (e.g., [4] for one rank-4 tensor)
- `hipdnn.output_count` - number of output tensors
- `hipdnn.output_ranks` - rank of each output

For each tensor with rank R, unpacking extracts **2 + 1 + R + R** parameters.

**Dynamic Shape Support:**
- Rank is compile-time (from metadata)
- Dimension values are runtime (loaded from memref struct)
- Unpacking preserves runtime dimension values
- No special handling needed - works automatically!

**Benefits:**
- ✅ Readable: 3 parameters instead of 23+
- ✅ Scalable: Works for any number of inputs/outputs
- ✅ Type-safe: Memref structs preserve shape information
- ✅ Satisfies Prerequisite 1: Required by GenerateInterfacePass
- ✅ Dynamic shape ready: Runtime dimensions flow through unchanged

**Implementation:** lib/HipDialect/HipToLLVM.cpp

---

#### 2. Lower HIP Operations

The pass converts HIP dialect operations to calls to external C++ runtime functions.

**Pattern:** `hip.conv` → external declaration + LLVM::CallOp

```cpp
struct ConvOpLowering : public ConvertOpToLLVMPattern<hip::ConvOp> {
  LogicalResult matchAndRewrite(hip::ConvOp op, ...) {
    // Build arguments: handle, stream, data pointers, shape pointers, attributes
    SmallVector<Value> args = {
      adaptor.getHandle(),
      adaptor.getStream(),
      adaptor.getInputPtr(),
      adaptor.getInputShapePtr(),
      adaptor.getWeightsPtr(),
      adaptor.getWeightsShapePtr(),
      adaptor.getOutputPtr(),
      adaptor.getOutputShapePtr(),
      // ... kernel, stride, pad, dilation params from attributes
    };

    // Replace hip.conv with call to external C++ runtime wrapper
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, "@wrap_miopenConvolutionForward", args);
    return success();
  }
};
```

**Type conversions:**
- !hip.context → !llvm.ptr
- memref → !llvm.struct (allocated, aligned, offset, sizes, strides)

**Constant access:** `hip.get_constant` → call to runtime function

```mlir
// Before
%weights = hip.get_constant(%ctx, 0) : (!hip.context, i64) -> memref<64x3x3x3xf32, 1>

// After
%weights_ptr = llvm.call @hipdnn_ep_get_constant(%context, %c0) : (!llvm.ptr, i64) -> !llvm.ptr
```

Runtime implementation retrieves the GPU pointer from RuntimeState.gpu_constants[index]. See [../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md) for constant upload and access flow.

---

## Implementation Strategy

### Why External Runtime Wrappers

The HipToLLVM pass uses a **Phase 1 architecture** that calls pre-existing C++ runtime functions rather than generating elaborate inline MIOpen code in MLIR.

**Rationale:**
1. **Simpler debugging** - Runtime code can be debugged with standard C++ tools
2. **Faster development** - Changes don't require recompilation of generated DLLs
3. **Runtime evolution** - MIOpen API updates only touch lib/Runtime/*.cpp
4. **Cleaner IR** - Generated LLVM is minimal (external declarations + calls)

**Current design:** External function declarations in MLIR, implementations in lib/Runtime/hipdnn_ep_runtime_miopen.cpp

**Future Phase 2:** May generate wrapper bodies in MLIR for more aggressive optimization opportunities, but Phase 1 provides foundation.

See [WHY-HIP-WRAPPERS.md](WHY-HIP-WRAPPERS.md) for detailed rationale comparing wrapper vs inline approaches.

---

### Pass Implementation

**Lowering patterns:**
- `ConvOpLowering` - hip.conv → @wrap_miopenConvolutionForward
- `GemmOpLowering` - hip.gemm → @wrap_hipblasSgemm
- `GetConstantOpLowering` - hip.get_constant → @hipdnn_ep_get_constant

**Pattern registration:**
```cpp
void populateHipToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                RewritePatternSet &patterns) {
  patterns.add<ConvOpLowering>(typeConverter);
  patterns.add<AllocOpLowering>(typeConverter);
  patterns.add<GetConstantOpLowering>(typeConverter);
  // ... more patterns
}
```

**Type conversion:**
```cpp
typeConverter.addConversion([](hip::ContextType) -> Type {
  return LLVM::LLVMPointerType::get(context);
});

typeConverter.addConversion([](MemRefType type) -> Type {
  return convertMemRefToLLVMStruct(type);  // Struct-by-value!
});
```

**File locations:**
The generated LLVM bitcode is merged with runtime bitcode at the IR level before linking. See [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) for details.

---

## Related Documents

- **[01-OnnxToHip.md](01-OnnxToHip.md)** - Previous pass (generates HIP dialect)
- **[04-MemoryPooling.md](04-MemoryPooling.md)** - Pool metadata generation
- **[06-GenerateInterfacePass.md](06-GenerateInterfacePass.md)** - Next pass (C interface generation, pool lifecycle)
- **[WHY-HIP-WRAPPERS.md](WHY-HIP-WRAPPERS.md)** - Wrapper architecture rationale
- **[../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md)** - Interface design
- **[../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** - Pipeline overview
- **[../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md)** - Constant management
- **[../../MEMORY-MANAGEMENT.md](../../MEMORY-MANAGEMENT.md)** - Overall memory management strategy
- **[../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md)** - Bitcode merging pipeline
- **[../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md)** - Dynamic shape flow
