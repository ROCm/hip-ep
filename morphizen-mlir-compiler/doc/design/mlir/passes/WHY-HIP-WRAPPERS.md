<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Why HipToLLVM Uses Wrapper Functions

**Document Type:** Design
**Review Status:** Draft
**Context:** MLIR HipToLLVM pass - explains why we use wrapper functions (whether generated in MLIR or pre-existing in C++)

**Current Implementation:** Phase 1 uses pre-existing C++ runtime functions (`lib/Runtime/hipdnn_ep_runtime_miopen.cpp`). Future Phase 2 may generate MLIR wrappers.

---

## The Problem: Code Duplication Without Wrappers

Without wrapper functions, every `hip.conv` operation in the IR would need to duplicate the same complex logic:

1. **Extract memref dimensions** - Pull runtime dimension values from memref structs
2. **Extract data pointers** - Get GPU memory addresses from memref structs
3. **Create GPU API descriptors** - Set up MIOpen tensor/convolution descriptors
4. **Call GPU library API** - Invoke `miopenConvolutionForward` with 13+ parameters
5. **Cleanup descriptors** - Destroy MIOpen descriptors

For a module with 10 convolution operations, this logic would be **duplicated 10 times** in the generated LLVM IR.

**Example of what we'd generate without wrappers (per operation):**

```mlir
// For EACH hip.conv operation, we'd inline ALL of this:
%input_ptr = llvm.extractvalue %input[1] : !llvm.struct<...>
%input_n = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64
%input_c = llvm.extractvalue %input[3, 1] : !llvm.struct<...> -> i64
%input_h = llvm.extractvalue %input[3, 2] : !llvm.struct<...> -> i64
%input_w = llvm.extractvalue %input[3, 3] : !llvm.struct<...> -> i64
// ... 40+ more extractvalue ops for weights, bias, output
// ... create descriptors
// ... call MIOpen
// ... cleanup
// Repeated for EVERY hip.conv in the module!
```

**Problems:**
- Bloated IR (50+ lines per operation)
- Harder to read and debug
- Harder to optimize (LLVM must recognize duplicate code)
- Maintenance burden (bug fixes need to update pattern generation)

---

## The Solution: Reusable Wrapper Functions

Instead, HipToLLVM uses **one wrapper function** that encapsulates the common logic.

**Phase 1 Implementation:** External C++ runtime functions (current):

```mlir
// External declaration to C++ function in lib/Runtime/hipdnn_ep_runtime_miopen.cpp
llvm.func @wrap_miopenConvolutionForward(
    !llvm.ptr,      // handle
    !llvm.ptr,      // stream
    !llvm.ptr,      // input data pointer
    !llvm.ptr,      // input_shape (int64_t[4])
    !llvm.ptr,      // weights data pointer
    !llvm.ptr,      // weights_shape (int64_t[4])
    !llvm.ptr,      // output data pointer
    !llvm.ptr,      // output_shape (int64_t[4])
    i64, i64,       // pad_h, pad_w
    i64, i64,       // stride_h, stride_w
    i64, i64        // dilation_h, dilation_w
) -> i32
```

**Future Phase 2:** May generate inline MLIR wrappers:

```mlir
// Generated ONCE per module
llvm.func @hip_conv_wrapper(
    %ctx: !llvm.ptr,
    %input: !llvm.struct<...>,
    %weights: !llvm.struct<...>,
    %bias: !llvm.struct<...>,
    %output: !llvm.struct<...>,
    %kernel_h: i64, %kernel_w: i64,
    %stride_h: i64, %stride_w: i64,
    %pad_top: i64, %pad_left: i64, %pad_bottom: i64, %pad_right: i64,
    %dilation_h: i64, %dilation_w: i64,
    %group: i64) -> i32 {

  // Extract dimensions from memref structs
  %input_ptr = llvm.extractvalue %input[1] : !llvm.struct<...>
  %input_n = llvm.extractvalue %input[3, 0] : !llvm.struct<...> -> i64
  %input_c = llvm.extractvalue %input[3, 1] : !llvm.struct<...> -> i64
  // ... extract all dimensions

  // Create MIOpen descriptors
  %xDesc = llvm.call @miopenCreateTensorDescriptor()
  llvm.call @miopenSet4dTensorDescriptor(%xDesc, %dataType,
                                          %input_n, %input_c, %input_h, %input_w)
  // ... create weight, output, convolution descriptors

  // Call MIOpen
  %ret = llvm.call @miopenConvolutionForward(%miopen, %alpha_ptr,
                                              %xDesc, %input_ptr,
                                              %wDesc, %weights_ptr,
                                              %convDesc, %algo,
                                              %beta_ptr,
                                              %yDesc, %output_ptr,
                                              %workspace, %workspace_size)

  // Cleanup
  llvm.call @miopenDestroyTensorDescriptor(%xDesc)
  llvm.call @miopenDestroyTensorDescriptor(%wDesc)
  // ... destroy remaining descriptors

  llvm.return %ret : i32
}
```

**Phase 1:** Every `hip.conv` operation calls the C++ runtime wrapper:

```mlir
// For EACH hip.conv operation - just 1 call!
%ret = llvm.call @wrap_miopenConvolutionForward(
  %handle, %stream,
  %input_ptr, %input_shape_ptr,
  %weights_ptr, %weights_shape_ptr,
  %output_ptr, %output_shape_ptr,
  1, 1,        // pad_h, pad_w
  1, 1,        // stride_h, stride_w
  1, 1         // dilation_h, dilation_w
) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
     !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32
```

**Phase 2:** Would call generated MLIR wrapper (future):

```mlir
// For EACH hip.conv operation - call generated wrapper
%ret = llvm.call @hip_conv_wrapper(%ctx, %input, %weights, %bias, %output,
                                    3, 3,        // kernel
                                    1, 1,        // strides
                                    1, 1, 1, 1,  // pads
                                    1, 1,        // dilations
                                    1)           // group
```

---

## Wrapper Responsibilities

The wrapper functions (whether C++ runtime or MLIR-generated) handle:

### 1. **Memref Unpacking**
Extract runtime information from memref struct fields:
- Allocated pointer (`field[0]`)
- Aligned data pointer (`field[1]`)
- Offset (`field[2]`)
- Dimension sizes (`field[3, i]`) - values from size array
- Strides (`field[4, i]`)

### 2. **Descriptor Creation**
Create GPU library API descriptors:
- MIOpen: `miopenTensorDescriptor_t`, `miopenConvolutionDescriptor_t`
- hipBLAS: `hipblasHandle_t` (from context)

### 3. **GPU API Calls**
Invoke the actual GPU library functions:
- `miopenConvolutionForward`
- `miopenPoolingForward`
- `hipblasSgemm`
- etc.

### 4. **Cleanup**
Destroy descriptors to prevent resource leaks:
- `miopenDestroyTensorDescriptor`
- `miopenDestroyConvolutionDescriptor`

---

## Key Benefits

### **Code Reuse**
One wrapper function, many call sites. 10 convolutions = 1 wrapper + 10 calls, not 10 duplications.

### **Cleaner IR**
Each operation becomes a single function call instead of 50+ lines of inline code.

### **LLVM Optimization**
LLVM's inliner can inline the wrapper at call sites if profitable. LLVM decides optimal inlining strategy, not us.

### **Easier Maintenance**
Bug fixes or optimizations update one wrapper generation function, not the pattern for every operation.

### **Dynamic Shape Support**
Wrappers extract dimensions at runtime from memref structs. No compile-time dimension values needed!

Example:
```mlir
// Wrapper extracts RUNTIME dimension values
%input_h = llvm.extractvalue %input[3, 2] : !llvm.struct<...> -> i64  // RUNTIME!

// Passes runtime values to MIOpen
llvm.call @miopenSet4dTensorDescriptor(%xDesc, %dataType,
                                        %n, %c, %input_h, %w)  // Works for ANY height!
```

---

## Wrapper Strategy

**Phase 1 (Current):** External C++ Runtime Functions
- Wrapper functions pre-implemented in `lib/Runtime/hipdnn_ep_runtime_miopen.cpp`
- HipToLLVM generates external function declarations
- No MLIR code generation needed for wrapper bodies
- Simpler implementation, easier debugging

**Phase 2 (Future):** On-Demand MLIR Wrapper Generation

```cpp
static LLVM::LLVMFuncOp getOrCreateConvWrapper(ModuleOp module, OpBuilder &builder) {
  // Check if wrapper already exists
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>("hip_conv_wrapper"))
    return func;  // Reuse existing wrapper

  // First hip.conv in module - generate wrapper in MLIR
  auto wrapper = builder.create<LLVM::LLVMFuncOp>(...);
  // ... build wrapper body with MIOpen calls
  return wrapper;
}
```

**When lowering a `hip.conv` operation:**
1. Call `getOrCreateConvWrapper` - declares/generates wrapper
2. Build call arguments (context, memrefs, attributes)
3. Replace `hip.conv` with `llvm.call @wrap_miopenConvolutionForward` (Phase 1) or `@hip_conv_wrapper` (Phase 2)

---

## Example: Before vs After

### Before (Without Wrappers)

```mlir
func.func @main(...) {
  hip.conv(%ctx, %input1, %weights1, %bias1, %output1) {...}
  // Would inline 50+ lines of extraction + MIOpen calls

  hip.conv(%ctx, %input2, %weights2, %bias2, %output2) {...}
  // Would duplicate the same 50+ lines again

  hip.conv(%ctx, %input3, %weights3, %bias3, %output3) {...}
  // Would duplicate the same 50+ lines again
}
```

**Generated IR size:** ~150 lines (50 lines × 3 operations)

### After (With Wrappers)

**Phase 1 (Current - C++ Runtime):**

```mlir
// External declaration (1 line)
llvm.func @wrap_miopenConvolutionForward(...) -> i32

llvm.func @main(...) {
  llvm.call @wrap_miopenConvolutionForward(%handle, %stream, ...)  // 1 line
  llvm.call @wrap_miopenConvolutionForward(%handle, %stream, ...)  // 1 line
  llvm.call @wrap_miopenConvolutionForward(%handle, %stream, ...)  // 1 line
}
```

**Generated IR size:** ~4 lines (1 declaration + 3 calls)

**Phase 2 (Future - MLIR-Generated):**

```mlir
// Generated once
llvm.func @hip_conv_wrapper(...) { ... }  // 50 lines

llvm.func @main(...) {
  llvm.call @hip_conv_wrapper(%ctx, %input1, ...)  // 1 line
  llvm.call @hip_conv_wrapper(%ctx, %input2, ...)  // 1 line
  llvm.call @hip_conv_wrapper(%ctx, %input3, ...)  // 1 line
}
```

**Generated IR size:** ~53 lines (50-line wrapper + 3 calls)

---

## Wrapper Functions per Operation Type

HipToLLVM uses specialized wrappers for each HIP dialect operation:

| HIP Operation | Phase 1 (C++ Runtime) | Phase 2 (MLIR-Generated) | GPU Library API |
|--------------|----------------------|---------------------------|-----------------|
| `hip.conv` | `@wrap_miopenConvolutionForward` | `@hip_conv_wrapper` | `miopenConvolutionForward` |
| `hip.pool` | `@wrap_miopenPoolingForward` | `@hip_pool_wrapper` | `miopenPoolingForward` |
| `hip.gemm` | `@wrap_hipblasSgemm` | `@hip_gemm_wrapper` | `hipblasSgemm` |
| `hip.batch_norm` | `@wrap_miopenBatchNorm` | `@hip_batch_norm_wrapper` | `miopenBatchNormalizationForwardInference` |
| `hip.relu` | `@wrap_miopenActivation` | `@hip_relu_wrapper` | `miopenActivationForward` |

**Phase 1:** Each wrapper is pre-implemented in C++ runtime.
**Phase 2:** Each wrapper would be generated **once per module** and reused by all operations of that type.

---

## Implementation

**Phase 1 (Current):**
- **MLIR Pass:** `lib/HipDialect/HipToLLVM.cpp`
  - `ConvOpLowering::matchAndRewrite()` - Lower `hip.conv` to call `@wrap_miopenConvolutionForward`
  - Generates external function declarations
- **C++ Runtime:** `lib/Runtime/hipdnn_ep_runtime_miopen.cpp`
  - `wrap_miopenConvolutionForward()` - Actual wrapper implementation
  - Contains all MIOpen descriptor creation, algorithm finding, cleanup logic

**Phase 2 (Future):**
- `getOrCreateConvWrapper()` - Generate/reuse `@hip_conv_wrapper` in MLIR
- `ConvOpLowering::matchAndRewrite()` - Lower `hip.conv` to wrapper call
- Similar functions for other operations (pool, gemm, etc.)

See [05-HipToLLVM.md](05-HipToLLVM.md) for complete pass documentation.

---

## Related Documents

- [05-HipToLLVM.md](05-HipToLLVM.md) - Complete HipToLLVM pass documentation
- [../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md) - Wrapper function design details
- [../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape support
