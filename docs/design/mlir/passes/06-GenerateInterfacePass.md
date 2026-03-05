<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

**Date:** 2026-03-02
**Document Type:** Implementation
**Status:** Draft
**Related:** [02-OnnxToHip.md](02-OnnxToHip.md), [05-HipToLLVM.md](05-HipToLLVM.md), [INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md)

---

## Table of Contents

- [GenerateInterfacePass](#generateinterfacepass)
- [Overview](#overview)
- [Stable Contract Summary](#stable-contract-summary)
- [Prerequisites](#prerequisites)
  - [A. Verified Prerequisites](#a-verified-prerequisites)
  - [B. Design Contracts](#b-design-contracts)
  - [Prerequisites Summary Table](#prerequisites-summary-table)
- [Detailed MLIR Implementations](#detailed-mlir-implementations)
  - [Function 1: inference_init](#function-1-inference_init)
  - [Function 2: inference_compute](#function-2-inference_compute)
  - [Function 3: inference_cleanup](#function-3-inference_cleanup)
- [Verifying C-ABI Compliance](#verifying-c-abi-compliance)
- [Dynamic Shape Support](#dynamic-shape-support)
- [Implementation Strategy](#implementation-strategy)
- [Related Documents](#related-documents)

---

# GenerateInterfacePass

**Input:** LLVM dialect module with @main_graph + constant helpers
**Output:** LLVM dialect module + C interface wrappers

---

## Overview

The GenerateInterfacePass generates the three C interface functions that are exported from the compiled DLL. These functions implement the interface defined in [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) and wrap the internal @main_graph and constant helper functions.

**Generated functions:**
1. `inference_init` - Initializes runtime state, passes metadata blob to runtime
2. `inference_compute` - Parses span_t*, builds memrefs, calls @main
3. `inference_cleanup` - Frees GPU resources, destroys handles
4. `inference_get_metadata_json` - Returns pointer to `__metadata_json` global

**Generated globals (internal linkage):**
- `__metadata_blob` - FlatBuffers binary (`UdnaModelMetaInfo`), passed to runtime by `inference_init`
- `__metadata_json` - JSON string, returned by `inference_get_metadata_json`

**This document describes:** Implementation details - how to generate the MLIR code

**For design rationale:** See [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) for WHAT and WHY

---

## Why This Pass Exists

This pass serves as an **adapter layer** between two incompatible representations:
- **MLIR world**: Statically-typed memref structs with compile-time rank information
- **C API world**: Dynamic span_t/tensor_t structures with runtime rank information

The pass cannot be eliminated and moved to runtime because MLIR's type system requires compile-time knowledge of tensor ranks to generate memref struct types (`{ptr, ptr, i64, [N x i64], [N x i64]}`). Moving this to runtime would require hardcoding MLIR's internal memref layout in C++ code, creating fragile coupling and losing type safety.

**Rationale**: MLIR's type system requires compile-time knowledge of tensor ranks for memref struct types, making a pure runtime solution fragile and error-prone.

---

## Stable Contract Summary

This pass depends on a simple contract from prior passes. **Any future pass that satisfies this contract can work with GenerateInterfacePass.**

### Required Functions:

1. **@main_graph function** (from HipToLLVM)
   - Signature: `llvm.func @main_graph(%ctx: !llvm.ptr, %inputs: !llvm.ptr, %outputs: !llvm.ptr) -> i32`
   - Behavior: Processes arrays of input/output memref structs, returns status code

### Required Module Attributes:

2. **I/O Metadata** (from OnnxToHip)
   - `hipdnn.input_count: i64` - Number of model inputs
   - `hipdnn.input_shapes: [dense<[...]>, ...]` - Shape of each input tensor (rank = length of shape array)
   - `hipdnn.input_element_sizes: dense<[...]>` - Bytes per element for each input (e.g., 4 for float32, 2 for float16, 8 for int64)
   - `hipdnn.output_count: i64` - Number of model outputs
   - `hipdnn.output_shapes: [dense<[...]>, ...]` - Shape of each output tensor
   - `hipdnn.output_element_sizes: dense<[...]>` - Bytes per element for each output

**Extensibility:** Future passes can replace OnnxToHip/HipToLLVM as long as they produce these functions and attributes.

---

## Prerequisites

Before the `GenerateInterfacePass` can run, prior passes must establish certain prerequisites. This section is divided into:

1. **A. Verified Prerequisites** - What `verifyPrerequisites()` actually checks in the code
2. **B. Design Contracts** - Important conventions not enforced by code but critical for correct implementation

**Interface Design for Dynamic Shapes**

Interface designed to support runtime dimension values. For implementation challenges, see [DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md).

Design:
- Tensor **rank** is compile-time known (e.g., 4D tensor)
- Dimension **values** loaded from tensor_t.shape pointer
- Interface accepts both static and runtime dimension values

**Status:** Dynamic shapes not yet implemented (memory pooling incompatibility)

---

## A. Verified Prerequisites

The following prerequisites are **enforced by code** via the `verifyPrerequisites()` function. The pass will fail with an error message if any of these are missing.

### Prerequisite 0: Idempotency Check

**What the code checks:** Pass verifies that `inference_init`, `inference_compute`, and `inference_cleanup` don't already exist.

**Why it matters:** Prevents duplicate interface generation if pass runs multiple times.

**Error message:**
```
[GenerateInterface] Interface functions already exist. Pass already ran.
```

### Prerequisite 1: @main_graph Function

**Required signature:**
```mlir
llvm.func @main_graph(%context: !llvm.ptr,
                %inputs: !llvm.ptr,
                %outputs: !llvm.ptr) -> i32
```

**What the code checks:**
1. `@main_graph` exists as `llvm.func` (NOT `func.func`)
2. Has exactly 3 parameters, all `!llvm.ptr`
3. Returns `i32`

**Satisfied by:** HipToLLVM pass (`transformMainFunction()` method)

**Error messages:**
```
[GenerateInterface] @main_graph (llvm.func) not found
[GenerateInterface] @main_graph is func.func, needs llvm.func. Run --convert-hip-to-llvm first.
[GenerateInterface] @main_graph has wrong signature. Expected: (ptr, ptr, ptr) -> i32
```

**See also:** [05-HipToLLVM.md](05-HipToLLVM.md) for implementation details

### Prerequisite 2: External Constants Attributes

OnnxToHip emits these attributes when constants are extracted to `constants.bin`:

```mlir
module attributes {
  hipdnn.constant_sizes = array<i64: 4096, 8192>,
  hipdnn.constant_offsets = array<i64: 0, 4096>
}
```

These attributes are read by `buildMetadataBlob()` to populate
`UdnaModelMetaInfo.constants` in the FlatBuffers blob. The runtime reads sizes
and offsets from the blob to locate each constant in `constants.bin`.

**Required attributes:**
```mlir
module attributes {
  hipdnn.input_count = 2 : i64,
  hipdnn.input_shapes = [dense<[1, 3, 224, 224]> : tensor<4xi64>,
                         dense<[4, 8]> : tensor<2xi64>],
  hipdnn.input_element_sizes = dense<[4, 4]> : tensor<2xi64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [dense<[1, 10]> : tensor<2xi64>],
  hipdnn.output_element_sizes = dense<[4]> : tensor<1xi64>
}
```

Tensor rank is derived from the length of each shape array; no separate
`hipdnn.input_ranks` / `hipdnn.output_ranks` attribute is emitted or read.

**What the code checks:**
1. `hipdnn.input_count` attribute exists
2. `hipdnn.input_shapes` attribute exists
3. `hipdnn.output_count` attribute exists
4. `hipdnn.output_shapes` attribute exists

**Optional attributes (used if present):**
5. `hipdnn.input_element_sizes` - bytes per element per input (defaults to 4 if absent)
6. `hipdnn.output_element_sizes` - bytes per element per output (defaults to 4 if absent)

**Satisfied by:** OnnxToHip pass

**Error messages:**
```
[GenerateInterface] hipdnn.input_count attribute missing
[GenerateInterface] hipdnn.input_shapes attribute missing
[GenerateInterface] hipdnn.output_count attribute missing
[GenerateInterface] hipdnn.output_shapes attribute missing
```

**Why this matters:** When @main_graph signature becomes `(context, inputs, outputs) → i32`, type information is lost. Metadata preserves this for memref struct construction.

---

## B. Design Contracts

The following are **NOT enforced by code** in `verifyPrerequisites()`, but are critical design contracts that the GenerateInterfacePass implementation depends on. These document important conventions and requirements.

### @main_graph Function Behavioral Contract

The code only checks that @main_graph exists with the correct signature `(ptr, ptr, ptr) -> i32`. However, the implementation relies on the following behavioral contracts:

**Design decisions:**
- **Calling convention:** Struct-by-value memrefs (NOT unpacked descriptors)
- **Multiple I/O support:** Arrays of memref structs
- **Dynamic shape support:** Memref size/stride arrays contain **runtime values**

**What GenerateInterfacePass expects:**
1. First parameter is state/context pointer
2. Second parameter points to array of input memref structs with **runtime dimensions**
3. Third parameter points to array of output memref structs with **runtime dimensions**
4. Returns i32 status code (0 = success, non-zero = error)
5. **Critical:** Memref size/stride arrays populated with runtime values (from tensor_t.shape)

**Example usage in @main_graph (with dynamic shapes):**
```mlir
llvm.func @main_graph(%context: !llvm.ptr,
                %inputs: !llvm.ptr,
                %outputs: !llvm.ptr) -> i32 {
  // Access input 0 (memref struct at inputs[0])
  %input_0_ptr = llvm.getelementptr %inputs[0] : (!llvm.ptr) -> !llvm.ptr
  // Load memref struct - size array contains RUNTIME dimension values!
  %input_0 = llvm.load %input_0_ptr : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  // Extract runtime dimensions from memref struct
  %n = llvm.extractvalue %input_0[3, 0] : !llvm.struct<...> -> i64  // Batch size (runtime!)
  %c = llvm.extractvalue %input_0[3, 1] : !llvm.struct<...> -> i64  // Channels (runtime!)
  %h = llvm.extractvalue %input_0[3, 2] : !llvm.struct<...> -> i64  // Height (runtime!)
  %w = llvm.extractvalue %input_0[3, 3] : !llvm.struct<...> -> i64  // Width (runtime!)

  // Computation using runtime dimensions...
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %c0_i32 : i32
}
```

**See also:** [05-HipToLLVM.md](05-HipToLLVM.md) for call chain walkthrough and LLVM optimization details.

### Metadata Blob Contract

`__metadata_blob` is a FlatBuffers binary (`UdnaModelMetaInfo` schema) stored as an internal-linkage global in the generated module.

**Postconditions:**
1. Valid `UdnaModelMetaInfo` FlatBuffers binary, readable via `flatbuffers::GetRoot<UdnaModelMetaInfo>()`
2. `constants_filename` field contains the constants file name (default: `"constants.bin"`)
3. `constant_sizes` field contains per-constant byte sizes matching `hipdnn.constant_sizes` module attribute

**Consumer responsibilities:**
- `inference_init` (generated): passes blob address + size to `hipdnn_ep_state_init_with_fs`
- Runtime: reads blob zero-copy (header-only FlatBuffers, no `libflatbuffers` link needed)
- External tools: use `__metadata_json` instead (JSON string via `inference_get_metadata_json`)

**See also:** [../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md) for constant handling details.

### RuntimeState Contract (Opaque Handle Design)

**Reference:** [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) - Opaque Handle Design

**RuntimeState is opaque:**
- Generated code treats it as `!llvm.ptr` (opaque pointer)
- Internal structure is known ONLY to runtime implementation (`lib/Runtime/hipdnn_ep_runtime_state.cpp`)
- Generated code NEVER accesses fields directly - only passes the pointer around
- Access to internal data via accessor functions: `@hipdnn_ep_get_stream`, `@hipdnn_ep_get_constant`

**Design:**
- **All generated code** (`inference_init`, `inference_compute`, `inference_cleanup`): Treat RuntimeState as opaque `!llvm.ptr`
- **Generated interface functions**: Delegate to runtime library (`@hipdnn_ep_state_init`, `@hipdnn_ep_state_cleanup`)
- **Generated computation code** (@main_graph, wrappers): Use accessor functions (`@hipdnn_ep_get_stream`, `@hipdnn_ep_get_constant`)
- **Runtime library only** (`hipdnn_ep_runtime.cpp`): Knows internal structure and accesses fields directly
- **Benefit**: Runtime can evolve internal structure without breaking any generated code

### Tensor Interface Contract (span_t and tensor_t)

**Defined in:** CustomOp header (external to MLIR compilation)

**Required structures:**
```c
// C struct for tensor metadata
typedef struct {
    void* data;        // Pointer to tensor data (CPU or GPU)
    int64_t* shape;    // Pointer to shape array (runtime dimensions)
    int rank;          // Number of dimensions (compile-time known rank)
    int data_type;     // Enum: FLOAT32=0, FLOAT16=1, INT8=2, etc.
} tensor_t;

// C struct for array of tensors
typedef struct {
    tensor_t* data;    // Pointer to array of tensor_t
    size_t count;      // Number of tensors in array
} span_t;
```

**What GenerateInterfacePass must do (CRITICAL: Dynamic Shape Support):**

1. Parse `span_t` to get `tensor_t` array
2. For each `tensor_t`:
   - Extract `data` pointer (cast to !llvm.ptr<1> for GPU address space)
   - Extract `shape` pointer (**LOAD RUNTIME DIMENSION VALUES**)
   - Extract `rank` (compile-time constant for this model)
   - Map `data_type` to element size (FLOAT32=4, FLOAT16=2, INT8=1)
3. Build memref struct with **runtime dimensions**
4. Calculate **runtime strides** from dimension values

**Key Design Decisions:**
- **Keep tensor_t simple:** Don't match memref structure exactly
- **Runtime stride calculation:** Compute strides from dimension values
- **Type system:** Rank (4D) is compile-time, dimension values loaded from shape pointer

**See:** [../INTERFACE-DESIGN.md - Data Structures](../INTERFACE-DESIGN.md#32-data-structures) for complete tensor interface specification.

---

## Prerequisites Summary Table

**Verified Prerequisites (enforced by code):**

| # | Prerequisite | Satisfied By | Code Check |
|---|--------------|--------------|------------|
| 0 | Idempotency | (generated code) | Functions don't exist yet |
| 1 | @main_graph function | HipToLLVM | Signature: `(ptr, ptr, ptr) -> i32` |
| 2 | Module metadata | OnnxToHip | 4 required attributes + 2 optional element size attributes |

**Design Contracts (NOT enforced by code):**

| Contract | Critical For | Documented In |
|----------|--------------|---------------|
| @main_graph behavioral contract | Correct memref handling | [05-HipToLLVM.md](05-HipToLLVM.md) |
| RuntimeState opaque design | ABI stability | [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) |
| Tensor interface (span_t/tensor_t) | C-ABI compatibility | [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) |
| Dynamic shape support | Runtime flexibility | [../../DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md) |

---

## Detailed MLIR Implementations

**Architectural Note:**

This pass uses a **two-tier approach** for the three interface functions:

1. **inference_init/cleanup**: Simple wrappers that delegate to runtime library functions
   - Rationale: Complex initialization logic (handle creation, error handling, LIFO cleanup) is easier to maintain in C++ than OpBuilder code
   - Implementation: Runtime library merged via llvm::Linker - same final binary, zero overhead after LLVM optimization

2. **inference_compute**: Full MLIR IR generation (cannot be moved to runtime)
   - Rationale: MLIR's type system requires compile-time knowledge of tensor ranks to generate memref struct types
   - Implementation: Must generate IR here because rank information is only available at compile time

### Function 1: inference_init

**C Signature:**
```c
int inference_init(void** out_state, void* fs);
```

**For design and rationale:** See [../INTERFACE-DESIGN.md - inference_init](../INTERFACE-DESIGN.md#inference_init)

**MLIR Implementation:**

`inference_init` passes the address of the `__metadata_blob` global and its compile-time size to `hipdnn_ep_state_init_with_fs`. The runtime reads `constants_filename` and `constant_sizes` from the blob via `flatbuffers::GetRoot<UdnaModelMetaInfo>()`.

```mlir
llvm.func @inference_init(%out_state: !llvm.ptr, %fs: !llvm.ptr) -> i32
    attributes {
      llvm.emit_c_interface,
      sym_visibility = "public"
    } {
  // Address of __metadata_blob global (internal linkage, FlatBuffers binary)
  %blob_ptr = llvm.mlir.addressof @__metadata_blob : !llvm.ptr

  // Compile-time blob size (known at pass execution time)
  %blob_size = llvm.mlir.constant(<N> : i64) : i64

  // Runtime reads UdnaModelMetaInfo from blob: constants_filename, constant_sizes
  %result = llvm.call @hipdnn_ep_state_init_with_fs(
      %out_state, %fs, %blob_ptr, %blob_size)
    : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32

  llvm.return %result : i32
}
```

The blob is built at compile time from `hipdnn.*` module attributes by `buildMetadataBlob()` and stored as `__metadata_blob` (internal linkage `i8` array). After LLVM IR merging, `hipdnn_ep_state_init_with_fs` is inlined into the same module — zero call overhead.

**For runtime implementation details:** See [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md).

#### Pool Allocation (Optional - if metadata present)

If `hipdnn.pool_size` metadata exists, `inference_init` allocates memory pool for intermediate buffers.

**Generated code structure:**
1. Check for pool metadata (pool_size, buffer_offsets, buffer_count)
2. Create offsets array on stack
3. Call `hipdnn_ep_pool_init(state, pool_size, offsets, num_buffers)`

**What hipdnn_ep_pool_init does** (runtime implementation):
- Allocates single pool via `hipMalloc(&state->pool_base, pool_size)`
- Stores pool size in `state->pool_size`
- Copies buffer offsets to `state->buffer_offsets` array
- Returns 0 on success, 2 on allocation failure

**Pool metadata source**: [04-MemoryPooling.md](04-MemoryPooling.md) - Graph coloring algorithm computes pool size and offsets at compile-time.

**RuntimeState fields**: See [RUNTIME-ARCHITECTURE.md - RuntimeState with Pool](../../RUNTIME-ARCHITECTURE.md#runtimestate-with-memory-pool) for pool field definitions.

#### Memory Management Strategy: Compile-Time Decision

Pool size and buffer offsets are **hardcoded as LLVM constants** in generated `inference_init()` code.

**Implications:**
- ✅ Zero runtime overhead (no dynamic pool sizing logic)
- ✅ Optimal packing via graph coloring (60% savings for demo model)
- ❌ Cannot change pool strategy without model recompilation
- ❌ Cannot disable pooling at runtime

**Design rationale**: Memory allocation pattern is model-specific and known at compile-time. Hardcoding enables aggressive optimization (LLVM can constant-fold pool calculations).

**Future flexibility**: If runtime pool configuration is needed, GenerateInterfacePass could generate both pooled and non-pooled paths selected via runtime flag.

### Function 2: inference_compute

**C Signature:**
```c
int inference_compute(void* state, span_t* inputs, span_t* outputs);
```

**For design and rationale:** See [../INTERFACE-DESIGN.md - inference_compute](../INTERFACE-DESIGN.md#inference_compute)

**MLIR Implementation:**

**IMPLEMENTATION NOTE (2026-02-12):** The current implementation has been refactored to use runtime helper functions (`hipdnn_ep_tensor_prepare_input`, `hipdnn_ep_tensor_prepare_output`, `hipdnn_ep_tensor_finalize_output`, `hipdnn_ep_tensor_free_input`) which encapsulate parsing, validation, allocation, and H2D/D2H transfers. This reduces generated code size by ~70% while adding validation and better error handling.

**UPDATE (2026-02-28):** The `hipdnn_ep_tensor_prepare_input` and `hipdnn_ep_tensor_prepare_output` helpers now accept a `size_t element_size` parameter (6 params instead of 5). This enables correct buffer sizing for non-float32 types (e.g., float16, int64). Element sizes are embedded as compile-time constants from `hipdnn.input_element_sizes` / `hipdnn.output_element_sizes` module attributes, defaulting to 4 (float32) if absent.

The example below shows the **conceptual flow** of what the helpers do internally. For the actual generated code structure, see `lib/HipDialect/GenerateInterfacePass.cpp:generateInferenceCompute()`.

**NOTE:** This is a simplified example showing the key steps for a single 4D input and single 2D output.
For multiple inputs/outputs, the pass generates code for each tensor using the array-of-pointers pattern.

```mlir
llvm.func @inference_compute(%state: !llvm.ptr,
                              %inputs: !llvm.ptr,   // span_t* (CPU memory)
                              %outputs: !llvm.ptr)  // span_t* (CPU memory)
                              -> i32
    attributes {
      llvm.emit_c_interface,
      sym_visibility = "public"
    } {
  // Define constants
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %c0_i64 = llvm.mlir.constant(0 : i64) : i64
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %null = llvm.mlir.zero : !llvm.ptr

  // These come from module metadata (hipdnn.input_count, hipdnn.input_shapes, etc.)
  %expected_input_count = llvm.mlir.constant(1 : i64) : i64
  %expected_input_rank = llvm.mlir.constant(4 : i32) : i32  // 4D tensor
  %expected_output_count = llvm.mlir.constant(1 : i64) : i64
  %expected_output_rank = llvm.mlir.constant(2 : i32) : i32  // 2D tensor

  // ============================================================================
  // Step 1: Validate input span_t
  // ============================================================================
  // Extract inputs.count (span_t layout: {tensor_t* data, size_t count})
  %inputs_count_ptr = llvm.getelementptr %inputs[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %inputs_count = llvm.load %inputs_count_ptr : !llvm.ptr -> i64

  // Check count matches expected
  %count_ok = llvm.icmp "eq" %inputs_count, %expected_input_count : i64
  llvm.cond_br %count_ok, ^validate_output_count, ^error

^validate_output_count:
  // Extract outputs.count
  %outputs_count_ptr = llvm.getelementptr %outputs[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %outputs_count = llvm.load %outputs_count_ptr : !llvm.ptr -> i64

  // Check count matches expected
  %out_count_ok = llvm.icmp "eq" %outputs_count, %expected_output_count : i64
  llvm.cond_br %out_count_ok, ^validate_ranks, ^error

^validate_ranks:
  // ============================================================================
  // Step 2: Get tensor_t arrays and validate ranks
  // ============================================================================
  // Get inputs.data (tensor_t* array)
  %inputs_data_ptr = llvm.getelementptr %inputs[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %inputs_data = llvm.load %inputs_data_ptr : !llvm.ptr -> !llvm.ptr

  // Get pointer to input tensor 0 (tensor_t layout: {void* data, int64_t* shape, int rank, int data_type})
  %input_0_ptr = llvm.getelementptr %inputs_data[0] : (!llvm.ptr) -> !llvm.ptr

  // Validate rank (field 2 of tensor_t)
  %input_rank_ptr = llvm.getelementptr %input_0_ptr[0, 2] : (!llvm.ptr) -> !llvm.ptr
  %input_rank = llvm.load %input_rank_ptr : !llvm.ptr -> i32
  %rank_ok = llvm.icmp "eq" %input_rank, %expected_input_rank : i32
  llvm.cond_br %rank_ok, ^allocate_gpu_buffers, ^error

^allocate_gpu_buffers:
  // ============================================================================
  // Step 3: Load runtime dimensions and calculate buffer sizes
  // ============================================================================
  // Extract shape pointer (field 1 of tensor_t)
  %input_shape_ptr_ptr = llvm.getelementptr %input_0_ptr[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %input_shape_ptr = llvm.load %input_shape_ptr_ptr : !llvm.ptr -> !llvm.ptr

  // Load RUNTIME dimension values
  %dim0_ptr = llvm.getelementptr %input_shape_ptr[0] : (!llvm.ptr) -> !llvm.ptr
  %dim0 = llvm.load %dim0_ptr : !llvm.ptr -> i64  // Batch size (runtime!)

  %dim1_ptr = llvm.getelementptr %input_shape_ptr[1] : (!llvm.ptr) -> !llvm.ptr
  %dim1 = llvm.load %dim1_ptr : !llvm.ptr -> i64  // Channels (runtime!)

  %dim2_ptr = llvm.getelementptr %input_shape_ptr[2] : (!llvm.ptr) -> !llvm.ptr
  %dim2 = llvm.load %dim2_ptr : !llvm.ptr -> i64  // Height (runtime!)

  %dim3_ptr = llvm.getelementptr %input_shape_ptr[3] : (!llvm.ptr) -> !llvm.ptr
  %dim3 = llvm.load %dim3_ptr : !llvm.ptr -> i64  // Width (runtime!)

  // Calculate total elements and byte size
  %elem_count_01 = llvm.mul %dim0, %dim1 : i64
  %elem_count_012 = llvm.mul %elem_count_01, %dim2 : i64
  %total_elements = llvm.mul %elem_count_012, %dim3 : i64
  %element_size = llvm.mlir.constant(4 : i64) : i64  // sizeof(float) = 4
  %input_byte_size = llvm.mul %total_elements, %element_size : i64

  // Allocate GPU memory for input (H2D transfer target)
  %input_gpu_ptr_ptr = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
  %malloc_ret = llvm.call @wrap_hipMalloc(%input_gpu_ptr_ptr, %input_byte_size)
    : (!llvm.ptr, i64) -> i32

  // Check if allocation succeeded
  %malloc_failed = llvm.icmp "ne" %malloc_ret, %c0_i32 : i32
  llvm.cond_br %malloc_failed, ^error, ^copy_input_h2d

^copy_input_h2d:
  // ============================================================================
  // Step 4: Copy input data from CPU to GPU (H2D)
  // ============================================================================
  %input_gpu_ptr = llvm.load %input_gpu_ptr_ptr : !llvm.ptr -> !llvm.ptr

  // Extract CPU data pointer (field 0 of tensor_t)
  %input_cpu_ptr_ptr = llvm.getelementptr %input_0_ptr[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %input_cpu_ptr = llvm.load %input_cpu_ptr_ptr : !llvm.ptr -> !llvm.ptr

  // Get stream from state via accessor (OPAQUE - no GEP)
  %stream = llvm.call @hipdnn_ep_state_get_stream(%state) : (!llvm.ptr) -> !llvm.ptr

  // wrap_hipMemcpyH2D(dst=GPU, src=CPU, size, stream)
  %memcpy_ret = llvm.call @wrap_hipMemcpyH2D(
    %input_gpu_ptr, %input_cpu_ptr, %input_byte_size, %stream
  ) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32

  %memcpy_failed = llvm.icmp "ne" %memcpy_ret, %c0_i32 : i32
  llvm.cond_br %memcpy_failed, ^error_free_input, ^build_input_memref

^build_input_memref:
  // ============================================================================
  // Step 5: Build memref descriptor for input (pointing to GPU memory)
  // ============================================================================
  // Calculate runtime strides (row-major: stride[i] = product(dims[i+1:]))
  %stride3 = llvm.mlir.constant(1 : i64) : i64
  %stride2 = llvm.mul %dim3, %stride3 : i64
  %stride1 = llvm.mul %dim2, %stride2 : i64
  %stride0 = llvm.mul %dim1, %stride1 : i64

  // Build memref struct (4D tensor)
  %input_gpu_addrspace = llvm.addrspacecast %input_gpu_ptr : !llvm.ptr to !llvm.ptr<1>
  %input_memref = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %input_gpu_addrspace, %input_memref[0]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %input_gpu_addrspace, %input_memref[1]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %c0_i64, %input_memref[2]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim0, %input_memref[3, 0]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim1, %input_memref[3, 1]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim2, %input_memref[3, 2]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %dim3, %input_memref[3, 3]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride0, %input_memref[4, 0]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride1, %input_memref[4, 1]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride2, %input_memref[4, 2]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
  %input_memref = llvm.insertvalue %stride3, %input_memref[4, 3]
    : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>

  // Store in array to pass to @main
  %input_array = llvm.alloca %c1_i64 x !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>
    : (i64) -> !llvm.ptr
  llvm.store %input_memref, %input_array : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4xi64>, array<4xi64>)>, !llvm.ptr

  // ============================================================================
  // Step 6: Allocate GPU buffer for output (similar process)
  // ============================================================================
  // NOTE: Similar code for output tensor (2D in this example)
  // - Get outputs.data array
  // - Load output dimensions from output tensor_t
  // - Allocate GPU memory for output
  // - Build output memref (NO H2D copy - output buffer starts empty)
  // ... (omitted for brevity - same pattern as input)

  %output_array = <similar construction for 2D output>

  // ============================================================================
  // Step 7: Call @main_graph to perform GPU computation
  // ============================================================================
  %main_ret = llvm.call @main_graph(%state, %input_array, %output_array)
    : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // Check if @main_graph succeeded
  %main_failed = llvm.icmp "ne" %main_ret, %c0_i32 : i32
  llvm.cond_br %main_failed, ^error_free_all, ^copy_output_d2h

^copy_output_d2h:
  // ============================================================================
  // Step 8: Copy output data from GPU back to CPU (D2H)
  // ============================================================================
  %output_gpu_ptr = <extract from output_memref>
  %output_cpu_ptr = <extract from output tensor_t>
  %output_byte_size = <calculated from output dimensions>

  // wrap_hipMemcpyD2H(dst=CPU, src=GPU, size, stream)
  %d2h_ret = llvm.call @wrap_hipMemcpyD2H(
    %output_cpu_ptr, %output_gpu_ptr, %output_byte_size, %stream
  ) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32

  %d2h_failed = llvm.icmp "ne" %d2h_ret, %c0_i32 : i32
  llvm.cond_br %d2h_failed, ^error_free_all, ^synchronize

^synchronize:
  // ============================================================================
  // Step 9: Wait for all GPU operations to complete
  // ============================================================================
  %sync_ret = llvm.call @wrap_hipStreamSynchronize(%stream) : (!llvm.ptr) -> i32

  %sync_failed = llvm.icmp "ne" %sync_ret, %c0_i32 : i32
  llvm.cond_br %sync_failed, ^error_free_all, ^cleanup

^cleanup:
  // ============================================================================
  // Step 10: Free temporary GPU buffers and return success
  // ============================================================================
  llvm.call @hipFree(%input_gpu_ptr) : (!llvm.ptr) -> i32
  llvm.call @hipFree(%output_gpu_ptr) : (!llvm.ptr) -> i32

  llvm.return %c0_i32 : i32

// ==============================================================================
// ERROR PATHS
// ==============================================================================

^error_free_all:
  // Computation or D2H transfer failed - free both buffers
  llvm.call @hipFree(%input_gpu_ptr) : (!llvm.ptr) -> i32
  llvm.call @hipFree(%output_gpu_ptr) : (!llvm.ptr) -> i32
  %c8_i32 = llvm.mlir.constant(8 : i32) : i32
  llvm.return %c8_i32 : i32

^error_free_input:
  // H2D copy failed - free input buffer only
  llvm.call @hipFree(%input_gpu_ptr) : (!llvm.ptr) -> i32
  %c9_i32 = llvm.mlir.constant(9 : i32) : i32
  llvm.return %c9_i32 : i32

^error:
  // Validation or allocation failed before any GPU memory allocated
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32  // ERROR_INVALID_INPUT
  llvm.return %c5_i32 : i32
}
```

**Key implementation notes:**
- ✅ Validates input/output counts and ranks
- ✅ Loads **runtime dimensions** from tensor_t.shape pointers
- ✅ Calculates **runtime strides** from dimensions (row-major layout)
- ✅ **Allocates GPU buffers** for inputs and outputs
- ✅ **H2D transfer**: Copies input data from CPU to GPU via hipMemcpyAsync
- ✅ Builds memref structs pointing to **GPU memory** with runtime dimensions
- ✅ Calls @main_graph with memref arrays
- ✅ **D2H transfer**: Copies output data from GPU back to CPU
- ✅ **Synchronizes stream** to ensure transfers complete
- ✅ **Frees temporary GPU buffers** (model weights stay in state, only I/O buffers freed)
- ✅ Proper error handling with cleanup paths

### Function 3: inference_cleanup

**C Signature:**
```c
int inference_cleanup(void* state);
```

**For design and rationale:** See [../INTERFACE-DESIGN.md - inference_cleanup](../INTERFACE-DESIGN.md#inference_cleanup)

**MLIR Implementation:**

**NOTE:** This function is a **simple wrapper** that delegates cleanup to the runtime library (`hipdnn_ep_state_cleanup`). All cleanup logic (stream synchronization, handle destruction in LIFO order, best-effort error handling) is implemented in C++ code (`lib/Runtime/hipdnn_ep_runtime.cpp`), not in generated MLIR.

The pass generates this MLIR:

```mlir
llvm.func @inference_cleanup(%state: !llvm.ptr) -> i32
    attributes {
      llvm.emit_c_interface,
      sym_visibility = "public"
    } {
  // Delegate all cleanup to runtime library
  // This call handles:
  // - Synchronizing GPU stream
  // - Freeing GPU constant memory
  // - Destroying handles in LIFO order (hipBLAS → MIOpen → stream)
  // - Freeing CPU memory (gpu_constants array, RuntimeState struct)
  // - Best-effort cleanup on errors
  %result = llvm.call @hipdnn_ep_state_cleanup(%state) : (!llvm.ptr) -> i32

  llvm.return %result : i32
}
```

**Design rationale:**
- Delegation pattern simplifies MLIR IR generation code
- Complex cleanup logic easier to maintain in C++ than OpBuilder
- Runtime library merged via llvm::Linker - same final binary
- LLVM optimization inlines everything - zero overhead

**What the runtime function does** (see `lib/Runtime/hipdnn_ep_runtime.cpp` for implementation):
1. Synchronizes stream to ensure all GPU work completes
2. Frees all GPU constant memory
3. Destroys handles in LIFO order: hipBLAS → MIOpen → stream (reverse of creation)
4. Frees CPU memory: gpu_constants array, RuntimeState struct
5. Uses best-effort cleanup: continues even if some operations fail
6. Always frees memory to prevent leaks, even on error
7. Returns 0 on success, non-zero on error (with distinct error codes)

**For runtime implementation details:** See [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) and `lib/Runtime/hipdnn_ep_runtime.cpp`.

---

## Verifying C-ABI Compliance

After generating LLVM IR and compiling to DLL, verify exports:

**Windows:**
```bash
dumpbin /EXPORTS inference.dll
# Should show:
#   inference_init
#   inference_compute
#   inference_cleanup
```

**Linux:**
```bash
nm -D inference.so | grep inference
# Should show:
#   T inference_init
#   T inference_compute
#   T inference_cleanup
```

**Calling from C:**
```c
// Load DLL
HMODULE dll = LoadLibrary("inference.dll");  // Windows
// void* dll = dlopen("inference.so", RTLD_NOW);  // Linux

// Get function pointers
typedef int (*inference_init_t)(void**);
inference_init_t init = (inference_init_t)GetProcAddress(dll, "inference_init");

// Call
void* state = NULL;
int ret = init(&state);  // Must work without stack corruption
```

**Common Issues:**
- Missing exports → Symbol not found at runtime
- Wrong calling convention → Stack corruption, crashes
- Name mangling → Can't find symbol (looks for `_Z14inference_initPPv` instead of `inference_init`)

---

## Interface Design for Runtime Shapes

**Status:** Interface designed to accept runtime dimension values, but dynamic shapes not yet implemented.

**Interface design:**
1. User provides tensor with shape via `tensor_t.shape` pointer
2. `inference_compute` loads dimension values from pointer
3. Calculates strides from dimension values
4. Builds memref struct with size/stride arrays
5. Passes to @main

See [DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md) for implementation challenges.

---

## Implementation Strategy

```cpp
void runOnOperation() override {
  ModuleOp module = getOperation();

  // 1. Verify prerequisites
  if (failed(verifyPrerequisites(module))) { signalPassFailure(); return; }

  // 2. Build FlatBuffers blob first — size needed inside inference_init
  std::vector<uint8_t> blob = buildMetadataBlob(module);
  generateMetadataBlobGlobal(module, blob);   // → __metadata_blob global

  // 3. Declare runtime function prototypes
  declareMallocFree(module);
  declareRuntimeFunctions(module);

  // 4. Generate interface functions
  generateInferenceInit(module, blob.size()); // uses AddressOfOp(__metadata_blob)
  generateInferenceCompute(module, ...);
  generateInferenceCleanup(module);

  // 5. Generate JSON metadata (for external tools)
  std::string json = buildMetadataJson(module);
  generateMetadataGlobal(module, json);        // → __metadata_json global
  generateInferenceGetMetadataJson(module);
}
```

---

## Related Documents

**Design:**
- [../INTERFACE-DESIGN.md](../INTERFACE-DESIGN.md) - What and why (design rationale, function contracts)
- [DYNAMIC-SHAPE-DESIGN.md](../../DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape architecture

**Prerequisites:**
- [02-OnnxToHip.md](02-OnnxToHip.md) - Generates constant helpers and metadata
- [05-HipToLLVM.md](05-HipToLLVM.md) - Transforms @main_graph to final signature

**Supporting:**
- [../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md) - Complete lowering flow
- [../../CONSTANT-HANDLING-DESIGN.md](../../CONSTANT-HANDLING-DESIGN.md) - Constant handling details
- [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) - Runtime state structure
