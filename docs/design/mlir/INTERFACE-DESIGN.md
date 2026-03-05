<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# C Interface Design

**Document Type:** Design
**Review Status:** Draft

**Note:** This document describes the **WHAT** and **WHY** of the C interface. For **HOW** to implement it (GenerateInterfacePass details), see [passes/06-GenerateInterfacePass.md](passes/06-GenerateInterfacePass.md).

**Related:** [MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md)

---

## 1. Overview

This document describes the C interface exported from the compiled DLL. The interface provides three functions for model inference:
- `inference_init` - One-time setup (create GPU handles, upload model weights)
- `inference_compute` - Execute inference on input data
- `inference_cleanup` - Release all GPU resources

**Audience:** Users integrating the compiled DLL, designers planning the architecture

**For implementers:** See [passes/06-GenerateInterfacePass.md](passes/06-GenerateInterfacePass.md) for MLIR code generation details.

---

## 2. Architecture

### 2.1 Two-Layer Architecture

The compiled DLL has **two layers of functions**:

**Layer 1: C Interface (Public API)**
- `inference_init(void** out_state [, void* fs])` - Exported from DLL
- `inference_compute(void* state, span_t* inputs, span_t* outputs)` - Exported from DLL
- `inference_cleanup(void* state)` - Exported from DLL

**Layer 2: Internal MLIR Functions (Private)**
- `@main_graph(context, inputs, outputs) -> i32` - Actual computation
- `get_constant_registry() -> ptr` - Returns constant metadata (embedded mode only)

### 2.2 Call Chain (Conceptual)

**High-level flow:**
```
inference_compute (C interface)
    ↓ Parse span_t*, build memref structs
@main_graph (internal wrapper)
    ↓ Load structs, unpack to scalars
@main_internal (computation)
    ↓ Call GPU operations
Computation complete
```

**Why two layers?**

**Problem:** Type system impedance mismatch
- **CustomOp** (C code) uses: `span_t` (array of `tensor_t` structs with dynamic shapes)
- **MLIR** (compiled code) uses: `memref` structs (typed descriptors)

**Solution:** Wrapper functions bridge the gap
- `inference_compute` parses `span_t*` → builds `memref` descriptors → calls `@main_graph`
- `@main_graph` operates on memrefs (natural for MLIR, works with existing passes)

**Note:** For detailed call chain walkthrough showing pack/unpack operations and LLVM optimization, see [passes/06-GenerateInterfacePass.md - Call Chain Walkthrough](passes/06-GenerateInterfacePass.md#call-chain-walkthrough-inference_compute--main--main_internal).

### 2.3 Design Rationale

**Why init/compute/cleanup pattern?**
- Separate one-time setup from hot path (inference loop)
- Reuse GPU resources across multiple inferences
- Clean lifecycle management (acquire resources, use them, release them)

**Why C-ABI compatibility?**
- Cross-language DLL loading (C, C++, C#, Python, etc.)
- No C++ name mangling
- Standard calling conventions (cdecl/sysv)

**Why interface designed for dynamic shapes?**
- Interface accepts runtime dimension values via `tensor_t.shape`
- Rank (number of dimensions) is compile-time known
- Dimension values can be runtime-determined (not yet implemented)
- See [DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) for implementation challenges

---

## 3. C Interface Specification

### 3.1 Function Signatures

#### inference_init

**Signatures (two modes):**

*Embedded constants mode* (compiled with `udna_compile`):
```c
int inference_init(void** out_state);
```

*External constants mode* (compiled with `udna_compile_with_fs`):
```c
int inference_init(void** out_state, void* fs);
```

**Purpose:** One-time setup - create GPU handles and upload model weights

**Responsibilities (embedded mode):**
1. Allocate execution context (HipExecutionContext struct)
2. Create GPU stream (hipStream_t)
3. Create MIOpen handle and associate with stream
4. Create hipBLAS handle
5. Get constant registry (`get_constant_registry()`)
6. Allocate array for GPU constant pointers
7. Upload model weights/constants to GPU (loop: hipMalloc + hipMemcpy from `.data` section)

**Responsibilities (external constants mode):**
1–4. Same GPU handle creation as embedded mode
5. Open `constants.bin` via `fs` (`morphizen::FileSystem*` cast from `void*`)
6. Read each constant sequentially (sizes hardcoded in generated code)
7. Upload to GPU (loop: hipMalloc + hipMemcpy from FileSystem reader)

**Note:** For detailed constant management design, see [../CONSTANT-HANDLING-DESIGN.md](../CONSTANT-HANDLING-DESIGN.md).

**Parameters:**
- `out_state` (output): Pointer to receive allocated context pointer
- `fs` (input, external mode only): `morphizen::FileSystem*` as `void*`; used to open `constants.bin`

**Return value:** Status code
- 0: Success - context created and weights uploaded
- 1: Context allocation failed (malloc failed)
- 2: GPU handle creation failed (stream, MIOpen, or hipBLAS)
- 3: Constant initialization failed (upload to GPU failed)

**Usage (embedded mode):**
```c
void* state = NULL;
int ret = inference_init(&state);
if (ret != 0) {
    // Handle error
}
```

**Usage (external constants mode):**
```c
morphizen::FileSystem* fs = ...; // caller provides
void* state = NULL;
int ret = inference_init(&state, fs);
if (ret != 0) {
    // Handle error
}
```

**Implementation details:** See [passes/06-GenerateInterfacePass.md - Function 1: inference_init](passes/06-GenerateInterfacePass.md#function-1-inference_init)

#### inference_compute

**Signature:**
```c
int inference_compute(void* state, span_t* inputs, span_t* outputs);
```

**Purpose:** Execute inference on input tensors

**Responsibilities:**
1. Validate input/output counts and tensor ranks
2. Load runtime dimensions from `tensor_t.shape` pointers (dynamic shapes!)
3. Copy input data from CPU to GPU (H2D transfer)
4. Call `@main_graph` to perform computation
5. Copy output data from GPU back to CPU (D2H transfer)
6. Synchronize GPU stream to ensure completion

**Note:** GPU buffer allocation strategy is an implementation detail - see [../MEMORY-MANAGEMENT.md](../MEMORY-MANAGEMENT.md) for optimization strategies (inline allocation, allocation hoisting, memory pooling).

**Parameters:**
- `state` (input): Context pointer from `inference_init`
- `inputs` (input): Pointer to span_t containing input tensors (CPU memory)
- `outputs` (input/output): Pointer to span_t containing output tensors (CPU memory)

**Return value:** Status code
- 0: Success - inference completed
- 5: Invalid input (wrong count, rank, or null pointers)
- 8: Computation failed (@main_graph returned error or GPU operation failed)
- 9: Memory transfer failed (H2D or D2H)

**Usage:**
```c
// Prepare input tensor
int64_t input_shape[] = {1, 3, 224, 224};
tensor_t input = {
    .data = input_data,
    .shape = input_shape,
    .rank = 4,
    .data_type = 0  // FLOAT32
};
span_t inputs = { .data = &input, .count = 1 };

// Prepare output tensor
int64_t output_shape[] = {1, 1000};
tensor_t output = {
    .data = output_data,
    .shape = output_shape,
    .rank = 2,
    .data_type = 0  // FLOAT32
};
span_t outputs = { .data = &output, .count = 1 };

// Run inference
int ret = inference_compute(state, &inputs, &outputs);
if (ret != 0) {
    // Handle error
}
```

**Implementation details:** See [passes/06-GenerateInterfacePass.md - Function 2: inference_compute](passes/06-GenerateInterfacePass.md#function-2-inference_compute)

#### inference_cleanup

**Signature:**
```c
int inference_cleanup(void* state);
```

**Purpose:** Release all GPU resources

**Responsibilities:**
1. Synchronize GPU stream (wait for pending operations)
2. Free GPU constant memory (loop: hipFree for each constant)
3. Destroy GPU handles (hipBLAS, MIOpen, stream) in reverse creation order
4. Free constant pointer array
5. Free execution context struct

**Parameters:**
- `state` (input): Context pointer from `inference_init`

**Return value:** Status code
- 0: Success - all resources released cleanly
- 10: Stream destruction failed
- 11: MIOpen destruction failed
- 12: hipBLAS destruction failed
- 13: Constant release failed (potential GPU memory leak)
- 14: Stream synchronization failed (GPU may still be working)

**Note:** Even on error, the function attempts best-effort cleanup to prevent memory leaks. CPU memory is always freed.

**Usage:**
```c
int ret = inference_cleanup(state);
if (ret != 0) {
    // Log error, but context is freed
}
state = NULL;  // Mark as invalid
```

**Implementation details:** See [passes/06-GenerateInterfacePass.md - Function 3: inference_cleanup](passes/06-GenerateInterfacePass.md#function-3-inference_cleanup)

### 3.2 Data Structures

#### tensor_t

**Definition:**
```c
typedef struct {
    void* data;        // Pointer to tensor data (CPU or GPU memory)
    int64_t* shape;    // Pointer to shape array (runtime dimension values)
    int rank;          // Number of dimensions (compile-time known rank)
    int data_type;     // Element type enumeration
} tensor_t;
```

**Fields:**
- `data`: Pointer to contiguous tensor data
  - For inputs: CPU memory (user-provided)
  - For outputs: CPU memory (user-allocated, function fills)
- `shape`: Pointer to array of dimension sizes
  - Example: For 4D tensor with shape [2, 3, 224, 224], shape[0]=2, shape[1]=3, etc.
  - Runtime values (interface supports dynamic shapes, not yet implemented)
- `rank`: Number of dimensions (must match model's expected rank)
  - Example: 4 for NCHW image tensor, 2 for fully connected output
- `data_type`: Element type (see table below)

**Example:**
```c
// Input: batch=2, channels=3, height=224, width=224
int64_t input_shape[] = {2, 3, 224, 224};
float* input_data = malloc(2 * 3 * 224 * 224 * sizeof(float));

tensor_t input = {
    .data = input_data,
    .shape = input_shape,
    .rank = 4,
    .data_type = 0  // FLOAT32
};
```

#### span_t

**Definition:**
```c
typedef struct {
    tensor_t* data;    // Pointer to array of tensor_t
    size_t count;      // Number of tensors in array
} span_t;
```

**Fields:**
- `data`: Pointer to array of tensor_t structures
- `count`: Number of tensors in the array

**Purpose:** Support models with multiple inputs and/or outputs

**Example:**
```c
// Model with 2 inputs (image and metadata)
tensor_t inputs[2] = { ... };
span_t input_span = {
    .data = inputs,
    .count = 2
};
```

#### data_type Enumeration

**Element types:**

| Value | Type    | Element Size | Description              |
|-------|---------|--------------|--------------------------|
| 0     | FLOAT32 | 4 bytes      | 32-bit floating point    |
| 1     | FLOAT16 | 2 bytes      | 16-bit floating point    |
| 2     | INT8    | 1 byte       | 8-bit signed integer     |

**Note:** Current implementation focuses on FLOAT32. Additional types may be added in the future.

### 3.3 Error Codes (Consolidated)

**Complete error code table:**

| Code | Name                          | Function           | Description                          |
|------|-------------------------------|--------------------|--------------------------------------|
| 0    | SUCCESS                       | All                | Operation completed successfully     |
| 1    | ERROR_ALLOCATION              | inference_init     | Context allocation failed (malloc)   |
| 2    | ERROR_HANDLE_CREATION         | inference_init     | GPU handle creation failed           |
| 3    | ERROR_CONSTANT_INIT           | inference_init     | Constant upload to GPU failed        |
| 5    | ERROR_INVALID_INPUT           | inference_compute  | Invalid span_t/tensor_t data         |
| 8    | ERROR_COMPUTATION             | inference_compute  | GPU computation failed               |
| 9    | ERROR_MEMORY_TRANSFER         | inference_compute  | H2D or D2H copy failed               |
| 10   | ERROR_STREAM_DESTROY          | inference_cleanup  | Stream destruction failed            |
| 11   | ERROR_MIOPEN_DESTROY          | inference_cleanup  | MIOpen destruction failed            |
| 12   | ERROR_HIPBLAS_DESTROY         | inference_cleanup  | hipBLAS destruction failed           |
| 13   | ERROR_CONSTANT_RELEASE        | inference_cleanup  | GPU constant free failed             |
| 14   | ERROR_STREAM_SYNC             | inference_cleanup  | Stream synchronization failed        |

**Error handling policy:**
- Fail fast: Return error immediately, don't continue
- Cleanup on error: Free all resources allocated before error
- Propagate errors: Pass through error codes from internal functions
- No exceptions: Pure C ABI, use integer error codes
- Best-effort cleanup: Even if cleanup fails, free CPU memory to prevent leaks

---

## 4. Design Decisions

### 4.1 Why init/compute/cleanup Pattern?

**Problem:** GPU setup is expensive (handle creation, constant upload)

**Solution:** Separate one-time setup from hot path

**Benefits:**
- Initialization cost amortized across many inferences
- Reuse GPU handles and constant memory
- Clear resource lifecycle (acquire → use → release)

**Alternative considered:** Single `inference(inputs, outputs)` function
- **Rejected:** Would recreate handles and upload constants on every inference
- **Performance impact:** 10-100ms overhead per inference vs. 0.1-1ms for compute-only

### 4.2 Why span_t/tensor_t Structures?

**Problem:** Need to support models with variable numbers of inputs/outputs

**Solution:** Array-based interface with dynamic counts

**Benefits:**
- Single interface supports all model topologies
- No recompilation for different I/O configurations
- Dynamic shape support (runtime dimension values)
- Simple C-compatible structures (no vtables, no name mangling)

**Alternative considered:** Fixed arrays `inference_compute(input1, input2, ..., output1, output2, ...)`
- **Rejected:** Requires different signature for each model topology
- **Scalability problem:** Cannot compile generic DLL loader

### 4.3 Why Interface Designed for Dynamic Shapes?

**Problem:** Production models need batch size flexibility

**Interface design:** Runtime dimension values via `tensor_t.shape` pointer

**Design invariants:**
- Tensor **rank** is compile-time known (e.g., always 4D for images)
- Dimension **values** can be runtime-determined (interface supports it)
- Same C API for static and dynamic

**Implementation status:** Dynamic shapes not yet supported due to memory pooling incompatibility.

**See:** [DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) for implementation challenges and potential solutions

### 4.4 Why C-ABI Compatibility?

**Problem:** Need to load DLL from multiple languages

**Solution:** C calling convention and no name mangling

**Benefits:**
- Cross-language compatibility (C, C++, C#, Python, Rust, etc.)
- Standard DLL export format (Windows: dllexport, Linux: visibility=public)
- Predictable stack layout and calling conventions

**Requirements:**
- All exported functions use `extern "C"` semantics
- No C++ features in interface (no classes, templates, exceptions)
- Standard types only (void*, int, struct)

---

## 5. Related Documents

**Core design:**
- [MLIR-COMPILATION-OVERVIEW.md](../MLIR-COMPILATION-OVERVIEW.md) - Overall compilation pipeline
- [DYNAMIC-SHAPE-DESIGN.md](../DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape support details
- [RUNTIME-ARCHITECTURE.md](../RUNTIME-ARCHITECTURE.md) - Runtime state structure

**Implementation:**
- [passes/06-GenerateInterfacePass.md](passes/06-GenerateInterfacePass.md) - How to implement the C interface in MLIR
- [passes/02-OnnxToHip.md](passes/02-OnnxToHip.md) - Generates constant helpers and metadata
- [passes/05-HipToLLVM.md](passes/05-HipToLLVM.md) - Transforms @main_graph signature

**Supporting details:**
- [LOWERING-PIPELINE.md](LOWERING-PIPELINE.md) - Complete lowering flow
- [../CONSTANT-HANDLING-DESIGN.md](../CONSTANT-HANDLING-DESIGN.md) - Constant handling
