<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Runtime Architecture

**Date:** 2026-02-12
**Document Type:** Design Document
**Review Status:** Self-Reviewed
**Branch:** `mlir-integration`
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md), [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md)

---

## 1. Core Design: Opaque RuntimeState Pattern

### Problem

MLIR-generated code for compiled models needs to:
- Manage persistent GPU resources (streams, library handles)
- Access pre-uploaded model weights (constants on GPU)
- Call GPU operations (convolution, GEMM) without coupling to runtime implementation details

**Constraint**: Runtime must be able to evolve (add libraries, optimize layout) without breaking already-compiled models.

### Solution: Opaque Handle with Accessor Functions

Generated code sees only `void* state` (opaque pointer). Runtime owns the internal structure.

**Abstraction Boundary:**

```
┌─────────────────────────────────────────────────────────┐
│  Generated Code (LLVM IR)                                │
│  - Sees: void* state (opaque pointer)                   │
│  - Calls: hipdnn_ep_get_stream(state)                   │
│  - Calls: hipdnn_ep_get_constant(state, index)          │
│  - Cannot: Access internal fields directly              │
└─────────────────────────────────────────────────────────┘
                         ↕ (Abstraction Boundary)
┌─────────────────────────────────────────────────────────┐
│  Runtime Implementation (C++)                            │
│  - Owns: struct RuntimeState {                          │
│           hipStream_t stream;                            │
│           miopenHandle_t miopen_handle;                  │
│           hipblasLtHandle_t hipblas_handle;              │
│           void** gpu_constants;                          │
│           size_t num_constants;                          │
│         }                                                │
│  - Provides: Accessor functions                         │
│  - Can: Evolve internal layout freely                   │
└─────────────────────────────────────────────────────────┘
```

**Correct Pattern (Opaque Access):**
```mlir
%stream = llvm.call @hipdnn_ep_get_stream(%state) : (!llvm.ptr) -> !llvm.ptr
%constant = llvm.call @hipdnn_ep_get_constant(%state, %idx) : (!llvm.ptr, i64) -> !llvm.ptr
```

**Forbidden Pattern (Direct Access):**
```mlir
%stream_ptr = llvm.getelementptr %state[0, 0] : (!llvm.ptr) -> !llvm.ptr
```

### Design Implications

**Extensibility**: Can add GPU libraries without breaking generated code
- Example: Adding `rocfft_plan fft_plan;` to RuntimeState requires:
  - ✅ No changes to generated code (still uses `void* state`)
  - ✅ No recompilation of existing models
  - ✅ No changes to C interface

**Evolution**: Can optimize internal layout independently
- Reorder fields for cache efficiency
- Change memory allocation strategy
- Add descriptor caches

**Self-Contained Models**: Each compiled model.dll is independent
- No shared runtime state across models
- No version conflicts

---

## 2. Design Decision: Runtime as Embedded Bitcode

### Primary Rationale: Self-Contained Compiler (MUST HAVE)

**Problem**: Model compilation requires runtime functions (stream access, constant registry, etc.)

**Traditional approach**: Link generated code against external runtime library (.lib/.a file)

**Issues with traditional approach:**
- Compiler must locate/ship runtime library files
- Version management complexity (which runtime version to use?)
- Deployment overhead (multiple files to distribute)
- Risk of version conflicts (user's runtime vs. compiler's runtime)

**Our solution**: Embed runtime as LLVM bitcode in EP DLL

**How it works:**
1. Runtime compiled to bitcode (runtime.bc) at EP build time
2. Bitcode embedded as binary resource in EP DLL
3. During model compilation, bitcode extracted and merged via `llvm::Linker::linkInModule()`
4. No external files required - everything self-contained

**Benefits:**
- **Self-contained compiler**: Everything needed in single EP DLL
- **No external dependencies**: Model compilation works out-of-the-box
- **Version consistency**: Runtime bitcode frozen at EP build time
- **Simple deployment**: One EP DLL contains compiler + runtime

This is the **primary architectural requirement** - without embedded bitcode, the compiler would need external runtime files.

### Secondary Benefit: Zero-Cost Abstraction (NICE TO HAVE)

Additionally, IR merging provides a performance benefit.

**The concern**: Opaque access requires function calls like `hipdnn_ep_get_stream(state)` instead of direct field access. Naive expectation would be function call overhead on every GPU operation (5-10 cycles per call, register spills, ABI overhead).

**The solution**: LLVM optimization fully inlines accessor functions during model compilation.

**Before optimization (after IR merging):**
```llvm
%stream = call ptr @hipdnn_ep_get_stream(ptr %state)
```

**After LLVM O2 inlining:**
```llvm
%stream_ptr = getelementptr inbounds %struct.RuntimeState, ptr %state, i32 0, i32 0
%stream = load ptr, ptr %stream_ptr
```

**Result**: Accessor function disappeared - transformed to direct memory load.

### Verification

Final model.dll should NOT contain runtime function symbols:
```bash
grep "hipdnn_ep_get_stream" optimized.ll  # Should be empty
```

All accessor calls are inlined to load instructions - zero runtime overhead.

### Design Trade-off

**Why not just use direct field access?**
- Generated code would couple to RuntimeState layout
- Adding fields would break existing models (ABI breakage)
- Cannot evolve runtime independently

**IR merging gives us both:**
- Clean abstraction at source level (maintainability)
- Direct access at binary level (performance)

For architectural rationale, see [ARCHITECTURE.md Design Decision #7](ARCHITECTURE.md#7-llvm-ir-merging-for-zero-cost-runtime-abstraction).

---

## 3. Integration Architecture

This diagram shows the complete pipeline from Runtime development to model inference:

```
┌──────────────────────────────────────────────────────────────────┐
│  BUILD TIME (Once - when building EP DLL)                        │
│  Produces: Embedded Runtime bitcode                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  hipdnn_ep_runtime.cpp + hipdnn_ep_runtime.h                     │
│         ↓                                                         │
│  clang -c -emit-llvm -O2 -std=c++17                              │
│         ↓                                                         │
│  runtime.bc (LLVM bitcode)                                       │
│         ↓                                                         │
│  xxd.py --var runtime_bc_data                                    │
│         ↓                                                         │
│  runtime_ir_data.cpp (embedded as unsigned char array)           │
│         ↓                                                         │
│  Compiled into EP DLL (contains embedded bitcode)                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  MODEL COMPILATION TIME (Per model - Level-1 Pass)               │
│  Input: ONNX model    Output: EPContext with embedded DLL        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [From ARCHITECTURE.md: ONNX → MLIR transformations]             │
│         ↓                                                         │
│  MLIR (HIP dialect) → LLVM dialect conversion                    │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ LLVM IR Generation                                 │          │
│  │  - @main(state, inputs, outputs) function          │          │
│  │  - @inference_init/compute/cleanup wrappers        │          │
│  │  - Calls to @hipdnn_ep_get_stream, etc.            │          │
│  │  - llvm.mlir.global for embedded constants         │          │
│  │  - @get_constant_registry for metadata             │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ Runtime IR Merging (llvm::Linker API)              │          │
│  │  1. Parse embedded runtime_bc_data → Runtime IR    │          │
│  │  2. llvm::Linker::linkInModule(Runtime IR)         │          │
│  │  3. Resolve function declarations:                 │          │
│  │     - @hipdnn_ep_state_init(ptr, ptr) -> i32       │          │
│  │     - @hipdnn_ep_get_stream(ptr) -> ptr            │          │
│  │     - @hipdnn_ep_get_constant(ptr, i64) -> ptr     │          │
│  │     - @wrap_miopenConvolutionForward(...)          │          │
│  │     - @wrap_hipblasLtGemm(...)                     │          │
│  │     - @wrap_hipMalloc/Free/MemcpyH2D/D2H/Sync     │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  Combined LLVM IR Module                                         │
│    (Generated code + Runtime implementation merged)              │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ LLVM Optimization (PassBuilder O2)                 │          │
│  │  Key transformations:                              │          │
│  │  - Inline @hipdnn_ep_get_stream() → load instr     │          │
│  │  - Inline @hipdnn_ep_get_constant() → array access │          │
│  │  - Dead code elimination (unused Runtime code)     │          │
│  │  - Cross-module inlining opportunities             │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  Optimized LLVM IR (accessor calls eliminated)                   │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ Native Code Generation                             │          │
│  │  - LLVM IR → Object code (.obj/.o)                 │          │
│  │  - Link ROCm libraries:                            │          │
│  │    * amdhip64.lib (HIP runtime)                    │          │
│  │    * MIOpen.lib (DNN operations)                   │          │
│  │    * hipblaslt.lib (BLAS operations)               │          │
│  │  - Produce: model.dll                              │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  model.dll (native x64 DLL)                                      │
│    Exports:                                                       │
│      - inference_init(void** out_state) -> i32                   │
│      - inference_compute(void* state, span_t*, span_t*) -> i32   │
│      - inference_cleanup(void* state) -> i32                     │
│    Contains:                                                      │
│      - Inlined Runtime code (no function call overhead)          │
│      - Embedded constants (weights/biases in .data section)      │
│      - GPU operation calls (MIOpen, hipBLAS)                     │
│         ↓                                                         │
│  ┌────────────────────────────────────────────────────┐          │
│  │ EPContext Packaging                                │          │
│  │  1. Read model.dll into memory buffer              │          │
│  │  2. Create EPContext node in ONNX graph            │          │
│  │  3. Embed DLL as binary attribute                  │          │
│  │  4. Replace original graph with EPContext node     │          │
│  │  5. Save as model_with_context.onnx                │          │
│  └────────────────────────────────────────────────────┘          │
│         ↓                                                         │
│  model_with_context.onnx (single file deployment)                │
│    Contains:                                                      │
│      - EPContext node (com.microsoft:EPContext)                  │
│      - Embedded model.dll (pre-compiled, optimized)              │
│      - No ONNX graph (replaced by compiled artifact)             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  INFERENCE TIME (Runtime - repeatedly executed)                  │
│  Dependencies: HIP runtime, MIOpen, hipBLASLt                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Load model_with_context.onnx                                    │
│         ↓                                                         │
│  CustomOp recognizes EPContext node                              │
│         ↓                                                         │
│  Extract embedded model.dll from EPContext attribute             │
│         ↓                                                         │
│  Load DLL from memory (MemoryModule or platform API)             │
│         ↓                                                         │
│  Resolve function pointers:                                      │
│    - fn_init = GetProcAddress("inference_init")                  │
│    - fn_compute = GetProcAddress("inference_compute")            │
│    - fn_cleanup = GetProcAddress("inference_cleanup")            │
│         ↓                                                         │
│  Session Initialization:                                         │
│    void* state = nullptr;                                        │
│    fn_init(&state);  // Creates GPU handles, allocates constants │
│         ↓                                                         │
│  Inference Execution (repeated):                                 │
│    fn_compute(state, inputs, outputs);  // Reuses GPU resources  │
│         ↓                                                         │
│  Session Cleanup:                                                │
│    fn_cleanup(state);  // Frees GPU resources                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Key Transformation Points

1. **Build Time**: Runtime compiled to bitcode once, embedded in EP DLL
2. **Model Compilation**: Runtime IR merged with generated IR via llvm::Linker
3. **Optimization**: LLVM inlines accessor functions (zero-cost abstraction achieved)
4. **Packaging**: Final DLL embedded in EPContext for single-file deployment
5. **Inference**: CustomOp loads DLL from memory, calls exported functions

### Pool Allocation Integration

Memory pooling is optional. If MemoryPoolingPass runs during compilation, the flow includes pool allocation:

```
┌─────────────────────────────────────────────────────────────┐
│  inference_init (generated code)                            │
│    ↓                                                         │
│  Call hipdnn_ep_state_init(out_state, registry)             │
│    ↓ (creates stream, handles, uploads constants)           │
│  IF pool metadata present:                                  │
│    Call hipdnn_ep_pool_init(state, pool_size, offsets, n)   │
│    ↓ (allocates pool, stores offsets in RuntimeState)       │
│  Return 0 (success)                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  inference_compute (generated code)                         │
│    ↓                                                         │
│  @main calls hip.alloc (lowered by HipToLLVM):              │
│    buffer_ptr = hipdnn_ep_get_buffer_from_pool(state, idx)  │
│    ↓ (returns pool_base + buffer_offsets[idx])              │
│  Use buffer_ptr for computation                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  inference_cleanup (generated code)                         │
│    ↓                                                         │
│  Call hipdnn_ep_state_cleanup(state)                        │
│    ↓ (frees pool via hipFree(state->pool_base))             │
│  Return 0 (success)                                         │
└─────────────────────────────────────────────────────────────┘
```

**Pool metadata**: [04-MemoryPooling.md](../mlir/passes/04-MemoryPooling.md) computes pool size and offsets using graph coloring.

**Generated code**: [GenerateInterfacePass.md - Pool Allocation](../mlir/passes/GenerateInterfacePass.md#pool-allocation) shows how pool_init call is generated.

---

## 4. API Contract

### Three-Function Lifecycle

The compiled DLL exports exactly three C-ABI functions. For complete specification including data structures, error codes, and usage examples, see [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md).

**`inference_init(void** out_state) -> i32`**
- Creates GPU handles (stream, MIOpen, hipBLAS)
- Allocates RuntimeState structure
- Uploads model constants to GPU
- Returns opaque state pointer
- One-time cost amortized across many inferences

**`inference_compute(void* state, span_t* inputs, span_t* outputs) -> i32`**
- Validates input/output tensors
- Allocates temporary GPU buffers
- Copies data H2D, executes @main, copies results D2H
- Synchronizes GPU stream
- Frees temporary buffers (constants remain on GPU)
- Reuses GPU handles and constant memory

**`inference_cleanup(void* state) -> i32`**
- Synchronizes GPU stream
- Frees GPU constant memory
- Destroys GPU handles in LIFO order
- Frees RuntimeState structure

### Design Rationale

For detailed design decisions including:
- Why init/compute/cleanup pattern? (with alternatives and performance analysis)
- Why span_t/tensor_t structures? (scalability and flexibility)
- Why dynamic shape support? (deployment flexibility)
- Why C-ABI compatibility? (cross-language integration)

See [mlir/INTERFACE-DESIGN.md - Section 4: Design Decisions](mlir/INTERFACE-DESIGN.md#4-design-decisions).

**Runtime-specific considerations:**
- **Opaque state pointer**: See [Section 1: Core Design](#1-core-design-opaque-runtimestate-pattern)
- **GPU handle reuse**: Handles and constants persist across inferences (init once, compute many, cleanup once)
- **Resource lifecycle**: Clear separation of one-time costs (init/cleanup) from hot path (compute)

---

## 5. RuntimeState Design

### Internal Structure

```cpp
struct RuntimeState {
  // GPU execution context
  hipStream_t stream;                    // GPU execution stream
  miopenHandle_t miopen_handle;          // MIOpen library handle for DNN ops
  hipblasLtHandle_t hipblas_handle;      // hipBLAS handle for GEMM

  // Model constants (weights, biases)
  void** gpu_constants;                  // Array of GPU pointers to constants
  size_t num_constants;                  // Array size (compile-time known)

  // Memory pool for intermediate buffers (optional - if pooling enabled)
  void* pool_base;                       // Base pointer to single allocated pool (GPU memory)
  size_t pool_size;                      // Total pool size in bytes
  size_t* buffer_offsets;                // Array of buffer offsets within pool
  size_t num_buffers;                    // Number of buffers in pool
};
```

**Pool allocation**: If model was compiled with memory pooling (MemoryPoolingPass), `inference_init` allocates single pool via `hipdnn_ep_pool_init`. Individual buffers retrieved via `hipdnn_ep_get_buffer_from_pool(state, index)` which returns `pool_base + buffer_offsets[index]`.

**Memory savings**: Demo model achieves 60% reduction (12.8MB vs 32.1MB) using graph coloring algorithm. See [04-MemoryPooling.md](../mlir/passes/04-MemoryPooling.md) for algorithm details.

### Design Properties

**Allocation**: Heap-allocated in `inference_init`, freed in `inference_cleanup`

**Lifetime**: Tied to inference session
- Created once per session
- Reused across multiple inference calls
- Destroyed when session ends

**Thread Safety**: NOT thread-safe
- One inference per state at a time
- Multiple states can run concurrently (different sessions)

**Size**: ~40 bytes + constant array
- Fixed overhead is minimal
- Constant array size known at model compile time

### Extensibility

Can add fields without breaking generated code:

```cpp
struct RuntimeState {
    // Existing fields...
    hipStream_t stream;
    miopenHandle_t miopen_handle;
    hipblasLtHandle_t hipblas_handle;
    void** gpu_constants;
    size_t num_constants;

    // NEW: Add library handles as needed
    rocfft_plan fft_plan;              // For FFT operations
    rocblas_handle blas_handle;        // For additional BLAS ops
    // ... more fields
};
```

**No impact on:**
- ✅ Generated code (uses opaque `void* state`)
- ✅ C interface (still `void*`)
- ✅ Existing compiled models (accessor functions still work)

### Potential Extensions

The RuntimeState design can accommodate:
- **Library handles**: rocBLAS, rocFFT, rocRAND, rocSPARSE, rocSOLVER, RCCL
- **Performance optimizations**: Descriptor caches, workspace pools, algorithm selection caches
- **Resource management**: Memory pools, buffer reuse strategies

Each addition is isolated to runtime implementation. Opaque design ensures zero impact on generated code.

---

## 6. Deployment Model

### Single-File Deployment

Each model is packaged as a single `.onnx` file containing:
- EPContext node (com.microsoft:EPContext)
- Embedded model.dll (pre-compiled, fully optimized)
- No ONNX graph (replaced by compiled artifact)

**Deployment**: Copy single file, no build tools needed at inference time.

### Runtime Distribution

**Runtime is NOT a separate library**. The Runtime code is:
- Merged with generated IR at model compilation time
- Fully inlined during LLVM optimization
- Embedded in final model.dll as inline code

**Result**: model.dll has no runtime dependencies beyond ROCm libraries.

### External Dependencies

Target system must have ROCm installed with:
- **amdhip64.dll** - HIP runtime (GPU execution)
- **MIOpen.dll** - DNN operations (convolution, pooling, etc.)
- **hipblaslt.dll** - BLAS operations (matrix multiplication)

These are standard ROCm components, not project-specific.

### No Static Library Linking

Common misconception: "Runtime is a static library linked into each model.dll"

**Reality**: Runtime is LLVM bitcode that is:
1. Merged at IR level (not linked at binary level)
2. Fully inlined during optimization (function calls eliminated)
3. Dead code eliminated (unused runtime functions removed)

Final model.dll contains runtime code as **inline instructions**, not function calls.

---

## Related Documents

**Architecture & Design:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture, design decisions
- [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) - Compilation pipeline overview

**Interface Specification:**
- [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) - Complete C interface specification (tensor_t, span_t, error codes, usage examples)

**Supporting Design:**
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - Constant extraction, upload, and management
- [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape support design

**Implementation Details** (for developers, not design review):
- lib/Runtime/hipdnn_ep_runtime.cpp - Runtime implementation
- lib/Backend/LLVMBackend.cpp - IR merging implementation
- CMakeLists.txt (lib/Runtime) - Build configuration
