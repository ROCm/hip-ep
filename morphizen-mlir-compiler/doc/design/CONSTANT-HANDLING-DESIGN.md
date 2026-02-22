<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Constant Handling Design

**Date:** 2026-02-13
**Document Type:** Design
**Review Status:** Self-Reviewed
**Related:** ARCHITECTURE.md, RUNTIME-ARCHITECTURE.md, mlir/passes/06-GenerateInterfacePass.md

---

## Table of Contents

- [Overview](#overview)
- [The Constant Registry Pattern](#the-constant-registry-pattern)
- [Separation of Concerns](#separation-of-concerns)
- [Design Decisions](#design-decisions)
- [Open Questions](#open-questions)
- [Related Documents](#related-documents)

---

## Overview

ONNX models contain hundreds of constant tensors: weights for 100+ convolutional layers, biases, batch normalization parameters (scale/bias/mean/variance), and embedding lookup tables. In ONNX-MLIR, these appear as `onnx.Constant` operations within function bodies.

**Problem:**

1. **Function signature explosion**: Passing constants as function arguments leads to unmaintainable signatures. A ResNet50 model with 200 constants would require `@main_graph(%input, %output, %w0, %w1, ..., %w199)` - 202 arguments. Adding one constant requires changing the function signature everywhere.

2. **Repeated GPU uploads**: Without persistent storage, constants must be uploaded to GPU on every inference. For a 100MB model, this adds 50-100ms overhead per inference (hipMalloc + hipMemcpy are expensive).

3. **Code coupling**: Hard-coded GPU memory management in generated code prevents runtime optimizations without model recompilation.

**Solution:** Constant Registry Pattern - generated code provides metadata, runtime bitcode handles GPU lifecycle. Both merge into single optimized DLL.

**Result:**

1. **Scalable signatures**: Function signature stays `@main_graph(%ctx, %inputs, %outputs)` regardless of constant count
2. **One-time upload**: GPU memory allocated and uploaded once during init (not per-inference)
3. **Runtime flexibility**: Memory management strategies can evolve without changing generated code
4. **Clean separation**: Model-specific knowledge (generated) ↔ GPU management (runtime bitcode)

---

## The Constant Registry Pattern

### Core Concept

Separate model-specific knowledge (what constants, what data) from model-agnostic GPU management (how to upload, how to manage memory) via a metadata interface.

```
┌─────────────────────────────────────────────────────────────┐
│ Generated Code (model-specific)                             │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ LLVM Globals (in .data section)                    │     │
│  │  constant_0: [64x3x3x3 float32 array]              │     │
│  │  constant_1: [64 float32 array]                    │     │
│  │  ...                                               │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │ ConstantRegistry (metadata)                        │     │
│  │  ConstantInfo[0]: {ptr: &constant_0, size: ...}    │     │
│  │  ConstantInfo[1]: {ptr: &constant_1, size: ...}    │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │ get_constant_registry() -> ptr                     │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │ @main_graph() calls:                                     │     │
│  │   %w = hip.get_constant(%ctx, 0)                   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  Knows: What constants exist, their indices, data          │
│  Does NOT: How to manage GPU memory                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   llvm::Linker merges ↓
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Runtime Bitcode (model-agnostic, pre-compiled)              │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ hipdnn_ep_state_init(out_state, registry)          │     │
│  │   Read metadata from registry                      │     │
│  │   Allocate gpu_constants[] array                   │     │
│  │   For each: hipMalloc + hipMemcpy (H2D)            │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │ hipdnn_ep_constant_get(state, index)               │     │
│  │   return state->gpu_constants[index]               │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │ hipdnn_ep_state_cleanup(state)                     │     │
│  │   For each: hipFree(gpu_constants[i])              │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  Knows: GPU APIs, memory strategies, RuntimeState          │
│  Does NOT: What constants exist in this model              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   LLVM Optimization ↓
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Final DLL (merged + optimized)                              │
│  - Constant data in .data section                           │
│  - Metadata (ConstantRegistry)                              │
│  - GPU management code (inlined from runtime bitcode)       │
│  - Zero overhead (accessor calls inlined to array access)   │
│                                                              │
│  Single binary, self-contained                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Structures

**ConstantInfo** (metadata for one constant):
```c
struct ConstantInfo {
    const void* cpu_data;      // Pointer to LLVM global in DLL .data section
    size_t size_bytes;         // Total size in bytes
    size_t element_size;       // sizeof(float), sizeof(int8_t), etc.
    size_t num_elements;       // For validation/debugging
};
```

**ConstantRegistry** (metadata for all constants):
```c
struct ConstantRegistry {
    const ConstantInfo* constants;  // Pointer to static array
    size_t count;                   // Number of constants
};
```

**Accessor function** (generated):
```mlir
llvm.func @get_constant_registry() -> !llvm.ptr {
  %ptr = llvm.mlir.addressof @constant_registry : !llvm.ptr
  llvm.return %ptr : !llvm.ptr
}
```

### Lifecycle

**Init:** Generated code calls `get_constant_registry()` → passes to runtime → runtime allocates GPU memory and uploads

**Compute:** Generated code calls `hip.get_constant(ctx, index)` → lowers to runtime accessor → runtime returns GPU pointer

**Cleanup:** Generated code calls cleanup → runtime frees GPU memory

**Key insight:** After llvm::Linker merge + LLVM optimization, runtime accessor calls are inlined. Final DLL has zero function call overhead.

---

## Separation of Concerns

This design achieves **elegant decoupling** between generated code (model-specific) and runtime bitcode (model-agnostic):

### 1. Generated Code ↔ Runtime Bitcode Decoupling

**Generated Code provides:**
- Constant data (LLVM globals in .data section)
- Metadata (ConstantRegistry with indices, sizes, pointers)
- Registry accessor (get_constant_registry)
- Constant references (hip.get_constant calls with indices)

**Generated Code does NOT:**
- Call GPU APIs (hipMalloc, hipMemcpy, hipFree)
- Know about RuntimeState structure
- Implement memory management strategy
- Know how constants are uploaded to GPU

**Runtime Bitcode provides:**
- GPU memory allocation (hipMalloc)
- Upload strategy (sequential, batched, async)
- Access strategy (array lookup, caching)
- Cleanup strategy (LIFO order, best-effort)
- RuntimeState structure and management

**Runtime Bitcode does NOT:**
- Know what constants exist in the model
- Know constant indices (model-specific)
- Parse LLVM globals directly
- Depend on model compilation details

**Interface between them:** ConstantRegistry metadata structure

**Benefit:** Model compilation (generated code) and GPU management (runtime bitcode) evolve independently. After merge, they form one optimized DLL.

### 2. Memory Management Flexibility

Because runtime bitcode owns GPU lifecycle and is merged via llvm::Linker (not hard-coded in generated MLIR), memory strategies can evolve without changing code generation:

**Current strategy (in runtime bitcode):**
```cpp
// Sequential upload
for (i = 0; i < registry->count; i++) {
    hipMalloc(&gpu_constants[i], size);
    hipMemcpy(gpu_constants[i], cpu_data, size, H2D);
}
```

**Future optimizations (change runtime bitcode only):**
- **Batched upload**: Single large buffer, one hipMemcpy
- **Pinned memory**: Use hipHostMalloc for zero-copy
- **Async upload**: Overlap with initialization using streams
- **Memory pooling**: Reuse allocations across sessions
- **On-demand loading**: Upload lazily on first `hip.get_constant` call
- **Compression**: Decompress on GPU after upload

**No model recompilation needed** - just rebuild runtime bitcode, re-merge with model IR during next compilation.

**Key insight:** Generated code uses `hip.get_constant(ctx, index)` which lowers to `hipdnn_ep_constant_get(state, index)`. Runtime bitcode can change internal strategy as long as accessor returns correct GPU pointer. After LLVM optimization, accessor is inlined anyway (zero overhead).

### 3. Constant Management ↔ Function Signatures Decoupling

**Without registry (rejected):**
```mlir
// Every constant = one argument
func.func @main_graph(%ctx, %in, %out, %w0, %w1, ..., %w99) -> i32   // 100 constants
func.func @main_graph(%ctx, %in, %out, %w0, %w1, ..., %w100) -> i32  // 101 constants (BREAKS ABI)
```

**With registry (chosen):**
```mlir
// Signature never changes
func.func @main_graph(%ctx: !hip.context, %in: memref<...>, %out: memref<...>) -> i32 {
  %w0 = hip.get_constant(%ctx, 0)
  %w1 = hip.get_constant(%ctx, 1)
  // Adding %w100 does NOT change signature
}
```

**Benefit:** Functions scale to any constant count (10 or 10,000). ABI stable.

---

## Design Decisions

### 1. Registry Pattern (vs Direct GPU Calls in Generated Code)

**Decision:** Generated code provides metadata registry, runtime bitcode handles GPU operations. Both merge into one DLL.

**Trade-offs:**

| Aspect | Registry (Chosen) | Direct GPU Calls in Generated Code |
|--------|-------------------|-------------------------------------|
| Code generation | Generate metadata only | Generate hipMalloc/hipMemcpy calls |
| Runtime flexibility | Change strategy in runtime bitcode | Hard-coded in generated MLIR |
| Recompilation | Runtime bitcode only | Entire model pipeline |
| Complexity | Simple metadata structs | Complex OpBuilder GPU sequences |
| Error handling | Centralized in runtime | Scattered in generated code |
| Optimization | Strategies evolve freely | Locked at model compilation |

**Rationale:**
- **Clean separation**: Model knows data, runtime knows GPU
- **Flexibility**: Upload strategies can change without regenerating models
- **Simplicity**: Metadata generation simpler than GPU code generation
- **Reusability**: Runtime bitcode is model-agnostic (pre-compiled once, merged into all models)

### 2. Embed Constants in DLL (vs External Files)

**Decision:** Constants embedded in DLL .data section, not separate weight files.

**Context:** EPContext stores a tar file (either embedded in ONNX or referenced by filename). This decision is about what goes INSIDE the tar file.

**Trade-offs:**

| Aspect | Embedded (Chosen) | External Files |
|--------|-------------------|----------------|
| Tar contents | 1 file: model.dll | N files: model.dll + constant_*.bin |
| Total size | Same (code + weights) | Same (code + weights) |
| DLL size | Larger (includes weights) | Smaller (code only) |
| Runtime extraction | Load DLL, done | Extract tar, load DLL, parse weight files |
| Runtime logic | Access .data section | Parse files, map indices to files |
| File format coupling | None (DLL is standard) | Custom weight format + parsing |

**Rationale:**
- **Simple runtime**: One file to load, no parsing
- **No format coupling**: DLL .data section is standard LLVM/OS behavior
- **Memory flexibility**: Runtime owns GPU upload strategy (see above). Can optimize without recompiling models. External files would couple to file format.
- **Single-artifact deployment**: Matches EPContext pattern

**Note:** Current DLL loading has 2x memory overhead (MemoryModule copies sections). Potential optimization via mmap + direct execution deferred. See [EPCONTEXT-MEMORY-OPTIMIZATION.md](EPCONTEXT-MEMORY-OPTIMIZATION.md).

### 3. Opaque Runtime Access (vs Direct Field Access)

**Decision:** Generated code uses accessor `hipdnn_ep_constant_get(state, index)`, not GEP into RuntimeState.

**Trade-offs:**

| Aspect | Opaque (Chosen) | Direct GEP |
|--------|-----------------|------------|
| ABI stability | Runtime can change RuntimeState layout | Breaks if fields reordered |
| Coupling | Loose (function interface) | Tight (struct layout) |
| Optimization | LLVM inlines to GEP anyway | Direct GEP |
| Maintainability | Runtime evolves freely | Generated code tied to struct |

**Rationale:** ABI stability - runtime can evolve RuntimeState structure without breaking generated code. After LLVM optimization, accessor inlined to direct access (zero overhead). See [RUNTIME-ARCHITECTURE.md - Opaque RuntimeState](RUNTIME-ARCHITECTURE.md#1-core-design-opaque-runtimestate-pattern).

---

## Open Questions

### Subgraph Constant Handling

**Context:** ONNX control flow (If/Loop/Scan) creates subgraphs as separate functions. Constants may be passed as arguments in original ONNX.

**Design decision:** Subgraphs receive only %ctx and load constants from state (no constant arguments).

**Open question:** How does the pass track that a constant passed as argument in original ONNX maps to global constant index after transformation?

**Example:**
```mlir
// BEFORE (ONNX)
func.func @main_graph(%input: tensor<...>) {
  %w0 = "onnx.Constant"() {value = dense<...>}
  %result = func.call @subgraph(%input, %w0)  // Pass constant as arg
}

func.func @subgraph(%arg0: tensor<...>, %arg1: tensor<...>) {  // %arg1 is constant
  %w1 = "onnx.Constant"() {value = dense<...>}  // Local constant
  %conv = "onnx.Conv"(%arg0, %arg1, %w1)
}

// AFTER (HIP)
func.func @main_graph(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32 {
  func.call @subgraph(%ctx, %input, %temp)  // No constant args
}

func.func @subgraph(%ctx: !hip.context, %arg0: memref<...>, %output: memref<...>) -> i32 {
  %w0 = hip.get_constant(%ctx, ???)  // Which index for original %arg1?
  %w1 = hip.get_constant(%ctx, ???)  // Which index for local %w1?
}
```

**TODO:** Implement constant provenance tracking across function boundaries.

---

## Related Documents

**Architecture:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design decisions
- [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) - Runtime bitcode design, opaque RuntimeState pattern, IR merging

**Implementation:**
- [mlir/passes/06-GenerateInterfacePass.md](mlir/passes/06-GenerateInterfacePass.md) - How inference_init uses registry
- [mlir/passes/01-OnnxToHip.md](mlir/passes/01-OnnxToHip.md) - Constant discovery and registry generation
- [mlir/passes/05-HipToLLVM.md](mlir/passes/05-HipToLLVM.md) - hip.get_constant lowering

**Related Design:**
- [EPCONTEXT-MEMORY-OPTIMIZATION.md](EPCONTEXT-MEMORY-OPTIMIZATION.md) - Future DLL loading memory optimizations
