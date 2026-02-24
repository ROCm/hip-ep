<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Dynamic Shape Support Design

**Status:** NOT IMPLEMENTED - Design Only
**Blocker:** Incompatible with memory pooling (60% memory savings)

**Date:** 2026-02-14
**Document Type:** Design
**Review Status:** Draft
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md), [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md), [04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md)

---

## Overview

Dynamic shapes (tensors with runtime-determined dimensions) are not currently supported due to fundamental incompatibility with memory pooling optimization.

The C interface design supports runtime shapes, but critical compilation passes reject dynamic dimensions, preventing real-world usage.

---

## Current Limitations

### Memory Pooling Rejects Dynamic Shapes

MemoryPoolingPass explicitly rejects dynamic dimensions:

```cpp
if (!memrefType.hasStaticShape()) {
  // For now, we don't support dynamic shapes
  // In the future, we could use conservative upper bounds
  return failure();
}
```

**Why this blocks dynamic shapes:**
- Graph coloring algorithm requires compile-time buffer sizes
- Pool size must be constant (stored as module attribute)
- Buffer offsets must be static (attached as compile-time metadata)
- Single pool allocation happens in `inference_init` with fixed size

**Impact:** Models with dynamic dimensions cannot compile with memory pooling enabled.

See [04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) for algorithm details.

### No Test Coverage for Dynamic Operations

Test coverage validates memory operations only:
- Memory allocation/deallocation with dynamic dimensions works
- No tests for Conv/GEMM/other compute ops with dynamic shapes
- No validation that MIOpen descriptors accept runtime dimensions correctly

### Missing Performance Validation

Design claims "negligible overhead" for dynamic shapes but provides no benchmarks:
- No measurements comparing static vs dynamic compilation
- No profiling of descriptor creation overhead
- Claims based on theoretical analysis, not empirical data

---

## What Actually Works

### C Interface Accepts Runtime Shapes

```c
typedef struct {
    void* data;
    int64_t* shape;  // Runtime dimension values
    int rank;
} tensor_t;

int inference_compute(void* state, span_t* inputs, span_t* outputs);
```

The interface can receive tensors with runtime-determined dimensions.

### GenerateInterfacePass Can Load Dimensions at Runtime

Code generation supports loading shape values from pointers:

```mlir
// Load dimension value from tensor_t.shape array
%shape_ptr = llvm.getelementptr %shape[0] : (!llvm.ptr) -> !llvm.ptr
%size_0 = llvm.load %shape_ptr : i64
```

### Memref Structs Support Runtime Dimensions

LLVM memref descriptors store dimension values in arrays:

```mlir
!llvm.struct<(
  ptr<1>,           // allocated_ptr
  ptr<1>,           // aligned_ptr
  i64,              // offset
  array<4 x i64>,   // sizes - can hold runtime values
  array<4 x i64>    // strides
)>
```

Static vs dynamic only differs in how the size array is populated (constants vs loads).

---

## Design Vision (Not Implemented)

The original design assumed dynamic shapes could work by:

1. CustomOp passes runtime shapes via `tensor_t.shape`
2. `inference_compute` loads dimensions from shape pointer
3. Memref structs built with runtime dimension values
4. Wrapper functions extract dimensions and pass to MIOpen
5. MIOpen handles runtime dimensions natively

**This design is sound for the interface and runtime layers, but breaks at the memory pooling optimization layer.**

---

## Implementation Challenges

### Challenge 1: Memory Pooling Requires Static Sizes

Graph coloring algorithm assigns buffer offsets based on:
- Computing exact buffer sizes (bytes = elements × element_size)
- Building interference graph from liveness analysis
- Assigning static offsets that don't overlap

Dynamic shapes break this because:
- Buffer size unknown at compile time
- Cannot assign fixed offsets without knowing sizes
- Pool size cannot be computed without knowing all buffer sizes

### Challenge 2: Conservative Upper Bounds Add Complexity

Potential workaround: Use conservative upper bounds for dynamic dimensions.

Challenges:
- How to specify upper bounds? (in model, in EP config, hardcoded?)
- Wastes memory for common case (batch=1 but plan for batch=64)
- Requires tracking actual vs maximum dimensions
- Pool may be much larger than needed (defeats pooling benefit)

### Challenge 3: Multi-Pool Strategy

Alternative: Separate pools for static and dynamic buffers.

Challenges:
- Static buffers use graph coloring (current implementation)
- Dynamic buffers use individual allocations (defeats pooling)
- Mixed models (some static, some dynamic ops) get no pooling benefit for dynamic parts
- Increased code complexity for marginal benefit

### Challenge 4: Runtime Pool Sizing

Alternative: Compute pool size at runtime in `inference_init`.

Challenges:
- Requires passing dimension information to `inference_init`
- Interface change (breaks C ABI stability)
- Complicates initialization (may fail due to OOM)
- Still need conservative bounds or multi-pool strategy

---

## Implementation Status

### What Needs Implementation

- [ ] ~~Generate `@main_graph` with rank-specific memref struct types~~ (assumed done, not tested with dynamic shapes)
- [ ] ~~Add function attributes for ranks~~ (assumed done, not tested)
- [ ] ~~Lower memrefs to structs in HipToLLVM~~ (assumed done for static shapes)
- [ ] ~~Generate wrapper functions~~ (assumed done for static shapes)
- [ ] ~~Generate `inference_compute` that loads dimensions~~ (assumed done, not tested)
- [ ] ~~Compute strides from dimensions~~ (assumed done, not tested)
- [ ] ~~Build memref struct with runtime dimensions~~ (assumed done, not tested)
- [ ] Test Conv with dynamic batch: `?x3x224x224xf32`
- [ ] Test GEMM with dynamic dims: `?x256xf32`, `256x?xf32`
- [ ] Test fully dynamic: `?x?x?x?xf32`
- [ ] Benchmark static vs dynamic overhead
- [ ] Resolve memory pooling incompatibility (choose approach from challenges)
- [ ] Implement chosen approach (conservative bounds, multi-pool, or runtime sizing)

### Current State

**Works:**
- C interface accepts runtime shapes
- Code generation can emit dimension loads
- Memref descriptors support runtime values

**Does NOT work:**
- Memory pooling with dynamic shapes (rejected at compile time)
- Any model with dynamic dimensions (pooling is mandatory)
- Compute operations with dynamic shapes (no test coverage)

---

## Technical Background

### MLIR Type System

MLIR distinguishes rank (compile-time) from dimension values (static or dynamic):

```mlir
memref<1x3x224x224xf32, 1>     // Static shape
memref<?x3x224x224xf32, 1>     // Dynamic batch
memref<?x?x?x?xf32, 1>         // Fully dynamic
```

All 4D tensors lower to same struct type (determined by rank=4), regardless of which dimensions are static.

### Lowering to LLVM

Difference between static and dynamic is how size array is populated:

```mlir
// Static: constant
%desc = llvm.insertvalue %c224, %desc[3, 2]

// Dynamic: runtime value
%size = llvm.load %shape_ptr : i64
%desc = llvm.insertvalue %size, %desc[3, 2]
```

The struct type is identical; only the initialization differs.

---

## Potential Paths Forward

### Option 1: Disable Pooling for Dynamic Models

**Approach:** Detect dynamic shapes, skip MemoryPoolingPass entirely.

**Trade-offs:**
- ✅ Simple to implement
- ✅ Dynamic shapes work (no pooling rejection)
- ❌ Lose 60% memory savings
- ❌ All-or-nothing (mixed models get no pooling)

### Option 2: Conservative Upper Bounds

**Approach:** User specifies maximum dimensions, pool uses worst-case sizes.

**Trade-offs:**
- ✅ Pooling still works
- ✅ Dynamic shapes supported
- ❌ Wastes memory (plan for max, use less)
- ❌ Requires interface change (how to specify bounds?)
- ❌ Complex metadata (actual dims vs max dims)

### Option 3: Multi-Pool Strategy

**Approach:** Static buffers use graph coloring pool, dynamic buffers use separate pool or individual allocations.

**Trade-offs:**
- ✅ Static ops get full pooling benefit
- ✅ Dynamic ops don't block static pooling
- ❌ Complex implementation (two allocation strategies)
- ❌ Dynamic ops lose pooling benefit
- ❌ May require multiple `hipMalloc` calls

### Option 4: Runtime Pool Computation

**Approach:** Defer pool size computation to `inference_init`, compute based on actual input dimensions.

**Trade-offs:**
- ✅ Exact pool size (no waste)
- ✅ Pooling benefit preserved
- ❌ Interface change (pass dimensions to init)
- ❌ Initialization can fail (OOM)
- ❌ Complex: need graph coloring at runtime or conservative compile-time bounds

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - C interface design
- [04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) - Memory pooling algorithm and limitations
- [BUFFER-LIFETIME-DESIGN.md](BUFFER-LIFETIME-DESIGN.md) - Buffer lifetime management
- [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) - Wrapper functions and compilation pipeline
