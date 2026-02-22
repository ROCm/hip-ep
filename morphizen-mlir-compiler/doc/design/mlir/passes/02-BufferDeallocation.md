<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# BufferDeallocation Pipeline

**Date:** 2026-02-20
**Document Type:** Implementation
**Status:** Draft
**Related:** [01-OnnxToHip.md](01-OnnxToHip.md), [03-Canonicalization.md](03-Canonicalization.md), [BUFFER-LIFETIME-DESIGN.md](../BUFFER-LIFETIME-DESIGN.md)

## Overview

Automatic memory management for HIP buffers. Three MLIR passes insert deallocation operations and optimize buffer lifetimes.

Runs after [OnnxToHip](01-OnnxToHip.md) generates allocations without matching deallocations. Before [Canonicalization](03-Canonicalization.md) cleans up dead code.

## Passes

### 1. BufferLoopHoisting

Moves `hip.alloc` outside loops when allocation can be reused across iterations.

**Before:**
```mlir
scf.for %i = %c0 to %c10 step %c1 {
  %temp = hip.alloc(%ctx) : memref<1x128x28x28xf32, 1>
  hip.relu(%ctx, %input, %temp) {...}
  hip.copy(%ctx, %temp, %output)
}
```

**After:**
```mlir
%temp = hip.alloc(%ctx) : memref<1x128x28x28xf32, 1>
scf.for %i = %c0 to %c10 step %c1 {
  hip.relu(%ctx, %input, %temp) {...}
  hip.copy(%ctx, %temp, %output)
}
```

Allocation overhead: O(n) → O(1) for loops with n iterations.

### 2. OwnershipBasedBufferDeallocation

Inserts `hip.free` after last use of function-owned buffers.

**Ownership model:**
- Function-owned: Created by `hip.alloc` inside function → freed automatically
- Caller-owned: Function arguments → NOT freed

**Before:**
```mlir
func.func @main_graph(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) {
  %temp = hip.alloc(%ctx) : memref<1x128x28x28xf32, 1>
  hip.conv(%ctx, %input, %weights, %bias, %temp) {...}
  memref.copy %temp, %output
  func.return
}
```

**After:**
```mlir
func.func @main_graph(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) {
  %temp = hip.alloc(%ctx) : memref<1x128x28x28xf32, 1>
  hip.conv(%ctx, %input, %weights, %bias, %temp) {...}
  memref.copy %temp, %output
  hip.free(%ctx, %temp)
  func.return
}
```

### 3. OptimizeAllocationLiveness

Moves `hip.free` earlier when safe. Reduces peak memory.

**Before:**
```mlir
func.func @main_graph(...) {
  %temp1 = hip.alloc(%ctx) : memref<1x64x56x56xf32, 1>  // 0.8 MB
  hip.conv(%ctx, %input, %w1, %b1, %temp1) {...}

  %temp2 = hip.alloc(%ctx) : memref<1x128x28x28xf32, 1>  // 1.6 MB
  hip.conv(%ctx, %temp1, %w2, %b2, %temp2) {...}

  hip.copy(%ctx, %temp2, %output)

  hip.free(%ctx, %temp1)
  hip.free(%ctx, %temp2)
}
// Peak: 0.8 + 1.6 = 2.4 MB
```

**After:**
```mlir
func.func @main_graph(...) {
  %temp1 = hip.alloc(%ctx) : memref<1x64x56x56xf32, 1>
  hip.conv(%ctx, %input, %w1, %b1, %temp1) {...}

  %temp2 = hip.alloc(%ctx) : memref<1x128x28x28xf32, 1>
  hip.conv(%ctx, %temp1, %w2, %b2, %temp2) {...}
  hip.free(%ctx, %temp1)  // Freed immediately after last use

  hip.copy(%ctx, %temp2, %output)
  hip.free(%ctx, %temp2)
}
// Peak: max(0.8, 1.6) = 1.6 MB
```

## MLIR Interface Requirements

### AllocationOpInterface

`hip.alloc` implements `buildDealloc()`:

```cpp
Value AllocOp::buildDealloc(OpBuilder &builder, Value alloc) {
  auto ctx = getContext();
  builder.create<FreeOp>(getLoc(), ctx, alloc);
  return Value();
}
```

Tells OwnershipBasedBufferDeallocation how to generate `hip.free`.

### MemoryEffectsOpInterface

All HIP operations declare memory effects (Allocate, Free, Read, Write).

**Examples:**
- `hip.alloc` → Allocate effect on result
- `hip.free` → Free effect on buffer operand
- `hip.conv` → Read on inputs, Write on output

Enables liveness analysis and optimization.

## Relationship to Other Passes

**OptimizeAllocationLiveness:** Temporal optimization (minimize lifetimes)
**[MemoryPooling](04-MemoryPooling.md):** Spatial optimization (reuse memory across non-overlapping buffers)

Complementary: OptimizeAllocationLiveness minimizes lifetimes → MemoryPooling uses lifetime info for graph coloring.

## Related Documents

- **[01-OnnxToHip.md](01-OnnxToHip.md)** - Previous pass (generates allocations)
- **[03-Canonicalization.md](03-Canonicalization.md)** - Next pass (dead code elimination)
- **[04-MemoryPooling.md](04-MemoryPooling.md)** - Uses liveness for spatial optimization
- **[BUFFER-LIFETIME-DESIGN.md](../BUFFER-LIFETIME-DESIGN.md)** - Ownership model
- **[MEMORY-MANAGEMENT.md](../MEMORY-MANAGEMENT.md)** - Overall strategy
- **[LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** - Pipeline overview
