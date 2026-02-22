<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Buffer Lifetime Management Design

**Date:** 2026-02-14
**Document Type:** Design
**Status:** Draft
**Related:** [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md), [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md)

---

## Overview

HIP dialect operations allocate GPU buffers (`hip.alloc`) but don't specify when to free them. Without explicit deallocation points, memory pooling optimizations cannot determine buffer lifetimes for reuse analysis.

MLIR provides a standard BufferDeallocation pipeline that automatically inserts deallocation operations based on liveness analysis.

---

## Design

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│ OnnxToHip Pass                                          │
│ - Creates hip.alloc operations                          │
│ - Does NOT insert hip.free                              │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ BufferDeallocation Pipeline (MLIR standard)             │
│ - Performs ownership analysis                           │
│ - Performs liveness analysis                            │
│ - Automatically inserts hip.free after last use         │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ OptimizeAllocationLiveness (MLIR standard)              │
│ - Temporal optimization: Minimize peak memory usage     │
│ - Moves hip.free earlier to reduce lifetime overlap     │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ 04-MemoryPooling                                       │
│ - Spatial optimization: Reuse memory for non-overlapping│
│   buffers by assigning them to same pool location       │
│ - Attaches pool metadata to module                      │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ HipToLLVM Pass                                          │
│ - hip.alloc → pool_ptr + offset[i]                      │
│ - hip.free → nop                                        │
└─────────────────────────────────────────────────────────┘
```

**Key distinction:**
- **OptimizeAllocationLiveness**: Temporal optimization - when to free buffers (minimizes peak memory)
- **[04-MemoryPooling](mlir/passes/04-MemoryPooling.md)**: Spatial optimization - where to place buffers (enables memory reuse)

Without pooling, each `hip.alloc` → separate `hipMalloc()` → different memory regions. With pooling, non-overlapping buffers → same memory location.

### Separation of Concerns

| Concern | Handler | Responsibility |
|---------|---------|----------------|
| **Allocation semantics** | OnnxToHip | Create `hip.alloc` for intermediate buffers |
| **Deallocation placement** | BufferDeallocation | Insert `hip.free` after last use |
| **Lifetime optimization** | OptimizeAllocationLiveness | Minimize peak memory (temporal) |
| **Pool assignment** | 04-MemoryPooling | Assign offsets for reuse (spatial) |
| **Lowering** | HipToLLVM | Lower to pool-based allocation |

### Example Transformation

**After OnnxToHip:**
```mlir
func.func @main_graph(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32 {
  %buf1 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
  hip.conv(%ctx, %input, %weights, %bias, %buf1)

  %buf2 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
  hip.relu(%ctx, %buf1, %buf2)

  memref.copy %buf2, %output

  return %c0 : i32
}
```

**After BufferDeallocation + OptimizeAllocationLiveness:**
```mlir
func.func @main_graph(%ctx: !hip.context, %input: memref<...>, %output: memref<...>) -> i32 {
  %buf1 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
  hip.conv(%ctx, %input, %weights, %bias, %buf1)

  %buf2 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
  hip.relu(%ctx, %buf1, %buf2)
  hip.free(%ctx, %buf1)  // Inserted: buf1 last used by relu

  memref.copy %buf2, %output
  hip.free(%ctx, %buf2)  // Inserted: buf2 last used by copy

  return %c0 : i32
}
```

**Liveness visualization:**
```
Operations:  [conv]  [relu]  [copy]
             ─────────────────────────────▶ time
buf1:        [████████████)
buf2:                [████████████)
                     ↑
                  overlap
                  peak: 2 buffers
```

---

## Interface Requirements

HIP operations must declare memory effects for BufferDeallocation to analyze lifetimes:

### AllocationOpInterface

```tablegen
// HipOps.td
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"

def Hip_AllocOp : Hip_Op<"alloc", [
  DeclareOpInterfaceMethods<AllocationOpInterface>
]> {
  let results = (outs AnyMemRef:$result);
}
```

Implementation declares ownership:
```cpp
// HipOps.cpp
bufferization::AllocationInfo HipAllocOp::getAllocationInfo() {
  return {
    .ownership = bufferization::Ownership::Owned,  // We own this buffer
    .hoistable = true  // Can move out of loops
  };
}
```

### Memory Effects

Declare effects on operations:
```tablegen
def Hip_AllocOp : Hip_Op<"alloc", [MemAlloc]>;
def Hip_FreeOp : Hip_Op<"free", [MemFree]>;
def Hip_ConvOp : Hip_Op<"conv", [MemRead, MemWrite]>;
```

BufferDeallocation uses these to determine:
- Which operations allocate (creates lifetime start)
- Which operations use buffers (extends lifetime)
- Where last use occurs (lifetime end point)

---

## Ownership Analysis

BufferDeallocation determines which buffers the function owns:

| Buffer Source | Ownership | Action |
|---------------|-----------|--------|
| Function argument | Caller owns | Do not insert `hip.free` |
| `hip.alloc` result | Function owns | Insert `hip.free` after last use |
| Constant (global) | Nobody owns | Do not insert `hip.free` |

Example with function argument:
```mlir
func.func @main_graph(%ctx: !hip.context, %input: memref<...>) {
  // %input is argument → caller owns → no free inserted

  %temp = hip.alloc(%ctx) : memref<...>
  // %temp is allocated → function owns → free inserted
  hip.conv(%ctx, %input, %weights, %bias, %temp)
  hip.free(%ctx, %temp)  // Inserted by BufferDeallocation
}
```

---

## Compiler Pipeline Integration

```cpp
void buildHipDnnPipeline(PassManager &pm) {
  // Conversion
  pm.addPass(createConvertOnnxToHipPass());

  // Buffer lifetime management (MLIR standard)
  pm.addPass(bufferization::createBufferLoopHoistingPass());
  bufferization::BufferDeallocationPipelineOptions opts;
  bufferization::buildBufferDeallocationPipeline(pm, opts);
  pm.addPass(bufferization::createOptimizeAllocationLivenessPass());

  // Memory pooling
  pm.addPass(createHip04-MemoryPooling());

  // Lowering
  pm.addPass(createConvertHipToLLVMPass());
  pm.addPass(createGenerateInterfacePass());
}
```

BufferDeallocation runs on HIP dialect IR before lowering to LLVM. This preserves type information needed for size calculation and ownership analysis.

---

## Related Documents

- [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) - Memory allocation strategy
- [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) - Transformation pipeline
- [mlir/HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) - HIP dialect operations
- [mlir/passes/01-OnnxToHip.md](mlir/passes/01-OnnxToHip.md) - ONNX to HIP conversion
- [mlir/passes/02-BufferDeallocation.md](mlir/passes/02-BufferDeallocation.md) - Buffer deallocation pipeline
- [mlir/passes/04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) - Memory pooling
- [mlir/passes/05-HipToLLVM.md](mlir/passes/05-HipToLLVM.md) - HIP to LLVM lowering
