<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Canonicalization Pass

**Date:** 2026-02-20
**Document Type:** Implementation
**Status:** Draft
**Related:** [01-OnnxToHip.md](01-OnnxToHip.md), [02-BufferDeallocation.md](02-BufferDeallocation.md), [HIP-COPY-DESIGN.md](../../HIP-COPY-DESIGN.md)

## Overview

MLIR Canonicalizer with HIP-specific copy elimination patterns. Runs after [BufferDeallocation](02-BufferDeallocation.md) to clean up dead allocations and eliminate redundant copies.

Two-level optimization:
1. Generic MLIR patterns (constant folding, dead code elimination)
2. HIP-specific patterns (copy elimination)

## Generic MLIR Patterns

Standard canonicalization patterns registered by `mlir::createCanonicalizerPass()`:

- Constant folding
- Dead code elimination (removes unused `hip.alloc` after copy elimination)
- Algebraic simplifications
- Control flow simplifications

## HIP-Specific Patterns

Registered via `hip::CopyOp::getCanonicalizationPatterns()`.

### Pattern 1: Self-Copy Elimination

**Pattern:** `hip.copy(%ctx, %buf, %buf)` → removed

**Implementation:** `EliminateSelfCopy` pattern

**Example:**
```mlir
// Before
hip.copy(%ctx, %buffer, %buffer)  // Copy buffer to itself

// After
// (operation removed)
```

### Pattern 2: Copy-After-DPS Elimination

**Pattern:** Operation writes to temporary, copy to output → Operation writes directly to output

**Implementation:** `EliminateCopyAfterDPSWrite` pattern

**Requires:** Source operation implements `DestinationStyleOpInterface`

**Algorithm:**

1. Check if copy source comes from `hip.alloc`
2. Find DPS operation writing to source buffer
3. Verify single use (only writer + copy)
4. Redirect DPS `init` operand from source to destination
5. Erase copy operation

**Example (hip.conv):**
```mlir
// Before
%temp = hip.alloc(%ctx) : memref<1x64x112x112xf32, 1>
hip.conv(%ctx, %input, %weights, %bias, %temp) {...}
hip.copy(%ctx, %temp, %output)

// After
hip.conv(%ctx, %input, %weights, %bias, %output) {...}
// %temp allocation becomes dead → removed by dead code elimination
```

**Example (hip.relu):**
```mlir
// Before
%temp = hip.alloc(%ctx) : memref<1x128x28x28xf32, 1>
hip.relu(%ctx, %input, %temp) {...}
hip.copy(%ctx, %temp, %output)

// After
hip.relu(%ctx, %input, %output) {...}
```

**Multiple uses (pattern does NOT apply):**
```mlir
// Pattern skipped - temp used twice
%temp = hip.alloc(%ctx) : memref<...>
hip.relu(%ctx, %input, %temp) {...}
hip.copy(%ctx, %temp, %output1)
hip.copy(%ctx, %temp, %output2)  // Second use prevents elimination
```

## DestinationStyleOpInterface

HIP operations that support copy elimination:

**Implementation pattern:**
```cpp
MutableOperandRange ConvOp::getDpsInitsMutable() {
  return getOutputMutable();  // Return output operand
}
```

**Operations:**
- `hip.conv` → output operand
- `hip.relu` → output operand
- `hip.max_pool` → output operand
- `hip.avg_pool` → output operand
- `hip.gemm` → result operand

Generic pattern works for ANY operation implementing this interface.

## Integration with BufferDeallocation

Copy elimination creates dead allocations:

```mlir
// After copy elimination
%temp = hip.alloc(%ctx) : memref<...>  // Allocated but never used
hip.conv(%ctx, %input, %weights, %bias, %output) {...}
hip.free(%ctx, %temp)  // Frees unused buffer

// After dead code elimination
hip.conv(%ctx, %input, %weights, %bias, %output) {...}
// Both hip.alloc and hip.free removed
```

Combined effect: Zero-copy semantics.

## Testing

Test file: `test/lit/Transforms/Canonicalize/test_hip_copy_elimination.mlir`

**Test 1:** Self-copy elimination
**Test 2:** Single-use buffer (hip.relu)
**Test 3:** Single-use buffer (hip.conv)
**Test 4:** Multiple uses (copy NOT eliminated)

## Related Documents

- **[HIP-COPY-DESIGN.md](../../HIP-COPY-DESIGN.md)** - Copy optimization strategy
- **[01-OnnxToHip.md](01-OnnxToHip.md)** - DPS optimization at generation time
- **[02-BufferDeallocation.md](02-BufferDeallocation.md)** - Dead code cleanup
- **[HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md)** - Operation definitions
- **[LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** - Pipeline overview
