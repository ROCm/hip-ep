<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #061: Remove Misleading Shape Inference Code

## Metadata
- **Type:** Code Cleanup
- **Priority:** MEDIUM
- **Created:** 2026-02-09
- **Related:** Architecture.md Section 4.6 (Shape Inference Requirements)

## Problem

The codebase contains misleading references to shape inference functionality that is effectively non-functional in MorphiZen:

**Current Reality:**
- **Default backend (MLIR)**: Shape inference NOT implemented (just logs warning)
- **Non-default backend (ONNX)**: Disabled by default, and when enabled has subtle bugs and doesn't support custom ONNX domains/operators
- **Conclusion**: Shape inference is effectively non-functional and should not be advertised as a MorphiZen feature

**Misleading Code:**
1. **Commented-out test** (`Test15_ShapeInferenceOperations` in `ort-bridge/test/src/test-morphizen-ort-implementation.cpp:1048-1064`)
   - Entire test body is commented out
   - Gives impression feature exists but is temporarily disabled

2. **Misleading documentation** (`morphizen-graph/include/morphizen/graph.hpp:630`)
   - Documents that `resolve()` includes "1. Shape inference"
   - While technically true for onnx-ir-imp backend, it's misleading because:
     - Default backend (MLIR) doesn't do it
     - Non-default backend (ONNX) is buggy and disabled by default
     - Not a supported MorphiZen feature

## Why It Matters

- Creates confusion about MorphiZen's capabilities
- Users may incorrectly assume shape inference is a supported feature
- Conflicts with architectural documentation (Section 4.6) stating shape inference is NOT a MorphiZen feature
- Dead/commented code clutters the codebase

## Solution

Remove misleading references to shape inference:

1. **Delete commented-out test**
   - Remove entire `Test15_ShapeInferenceOperations()` function
   - Update test documentation files that reference Test15

2. **Update graph.hpp documentation**
   - Remove "1. Shape inference" from the resolve() function's documentation
   - Keep accurate items: edge/node relationships, data structure cleanup

## Files to Modify

- `ort-bridge/test/src/test-morphizen-ort-implementation.cpp` - delete test function
- `morphizen-graph/include/morphizen/graph.hpp` - update resolve() documentation
- `ort-bridge/test/src/README-test-runner.md` - possibly update if Test15 is referenced
- `ort-bridge/test/src/SUMMARY-comprehensive-tests.md` - possibly update if Test15 is referenced

## Implementation Complexity

**Simple** - Straightforward deletions and documentation updates, no logic changes.

## Notes

- Public API header updates (`morphizen-ort-api-ext.hpp`) will be handled in separate issues
- This cleanup aligns with architecture.md Section 4.6 which documents that shape inference is NOT a MorphiZen feature
- See PR #164 for related documentation updates
