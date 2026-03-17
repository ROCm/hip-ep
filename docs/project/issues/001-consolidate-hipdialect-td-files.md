<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #001: Consolidate HipDialect .td files into include/

## Metadata
- **Type:** Refactoring
- **Priority:** HIGH
- **Dependencies:** None

## Description

`migrate-hip-compiler` added authoritative `.td` files in `lib/Dialect/IR/` (wrong location per MLIR convention). The `include/hip/Dialect/IR/` copies from `origin/main` are stale. The `lib/` files should overwrite the `include/` files (with include path fixes), then the `lib/` copies deleted, and `CMakeLists.txt` updated to drive TableGen from the correct location.

## Problem

**Two diverged sets of `.td` files — `lib/` is authoritative, `include/` is stale:**

```
include/hip/Dialect/IR/HipDialect.td  ← stale (origin/main)
include/hip/Dialect/IR/HipTypes.td    ← stale, has Hip_HandleType only
include/hip/Dialect/IR/HipOps.td      ← stale, 300 lines

lib/Dialect/IR/HipDialect.td          ← authoritative (migrate-hip-compiler)
lib/Dialect/IR/HipTypes.td            ← authoritative, has Hip_ContextType
lib/Dialect/IR/HipOps.td              ← authoritative, 766 lines with all current ops
```

**Why this is problematic:**
1. MLIR convention: `.td` files must live in `include/` so downstream consumers can find them via `mlir-tblgen -I include/`.
2. `lib/Dialect/IR/CMakeLists.txt` drives TableGen from `lib/*.td` (relative paths like `HipDialect.td`) and writes generated headers to `build/include/hip-compiler/Dialect/Hip/IR/` — both wrong.
3. Library is named `UdnaHipDialectIR`; the `Udna` prefix is inconsistent with the rest of the project naming.

**Code locations:**
- `lib/Dialect/IR/HipDialect.td`, `HipTypes.td`, `HipOps.td` — authoritative content, wrong location
- `include/hip/Dialect/IR/HipDialect.td`, `HipTypes.td`, `HipOps.td` — stale content, right location
- `lib/Dialect/IR/CMakeLists.txt:30` — wrong output path `hip-compiler/Dialect/Hip/IR`
- `lib/Dialect/IR/CMakeLists.txt:38,44,50` — `LLVM_TARGET_DEFINITIONS` uses bare filenames (relative to `lib/`)
- `lib/Dialect/IR/CMakeLists.txt:56` — library named `UdnaHipDialectIR`

## Solution

**Approach:**
1. Overwrite `include/hip/Dialect/IR/*.td` with content from `lib/Dialect/IR/*.td`, fixing only include paths from bare names to `hip/Dialect/IR/`-prefixed:
   - `"HipDialect.td"` → `"hip/Dialect/IR/HipDialect.td"`
   - `"HipTypes.td"` → `"hip/Dialect/IR/HipTypes.td"`
2. Delete `lib/Dialect/IR/HipDialect.td`, `HipTypes.td`, `HipOps.td`
3. Fix `lib/Dialect/IR/CMakeLists.txt`:
   - Output path: `hip-compiler/Dialect/Hip/IR` → `hip/Dialect/IR`
   - `LLVM_TARGET_DEFINITIONS`: bare filenames → full paths from `${CMAKE_SOURCE_DIR}/include/hip/Dialect/IR/`
   - Library name: `UdnaHipDialectIR` → `HipDialectIR` (all 5 occurrences)
4. Update all 7 downstream `CMakeLists.txt` that link `UdnaHipDialectIR` → `HipDialectIR`

**Benefits:**
- ✅ Single source of truth for dialect definition in the correct location
- ✅ Follows MLIR convention (`.td` in `include/`)
- ✅ Generated headers land at `build/include/hip/Dialect/IR/` matching consumer expectations
- ✅ Library name consistent with project naming (no `Udna` prefix)

## Plans

- [001-consolidate-hipdialect-td-files-plan.md](../plans/001-consolidate-hipdialect-td-files-plan.md) - Created 2026-03-05
