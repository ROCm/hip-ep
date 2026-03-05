<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Plan #001: Consolidate HipDialect .td files into include/

## Overview

Take the authoritative `.td` content from `lib/Dialect/IR/`, land it in `include/hip/Dialect/IR/` with corrected include paths, delete the `lib/` copies, and update `CMakeLists.txt` to drive TableGen from the new location.

---

## Step 1: Overwrite include/hip/Dialect/IR/HipDialect.td

Copy content from `lib/Dialect/IR/HipDialect.td` verbatim — no include path changes needed (it only includes MLIR files, not project files).

**Before** (`include/hip/Dialect/IR/HipDialect.td`):
```tablegen
// Copyright (C) 2026 ...
#ifndef HIP_DIALECT
#define HIP_DIALECT
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
def Hip_Dialect : Dialect {
  ...
  let useDefaultTypePrinterParser = 1;
  // no dependentDialects
}
#endif
```

**After** (content of `lib/Dialect/IR/HipDialect.td`):
```tablegen
#ifndef HIP_DIALECT
#define HIP_DIALECT
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
def Hip_Dialect : Dialect {
  ...
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "memref::MemRefDialect",
    "bufferization::BufferizationDialect"
  ];
}
#endif
```

---

## Step 2: Overwrite include/hip/Dialect/IR/HipTypes.td

Copy content from `lib/Dialect/IR/HipTypes.td`, fixing the include path.

**Include path fix:**
```tablegen
// Before (lib/ relative path):
include "HipDialect.td"

// After (include/ prefixed path):
include "hip/Dialect/IR/HipDialect.td"
```

The type definition changes from `Hip_HandleType` to `Hip_ContextType` — take it wholesale from `lib/`.

---

## Step 3: Overwrite include/hip/Dialect/IR/HipOps.td

Copy content from `lib/Dialect/IR/HipOps.td`, fixing two include paths.

**Include path fixes:**
```tablegen
// Before (lib/ relative paths):
include "HipDialect.td"
include "HipTypes.td"

// After (include/ prefixed paths):
include "hip/Dialect/IR/HipDialect.td"
include "hip/Dialect/IR/HipTypes.td"
```

All other content (766 lines of op definitions) copied verbatim.

---

## Step 4: Delete lib/ .td duplicates

```bash
git rm lib/Dialect/IR/HipDialect.td
git rm lib/Dialect/IR/HipTypes.td
git rm lib/Dialect/IR/HipOps.td
```

---

## Step 5: Fix lib/Dialect/IR/CMakeLists.txt

### 5a: Fix TableGen output path (line 30)

```cmake
# Before:
set(TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/include/udna-compiler/Dialect/Hip/IR)

# After:
set(TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/include/hip/Dialect/IR)
```

Also update the comment on line 29:
```cmake
# Before:
# This allows headers to use #include "udna-compiler/Dialect/Hip/IR/HipDialect.h.inc"

# After:
# This allows headers to use #include "hip/Dialect/IR/HipDialect.h.inc"
```

### 5b: Fix LLVM_TARGET_DEFINITIONS (lines 38, 44, 50)

```cmake
# Before:
set(LLVM_TARGET_DEFINITIONS HipDialect.td)
...
set(LLVM_TARGET_DEFINITIONS HipTypes.td)
...
set(LLVM_TARGET_DEFINITIONS HipOps.td)

# After:
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/hip/Dialect/IR/HipDialect.td)
...
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/hip/Dialect/IR/HipTypes.td)
...
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/hip/Dialect/IR/HipOps.td)
```

### 5c: Rename library UdnaHipDialectIR → HipDialectIR (lines 56, 62, 72, 83, 91)

```cmake
# Before:
add_library(UdnaHipDialectIR STATIC ...)
add_dependencies(UdnaHipDialectIR ...)
target_link_libraries(UdnaHipDialectIR PUBLIC ...)
target_include_directories(UdnaHipDialectIR PUBLIC ...)
set_target_properties(UdnaHipDialectIR PROPERTIES ...)

# After:
add_library(HipDialectIR STATIC ...)
add_dependencies(HipDialectIR ...)
target_link_libraries(HipDialectIR PUBLIC ...)
target_include_directories(HipDialectIR PUBLIC ...)
set_target_properties(HipDialectIR PROPERTIES ...)
```

---

## Step 6: Update downstream CMakeLists.txt

Replace `UdnaHipDialectIR` → `HipDialectIR` in all 7 downstream files:

| File | Line |
|------|------|
| `tools/hip-mlir-opt/CMakeLists.txt` | 19 |
| `lib/Compiler/CMakeLists.txt` | 22 |
| `lib/Conversion/OnnxToHip/CMakeLists.txt` | 16 |
| `lib/Compiler/Pipeline/CMakeLists.txt` | 16 |
| `lib/Dialect/Transforms/CMakeLists.txt` | 16 |
| `lib/Compiler/Passes/CMakeLists.txt` | 15 |
| `lib/Conversion/HipToLLVM/CMakeLists.txt` | 15 |

---

## Verification

```bash
# 1. Confirm no .td files remain in lib/Dialect/IR/
ls lib/Dialect/IR/*.td  # should return: no matches

# 2. Confirm no references to UdnaHipDialectIR remain
grep -r "UdnaHipDialectIR" --include="CMakeLists.txt"  # should return: nothing

# 3. Configure and build
cmake --build <build-dir> --target HipDialectIR

# 4. Confirm generated headers land in correct location
ls <build-dir>/include/hip/Dialect/IR/*.h.inc
```

## Success Criteria

- [ ] `lib/Dialect/IR/HipDialect.td`, `HipTypes.td`, `HipOps.td` deleted
- [ ] `include/hip/Dialect/IR/*.td` contain authoritative content
- [ ] Include paths in `HipTypes.td` and `HipOps.td` use `hip/Dialect/IR/` prefix
- [ ] Generated headers at `build/include/hip/Dialect/IR/`
- [ ] No references to `UdnaHipDialectIR` anywhere
- [ ] Build succeeds
