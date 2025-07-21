<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Excluded Packages

This document lists packages that are temporarily excluded from the Bazel build.

## onnx-ir-imp

**Status**: Excluded
**Reason**: BUILD.bazel file contains glob patterns that don't match existing files (`src/core/*.cpp`)
**File**: `onnx-ir-imp/BUILD.bazel.disabled` (renamed from `BUILD.bazel`)

### To Re-enable

1. Check the directory structure in `onnx-ir-imp/src/core/` and ensure there are `.cpp` files
2. Update the glob patterns in the BUILD.bazel file to match actual file locations
3. Rename `BUILD.bazel.disabled` back to `BUILD.bazel`

### Current Issue

The BUILD.bazel file expects:
```starlark
srcs = glob([
    "src/*.cpp",
    "src/core/*.cpp",  # This pattern matches no files
]),
```

But the actual structure is:
```
src/
├── core/
│   └── graph/
│       ├── contrib_ops/
│       └── op.h
├── *.cpp (various files exist here)
└── *.hpp (various files exist here)
```

The `src/core/` directory only contains subdirectories and header files, no `.cpp` files directly.

### Possible Solutions

1. Move `.cpp` files to `src/core/` if they belong there
2. Update glob pattern to include subdirectories: `"src/core/**/*.cpp"`
3. Remove the problematic glob pattern if not needed
4. Add `allow_empty = True` to the glob pattern
