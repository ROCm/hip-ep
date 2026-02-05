<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #057: MLIR Backend Metadata API Not Implemented

## Metadata
- **Type:** Bug
- **Priority:** MEDIUM
- **Status:** OPEN
- **Created:** 2026-02-04
- **Dependencies:** None
- **Discovered from:** Issue #028 test enablement

## Description

MLIR backend's metadata API (`set_metadata`, `get_metadata`, `has_metadata`) does not work correctly. Setting metadata and retrieving it returns an empty string instead of the set value.

## Problem

**Observed behavior:**
```cpp
auto model = morphizen_cxx::Model::load(RESNET_50_PATH);

std::string key = "author";
std::string value = "John Doe";
model->set_metadata(key, value);

std::string retrievedValue = model->ref().get_metadata(key);
// retrievedValue is "" (empty string), expected "John Doe"
```

**Error output:**
```
D:\ROCm\MorphiZen\unit-test\morphizen\test_model.cpp(49): error: Expected equality of these values:
  retrievedValue
    Which is: ""
  value
    Which is: "John Doe"
```

**Code locations:**
- Test: `unit-test/morphizen/test_model.cpp` - `ModelTest.SetAndGetMetadata`
- MLIR implementation: `mlir-imp/src/mlir-model.cpp` - metadata API functions

## Root Cause Analysis

**MLIR backend metadata storage is CORRECTLY implemented!** The actual bug is in the wrapper layer:

### Call Chain Analysis
1. `Model::set_metadata()` → `morphizen::model_set_meta_data()` → `MORPHIZEN_ORT_API(model_set_meta_data)()` → MLIR: `mlir_model->set_metadata_prop(key, value)` ✓
2. `ModelConstRef::get_metadata()` → `morphizen::model_get_meta_data()` → **BUG HERE!**

### The Bug Location: `morphizen-graph/src/graph.cpp`

```cpp
// BEFORE (BUG):
const std::string& model_get_meta_data(const Model& model, const std::string& key) {
  return *MORPHIZEN_ORT_API(model_get_meta_data)(model, key);  // Returns DllSafe<string>
}
```

**Problem:**
- `MORPHIZEN_ORT_API(model_get_meta_data)()` returns `DllSafe<std::string>` (a temporary object)
- The function dereferences this temporary and returns a **reference** to its contents
- When the function returns, `DllSafe` destructor runs → internal string is **deleted**
- The returned reference becomes a **dangling reference** (undefined behavior)
- Reading the dangling reference returns empty string or garbage

## Fix Applied

Changed `model_get_meta_data()` to return by **value** instead of by **reference**:

```cpp
// AFTER (FIXED):
std::string model_get_meta_data(const Model& model, const std::string& key) {
  // Return by value to avoid dangling reference
  return *MORPHIZEN_ORT_API(model_get_meta_data)(model, key);
}
```

Files modified:
- `morphizen-graph/include/morphizen/graph.hpp` - Changed declaration from `const std::string&` to `std::string`
- `morphizen-graph/src/graph.cpp` - Changed implementation to return by value

## Affected Tests

- `ModelTest.SetAndGetMetadata` - Fixed by this change

## Evidence

Test command:
```bash
.\bin\morphizen-unit-tests.exe --gtest_filter='ModelTest.SetAndGetMetadata'
```

Before fix: FAILED (empty string returned)
After fix: PASSED

## Status

**OPEN** - Root cause identified, fix pending implementation
