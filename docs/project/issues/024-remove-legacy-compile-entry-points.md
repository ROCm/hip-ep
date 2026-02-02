<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #024: Remove Legacy Compile Entry Points

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Tech Debt / Refactoring
- **Owner:** TBD
- **Created:** 2026-02-02
- **Dependencies:** None
- **Related:** #006, #010, #022 (legacy code cleanup issues)

## Description

Remove two unused legacy compile API entry points (`compile_onnx_model_morphizen_ep_with_options` and `compile_onnx_model_morphizen_ep_with_error_handling`) from morphizen-core, keeping only the v4 API as the single entry point for model compilation.

This cleanup eliminates ~50 LOC of dead/unused code and reduces API confusion by providing a single clear entry point for ONNX model compilation.

## Problem

**Current design:**
```cpp
// morphizen-core/include/morphizen/onnxruntime_morphizen_ep.hpp

// Legacy API - DEAD CODE (no callers)
std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_with_options(
    const std::string& model_path,
    const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options);

// Legacy API - UNUSED (only called by test)
std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_with_error_handling(
    const std::string& model_path,
    const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options,
    void* status,
    void (*func)(void*, int, const char*));

// Current API - IN USE (production code)
std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_v4(
    const std::string& model_path,
    const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options,
    void* status,
    void (*func)(void*, int, const char*),
    const OrtLogger* logger);  // ← Key difference: logger integration
```

**Why this is wrong:**
1. **API Confusion** - Three entry points with similar names but unclear purpose:
   - Documentation says `with_error_handling` is fallback "If not implements"
   - But v4 is the actual current API used in production
   - No clear deprecation markers or guidance on which to use

2. **Dead Code** - `compile_onnx_model_morphizen_ep_with_options`:
   - Not called by ANY production code or tests
   - Still exported from DLLs (`.def` files)
   - Still documented as valid entry point

3. **Unused Legacy Code** - `compile_onnx_model_morphizen_ep_with_error_handling`:
   - Only called by test code (`ort-bridge/test/src/test-compile-model.cpp:68`)
   - NOT exported from DLLs (not in `.def` files)
   - Test should use current v4 API instead

4. **Maintenance Burden** - Three implementations doing essentially the same thing:
   - All three call same internal `compile_onnx_model_3()` function
   - Must update all when changing compilation logic
   - Export different functionality with minor differences

**Current behavior:**
- Production code (`ort-bridge/src/morphizen-ep.cpp:325`) calls v4 API directly
- Test code uses deprecated `with_error_handling` API
- `with_options` API exists but is completely unused

## Solution

**Proposed design:**
```cpp
// morphizen-core/include/morphizen/onnxruntime_morphizen_ep.hpp

// Single, clear entry point for model compilation
std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_v4(
    const std::string& model_path,
    const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options,
    void* status,
    void (*func)(void*, int, const char*),
    const OrtLogger* logger);

// Legacy APIs removed:
// - compile_onnx_model_morphizen_ep_with_options (DELETED)
// - compile_onnx_model_morphizen_ep_with_error_handling (DELETED)
```

**Approach:**
1. **Migrate test** - Update `ort-bridge/test/src/test-compile-model.cpp` to use v4 API (add `nullptr` for logger parameter)
2. **Remove declarations** - Delete legacy function declarations from header
3. **Remove implementations** - Delete legacy function implementations from `.cpp` file
4. **Remove exports** - Delete legacy exports from both `.def` files
5. **Update comments** - Remove references to fallback chain, clarify v4 is THE entry point
6. **Verify** - Build and run tests to ensure migration successful

**Benefits:**
- ✅ **Single API** - One clear entry point (`compile_onnx_model_morphizen_ep_v4`)
- ✅ **No Confusion** - Developers don't need to guess which API to use
- ✅ **Less Maintenance** - Single implementation to maintain
- ✅ **Code Reduction** - ~50 LOC removed (header declarations, implementations, exports)
- ✅ **Cleaner DLL Interface** - Single compile function exported

**Migration path:**
- No external migration needed (only v4 is used in production)
- Single test migration: `with_error_handling` → `v4` (add `nullptr` parameter)

## Evidence

**Production usage (only v4 is called):**
- `ort-bridge/src/morphizen-ep.cpp:325` - Calls `compile_onnx_model_morphizen_ep_v4` in production

**Test usage (legacy API):**
- `ort-bridge/test/src/test-compile-model.cpp:68` - Calls `with_error_handling` (should migrate to v4)

**Dead code (no callers):**
- `compile_onnx_model_morphizen_ep_with_options` - NO CALLERS FOUND

**Implementations to remove:**
- `morphizen-core/include/morphizen/onnxruntime_morphizen_ep.hpp:120-123` - `with_error_handling` declaration
- `morphizen-core/include/morphizen/onnxruntime_morphizen_ep.hpp:141-144` - `with_options` declaration
- `morphizen-core/src/onnxruntime_morphizen_ep.cpp:256-262` - `with_options` implementation
- `morphizen-core/src/onnxruntime_morphizen_ep.cpp:266-279` - `with_error_handling` implementation

**DLL exports to remove:**
- `morphizen-core/onnxruntime_morphizen_ep.def:3` - `compile_onnx_model_morphizen_ep_with_options`
- `morphizen-core/onnxruntime_morphizen_ep_with_ort_bridge.def:3` - `compile_onnx_model_morphizen_ep_with_options`

**Evidence from VitisAI Provider (MorphiZen's ancestor):**

File: `C:\Develop\m\source\onnxruntime\onnxruntime\core\providers\vitisai\imp\global_api.cc:187-210`

The VitisAI provider shows a fallback chain that only uses v4 in practice:
```cpp
if (s_library_vitisaiep.compile_onnx_model_vitisai_ep_v4) {
    // Use v4 API (with logger integration)
} else if (s_library_vitisaiep.compile_onnx_model_vitisai_ep_v3) {
    // Fallback to v3 (OLD DLL)
} else if (s_library_vitisaiep.compile_onnx_model_vitisai_ep_with_error_handling) {
    // Fallback to error handling version (OLDER DLL)
} else if (s_library_vitisaiep.compile_onnx_model_vitisai_ep_with_options) {
    // Fallback to basic version (OLDEST DLL)
}
```

This confirms the legacy APIs exist purely for backwards compatibility with ancient VitisAI DLLs. MorphiZen doesn't need this fallback chain.

## Context

This issue follows the same pattern as recent legacy code cleanup issues:
- **Issue #006**: Remove cache_dir (~220 LOC) - Legacy disk-based cache system
- **Issue #022**: Remove fix_info (~73 LOC) - Unused proto field and API methods
- **Issue #010**: Remove cache_files (~10 LOC) - Dead proto field

All involve removing legacy/dead code for cleaner architecture.

**Why this cleanup matters:**

1. **Confusion Reduction** - Three entry points with unclear purpose creates confusion
2. **API Clarity** - Single entry point makes it obvious which API to use
3. **Maintenance Burden** - Multiple entry points mean updating all when changing compilation logic
4. **Consistency** - Aligns with project cleanup goals (Issues #006, #010, #022)

**Code reduction estimate:**
- Header declarations: ~25 lines
- Implementation: ~25 lines
- DLL exports: 2 lines
- **Total removed: ~50 LOC**
- Test migration: ~10 lines modified
- **Net reduction: ~45 LOC**

## Implementation Steps

### Step 1: Update Test to Use v4 API

**File:** `ort-bridge/test/src/test-compile-model.cpp:68-74`

**Before:**
```cpp
compile_onnx_model_morphizen_ep_with_error_handling(
    model_path.u8string(), graph, provider_options, (void*)&status,
    [](void* status, int code, const char* error_message) {
      OrtStatus** ort_status = static_cast<OrtStatus**>(status);
      *ort_status = Ort::GetApi().CreateStatus((OrtErrorCode)code, error_message);
    })
```

**After:**
```cpp
compile_onnx_model_morphizen_ep_v4(
    model_path.u8string(), graph, provider_options, (void*)&status,
    [](void* status, int code, const char* error_message) {
      OrtStatus** ort_status = static_cast<OrtStatus**>(status);
      *ort_status = Ort::GetApi().CreateStatus((OrtErrorCode)code, error_message);
    },
    nullptr)  // Add nullptr for OrtLogger (test doesn't need logging)
```

### Step 2: Remove Declarations from Header

**File:** `morphizen-core/include/morphizen/onnxruntime_morphizen_ep.hpp`

**Remove lines 118-123** (`with_error_handling` declaration + comment):
```cpp
/**
 * @brief Compiles an ONNX model using error handling callback.
 * ...
 */
MORPHIZEN_DLL_SPEC
std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_with_error_handling(...);
```

**Remove lines 125-144** (`with_options` declaration + comment):
```cpp
/**
 * @brief Compiles an ONNX model using the MorphiZen Execution Provider with
 * specified options.
 *
 * If compile_onnx_model_morphizen_ep_with_error_handing not implements, will
 * call this function. Not throw ONNXRuntime Error when compile ONNX model
 * error.
 * ...
 */
MORPHIZEN_DLL_SPEC std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_with_options(...);
```

**Update comment for v4 API** (lines 146-153): Remove reference to fallback chain, clarify this is THE compile entry point.

### Step 3: Remove Implementations

**File:** `morphizen-core/src/onnxruntime_morphizen_ep.cpp`

**Remove lines 254-262** (`with_options` implementation):
```cpp
MORPHIZEN_DLL_SPEC
std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_with_options(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options) {
  update_log_level(options);
  return new std::vector<std::unique_ptr<morphizen::ExecutionProvider>>(
      morphizen::compile_onnx_model_3(model_path, graph, options, nullptr));
}
```

**Remove lines 264-279** (`with_error_handling` implementation):
```cpp
MORPHIZEN_DLL_SPEC
std::vector<std::unique_ptr<morphizen::ExecutionProvider>>*
compile_onnx_model_morphizen_ep_with_error_handling(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options, [[maybe_unused]] void* status,
    [[maybe_unused]] void (*func)(void*, int, const char*)) {
  update_log_level(options);
  auto set_ort_status = [&](int error_code, const char* error_message) {
    if (func != nullptr) {
      func(status, error_code, error_message);
    }
  };
  return new std::vector<std::unique_ptr<morphizen::ExecutionProvider>>(
      morphizen::compile_onnx_model_3(model_path, graph, options,
                                      set_ort_status));
}
```

### Step 4: Remove DLL Exports

**File:** `morphizen-core/onnxruntime_morphizen_ep.def`

**Before:**
```
LIBRARY onnxruntime_morphizen_ep
EXPORTS
    compile_onnx_model_morphizen_ep_with_options
    compile_onnx_model_morphizen_ep_v4
    create_ep_context_nodes
    ...
```

**After:**
```
LIBRARY onnxruntime_morphizen_ep
EXPORTS
    compile_onnx_model_morphizen_ep_v4
    create_ep_context_nodes
    ...
```

**File:** `morphizen-core/onnxruntime_morphizen_ep_with_ort_bridge.def`

Same change - remove `compile_onnx_model_morphizen_ep_with_options` export.

### Step 5: Verify Compilation

```bash
# Build the project
cmake --build ../../build/$(basename $PWD) --config Debug --parallel

# Run tests to verify test migration works
../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe
```

### Step 6: Optional - Rename v4 API

**Consider removing "v4" suffix since it's now the only version:**

**Before:**
```cpp
compile_onnx_model_morphizen_ep_v4(...)
```

**After:**
```cpp
compile_onnx_model_morphizen_ep(...)
```

This is optional - "v4" can stay for backwards compatibility with external code that might dynamically load the function by name.

## Acceptance Criteria

- [ ] `compile_onnx_model_morphizen_ep_with_options` removed from header
- [ ] `compile_onnx_model_morphizen_ep_with_error_handling` removed from header
- [ ] `compile_onnx_model_morphizen_ep_with_options` removed from implementation
- [ ] `compile_onnx_model_morphizen_ep_with_error_handling` removed from implementation
- [ ] Both `.def` files updated to remove legacy exports
- [ ] Test migrated to use v4 API
- [ ] All tests pass
- [ ] Project compiles without errors
- [ ] Grep confirms no remaining references to removed functions (except in docs/commits)
- [ ] ~45-50 LOC net reduction

## Verification Commands

```bash
# Search for remaining usage (should return empty after cleanup)
grep -r "compile_onnx_model_morphizen_ep_with_options" morphizen-core/ ort-bridge/ --include="*.cpp" --include="*.hpp"
grep -r "compile_onnx_model_morphizen_ep_with_error_handling" morphizen-core/ ort-bridge/ --include="*.cpp" --include="*.hpp"

# Verify only v4 API remains
grep -r "compile_onnx_model_morphizen_ep" morphizen-core/include/morphizen/onnxruntime_morphizen_ep.hpp

# Build and test
cmake --build ../../build/$(basename $PWD) --config Debug --parallel
../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe
```

## Notes

**Optional consideration:** After removing legacy APIs, `compile_onnx_model_morphizen_ep_v4` could be renamed to `compile_onnx_model_morphizen_ep` (drop the "v4" suffix) since it's the only version. However, keeping "v4" is fine for backwards compatibility with code that dynamically loads the function by name.
