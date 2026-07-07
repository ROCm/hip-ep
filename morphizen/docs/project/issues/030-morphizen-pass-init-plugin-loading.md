<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #030: morphizen-pass_init Plugin Loading Failure

## Metadata
- **Type:** Bug
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Runtime cannot find `morphizen-pass_init` plugin, causing 7 tests to fail. This may be due to incorrect plugin registration or path configuration after incremental compilation.

## Problem

**Current design/code:**
```cpp
// pass_imp.cpp:399
auto plugin = morphizen::Plugin::get("morphizen-pass_init");
if (!plugin) {
    LOG(FATAL) << "cannot find plugin: morphizen-pass_init "
               << "enable env MORPHIZEN_DEBUG_PLUGIN=1 to see more details";
}
```

**Why this is problematic:**
1. Plugin system cannot find `morphizen-pass_init` plugin at runtime
2. Possible causes:
   - Plugin DLL/SO not correctly linked
   - Plugin registration code not executed (static initialization issue)
   - Plugin path configuration error
   - Symbol not updated after incremental compilation
3. Affects all tests depending on Pass system

**Code locations:**
- `morphizen-core/src/pass_imp.cpp:399` - Plugin loading failure point
- `morphizen-pass-init/` - Plugin implementation directory
- `unit-test/CMakeLists.txt:67` - Link configuration `WHOLE_ARCHIVE,morphizen-pass-init`

**Error output:**
```
F20260203 01:59:51.542456 pass_imp.cpp:399] cannot find plugin: morphizen-pass_init enable env MORPHIZEN_DEBUG_PLUGIN=1 to see more details
```

## Solution

**Investigation steps:**

1. **Enable Debug Output**
```bash
set MORPHIZEN_DEBUG_PLUGIN=1
ctest --test-dir . -C Release -R TestAnchorPoint.Case0 --output-on-failure
```

2. **Check Link Configuration**
```cmake
# unit-test/CMakeLists.txt
target_link_libraries(${TEST_EXE_NAME}
    PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,morphizen-pass-init>  # Verify exists
)
```

3. **Verify Plugin Registration**
```cpp
// morphizen-pass-init/ should have static registration code
static auto register_plugin = []() {
    morphizen::Plugin::register_plugin("morphizen-pass_init", ...);
    return 0;
}();
```

### Possible Fix Approaches:

**Option A: Ensure Static Initialization Executes**
- Check static initialization code in `morphizen-pass-init`
- Confirm link options correctly use WHOLE_ARCHIVE

**Option B: Explicit Plugin Initialization**
```cpp
// morphizen_unit_test_main.cpp
extern "C" void morphizen_pass_init_register();

int main() {
    morphizen_pass_init_register();  // Explicit registration call
    // ...
}
```

**Option C: Check Incremental Compilation Issues**
```bash
# Clean rebuild morphizen-pass-init
cmake --build . --target morphizen-pass-init --clean-first
cmake --build . --target morphizen-unit-tests
```

## Approach

1. Run `MORPHIZEN_DEBUG_PLUGIN=1` to see detailed logs
2. Check `morphizen-pass-init` registration mechanism
3. Verify link options and library dependencies
4. Implement corresponding fix based on investigation
5. Verify 7 tests pass

## Benefits

- ✅ 7 Pass system dependent tests restored
- ✅ More robust plugin system
- ✅ Better error diagnostics

## Evidence

**Affected tests (7 total):**
- TestAnchorPoint.Case0
- TestAnchorPoint.Case1
- TestAnchorPoint.Case2
- TestAnchorPoint.Case3
- TestAnchorPoint.Case4
- TestAnchorPoint.Append
- MorphizenOrtApiTest.TestAll (partial failure - Test21)

**Test results:** All 7 tests fail with "cannot find plugin: morphizen-pass_init"

**Notes:**
- This issue may be related to incremental compilation
- Need to investigate static initialization order issues
- May need to reference plugin registration approach in `onnx-ir-imp` or `mlir-imp`
