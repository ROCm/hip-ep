<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #031: Target Auto-Discovery Failure Causing Compilation Errors

## Metadata
- **Type:** Bug
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Target auto-discovery fails during model compilation with empty target string, unable to find valid target configuration. Affects 2 test cases.

## Problem

**Current design/code:**
```cpp
// pass_context_imp.cpp:1167
LOG(ERROR) << "Target auto-discovery: target proto not found for target: "
           << target << ", valid target names: ";
```

**Why this is problematic:**
1. Target string is empty (`""`), causing lookup failure
2. Error message shows `valid target names:` followed by empty list, indicating no available target configurations
3. Possible causes:
   - Target configuration file not loaded
   - Provider options missing target information
   - Target auto-discovery logic failure
4. Causes model compilation failure, preventing test continuation

**Code locations:**
- `morphizen-core/src/pass_context_imp.cpp:1167` - Error logging point
- `ort-bridge/test/test-hello-ep.cpp:91` - HelloEpTest.CreateSession test
- `ort-bridge/test/test-compile-model.cpp` - CompileModel.T0 test

**Error output:**
```
E20260203 01:59:56.078714 pass_context_imp.cpp:1167] Target auto-discovery: target proto not found for target: , valid target names:
2026-02-03 01:59:56.0789238 [E:onnxruntime:, pass_context_imp.cpp:1167] Target auto-discovery: target proto not found for target: , valid target names:
```

## Solution

**Investigation steps:**

1. **Check target configuration loading**
```cpp
// Verify target configuration file is correctly loaded
// Check config.json or vaip_config.json location
```

2. **Check provider_options**
```cpp
// Does test provide target parameter?
onnxruntime::ProviderOptions options = {
    {"target", "CPU"},  // Does this exist?
    // ...
};
```

3. **Check default target logic**
```cpp
// pass_context_imp.cpp
// If user doesn't specify target, is there a reasonable default?
```

### Possible Fix Approaches:

**Option A: Explicitly Provide Target in Tests**
```cpp
// test-hello-ep.cpp
onnxruntime::ProviderOptions provider_options = {
    {"target", "CPU"},  // Or other valid target
    {"config_file", "/path/to/config.json"},
};
session_options.AppendExecutionProvider("MorphiZenExecutionProvider",
                                       provider_options);
```

**Option B: Provide Default Target Configuration**
```cpp
// pass_context_imp.cpp
std::string get_default_target() {
    // If target not specified, return "CPU" or other reasonable default
    return "CPU";
}
```

**Option C: Ensure Target Configuration File Available**
```cmake
# CMakeLists.txt
# Ensure config.json is findable at test runtime
configure_file(
    ${CMAKE_SOURCE_DIR}/config.json
    ${CMAKE_BINARY_DIR}/config.json
    COPYONLY
)
```

**Recommended:** Option A + Option B (short-term) + Option C (long-term)

## Approach

1. Investigate current target configuration loading mechanism
2. Check provider_options settings in tests
3. Add reasonable default target value
4. Ensure available target configuration in test environment
5. Verify 2 tests pass

## Benefits

- ✅ 2 compilation tests restored
- ✅ Better error messages (display available target list)
- ✅ More robust target auto-discovery logic

## Evidence

**Affected tests (2 total):**
- HelloEpTest.CreateSession
- CompileModel.T0

**Test results:** Both tests fail with "target proto not found for target: " error

**Observations:**
- Target string is empty
- `valid target names:` list is also empty, indicating no target configurations loaded
- This may be a configuration file path issue or initialization order problem
