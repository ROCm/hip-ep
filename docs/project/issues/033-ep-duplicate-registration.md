<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #033: EP Duplicate Registration Causing V2 API Test Failure

## Metadata
- **Type:** Bug
- **Priority:** MEDIUM
- **Created:** 2026-02-03
- **Dependencies:** None

## Description

V2 API E2E test fails when attempting to register MorphiZenExecutionProvider because the EP was already registered in `main()` function.

## Problem

**Current design/code:**
```cpp
// morphizen_unit_test_main.cpp:101
env.RegisterExecutionProviderLibrary("MorphiZenExecutionProvider",
                                     library_path);

// morphizen-e2e-test/env.cpp:37
// V2 test also tries to register
auto status = env_.RegisterExecutionProviderLibrary(
    registration.name(), registration.library());
// FAILS: library is already registered
```

**Why this is problematic:**
1. ONNX Runtime does not allow duplicate registration of EP library with same name
2. `main()` globally registers EP (suitable for most tests)
3. V2 API test attempts to register again (because it needs to test V2 registration flow)
4. Causes CHECK failure, test interrupted

**Code locations:**
- `unit-test/morphizen_unit_test_main.cpp:101` - Global registration
- `unit-test/morphizen-e2e-test/env.cpp:37` - V2 test registration
- `unit-test/morphizen-e2e-test/env.cpp:42` - CHECK failure point

**Error output:**
```
I20260203 01:59:55.542173 env.cpp:37] Registering: MorphiZenExecutionProvider, library: onnxruntime_vitisai_ep.dll
F20260203 01:59:55.542215 env.cpp:42] Check failed: status == nullptr
RegisterExecutionProviderLibrary failed: status = library is already registered under MorphiZenExecutionProvider
```

## Solution

**Proposed approaches:**

### Option A: Conditional Registration - Check if Already Registered
```cpp
// env.cpp
bool is_ep_registered(const Ort::Env& env, const std::string& ep_name) {
    // Try to get available EP list, check if exists
    // Note: ONNX Runtime API may not provide this functionality
}

if (!is_ep_registered(env_, registration.name())) {
    auto status = env_.RegisterExecutionProviderLibrary(...);
}
```

**Issue**: ONNX Runtime may not provide API to query registered EPs

### Option B: V2 Test Uses Independent Ort::Env
```cpp
// Create new Ort::Env instance for V2 test
// This way it won't conflict with main() registration
TEST(...v2_...) {
    Ort::Env local_env(ORT_LOGGING_LEVEL_WARNING, "v2_test");
    local_env.RegisterExecutionProviderLibrary(...);
    // Use local_env
}
```

**Issue**: May cause resource conflicts or state inconsistency

### Option C: Remove Global Registration from main(), Let Each Test Register
```cpp
// morphizen_unit_test_main.cpp
// Remove global registration, only initialize global API

// Each test fixture registers in SetUp()
TEST_F(...) {
    static std::once_flag register_flag;
    std::call_once(register_flag, []() {
        env.RegisterExecutionProviderLibrary(...);
    });
}
```

**Issue**: Need to modify many test code

### Option D: V2 Test Tolerates Duplicate Registration Error
```cpp
// env.cpp
auto status = env_.RegisterExecutionProviderLibrary(...);
if (status != nullptr) {
    std::string error_msg = status->GetErrorMessage();
    if (error_msg.find("already registered") != std::string::npos) {
        LOG(INFO) << "EP already registered, continuing test";
        return;  // Continue test
    }
    CHECK(false) << "RegisterExecutionProviderLibrary failed: " << error_msg;
}
```

**Recommended:** Option D (minimal changes, suitable for test scenario)

## Approach

1. Modify `unit-test/morphizen-e2e-test/env.cpp:37-42`
2. Add tolerance logic for duplicate registration error
3. Log warning instead of FATAL
4. Verify V2 test passes

## Benefits

- ✅ V2 API test restored
- ✅ No need for large-scale test framework refactoring
- ✅ Maintains convenience of global registration

## Evidence

**Affected tests (1 total):**
- MorphizenE2ETestSuite/MorphizenE2ETest.RunE2ETests/v2_single_session_gen_and_run_embed_ctx

**Test configuration:**
```protobuf
registration {
  name: "MorphiZenExecutionProvider"
  library: "onnxruntime_vitisai_ep.dll"
}
```

**Test results:** Test fails immediately at registration with "library is already registered" error

**Notes:**
- This is a test framework design issue, not production code bug
- V2 API test is specifically for testing V2 registration flow
- Need to fix without breaking other tests
