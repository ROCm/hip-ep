<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #035: ConstDataTest Series Tests Skipped - Missing Boost::Process Support

## Metadata
- **Type:** Feature
- **Priority:** LOW
- **Dependencies:** None

## Description

16 ConstDataTest tests require Boost::Process to dynamically generate test models, skipped when `morphizen_ENABLE_BOOST=OFF`.

## Problem

**Current design/code:**
```cpp
// test_const_data.cpp
template <typename F> void run_test(int line, F check) {
#ifdef MORPHIZEN_ENABLE_BOOST
    // Use Python script to generate test model
    auto exit_code = boost::process::system(
        PYTHON_EXE.u8string(),
        (TEST_SRC_DIR / "morphizen" / "test_constant_initializer.py").u8string(),
        test_constant_initializer_onnx.u8string());
    // ... load and test
#else
    GTEST_SKIP() << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
#endif
}
```

**Why this is problematic:**
1. Tests depend on external Python script to dynamically generate models
2. Requires Boost::Process to invoke Python
3. When Boost unavailable, tests completely skipped
4. Reduces test coverage

**Code locations:**
- `unit-test/morphizen/test_const_data.cpp:21-44` - Test template
- `unit-test/morphizen/test_constant_initializer.py` - Python generation script
- `cmake/morphizen_options.cmake` - BOOST switch

**Affected tests (16 total):**
- ConstDataTest.int8_scalar, ConstDataTest.int8
- ConstDataTest.uint8_scalar, ConstDataTest.uint8
- ConstDataTest.int16_scalar, ConstDataTest.int16
- ConstDataTest.uint16_scalar, ConstDataTest.uint16
- ConstDataTest.int32_scalar, ConstDataTest.int32
- ConstDataTest.uint32_scalar, ConstDataTest.uint32
- ConstDataTest.int64_scalar, ConstDataTest.int64
- ConstDataTest.uint64_scalar, ConstDataTest.uint64

## Solution

**Proposed approaches:**

### Option A: Enable Boost::Process
```cmake
# CMakeLists.txt or cmake configuration
set(morphizen_ENABLE_BOOST ON)
```

**Pros**:
- Simplest, immediately enables all tests
- Can use other Boost features

**Cons**:
- Increases dependency complexity
- Boost is large library, long compilation time
- May not suit lightweight builds

### Option B: Pre-generate Test Model Files
```bash
# Run once in development environment
python test_constant_initializer.py test_int8_scalar.onnx --type int8 --scalar
python test_constant_initializer.py test_int8.onnx --type int8
# ... generate models for all 16 tests

# Commit generated .onnx files to repository
```

```cpp
// test_const_data.cpp
template <typename F> void run_test(const char* model_file, F check) {
    auto test_model_path = TEST_SRC_DIR / "morphizen" / "test_data" / model_file;
    ASSERT_TRUE(std::filesystem::exists(test_model_path));
    auto model = morphizen_cxx::Model::load(test_model_path);
    // ... test
}
```

**Pros**:
- No Boost dependency
- Faster test startup (no dynamic generation)
- Suitable for CI environment

**Cons**:
- Need to maintain binary files
- Need to regenerate when model format changes

### Option C: Use CMake execute_process Instead of Boost::Process
```cmake
# CMakeLists.txt
if(morphizen_ENABLE_UNIT_TEST)
    # Generate test models at configuration time
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_constant_initializer.py
                ${CMAKE_CURRENT_BINARY_DIR}/test_int8_scalar.onnx
                --type int8 --scalar
        RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
        message(WARNING "Failed to generate test model")
    endif()
endif()
```

```cpp
// test_const_data.cpp - simplify to direct loading
template <typename F> void run_test(const char* model_file, F check) {
    auto test_model_path = TEST_CWD / model_file;
    ASSERT_TRUE(std::filesystem::exists(test_model_path))
        << "Test model not generated: " << model_file;
    auto model = morphizen_cxx::Model::load(test_model_path);
    // ... test
}
```

**Pros**:
- No Boost dependency
- Automatic generation at configuration time
- Simplified test code

**Cons**:
- Increased configuration time
- Requires Python availability

**Recommended:** Option B (short-term) + Option C (long-term)

## Approach

### Short-term (Option B):
1. Run Python script in development environment to generate all 16 test models
2. Commit `.onnx` files to `unit-test/morphizen/test_data/`
3. Modify `test_const_data.cpp` to directly load pre-generated files
4. Remove Boost::Process dependency check

### Long-term (Option C):
1. Add model generation step in CMakeLists.txt
2. Automatically generate required test models at configuration time
3. Load directly at test runtime
4. Add Python dependency check

## Benefits

- ✅ 16 ConstDataTest tests restored
- ✅ Reduce or eliminate Boost dependency
- ✅ Improve test coverage
- ✅ Simpler CI builds

## Evidence

**Test output:**
```
22/85 Test #24: ConstDataTest.int8_scalar ...Skipped   0.06 sec
23/85 Test #25: ConstDataTest.int8 ..........Skipped   0.06 sec
... (16 tests skipped)
```

**Skip reason:**
```cpp
GTEST_SKIP() << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
```

**Notes:**
- These tests specifically verify constant initializer functionality for various data types
- Test coverage includes int8/16/32/64, uint8/16/32/64 in scalar and tensor forms
- Python script `test_constant_initializer.py` already exists and can be reused
