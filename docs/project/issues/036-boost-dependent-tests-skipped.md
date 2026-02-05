<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #036: Other Boost-Dependent Tests Skipped

## Metadata
- **Type:** Feature
- **Priority:** LOW
- **Dependencies:** None

## Description

GraphTest.NewConstantInitializer and TarFileTest.WriteTo depend on Boost::Process for external validation, skipped when `morphizen_ENABLE_BOOST=OFF`.

## Problem

**Current design/code:**

### GraphTest.NewConstantInitializer
```cpp
// test_graph.cpp:538-541
#ifdef MORPHIZEN_ENABLE_BOOST
    auto exit_code =
        boost::process::system(PYTHON_EXE.u8string(), "-c", python_code.str());
    EXPECT_EQ(exit_code, 0) << "onnx.checker.check_model failed";
#else
    GTEST_SKIP() << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
#endif
```

### TarFileTest.WriteTo
```cpp
// test_tar_file.cpp:349-351
#ifdef MORPHIZEN_ENABLE_BOOST
    auto exit_code = boost::process::system(
        tar_exe_path, "-tvf", tarFileName.filename().u8string(),
        boost::process::start_dir(CMAKE_CURRENT_BINARY_PATH.u8string()));
    ASSERT_EQ(exit_code, 0) << "Failed to run tar command. Exit code: " << exit_code;
#else
    GTEST_SKIP() << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
#endif
```

**Why this is problematic:**
1. **GraphTest.NewConstantInitializer**: Uses Python's `onnx.checker` to validate generated ONNX model
2. **TarFileTest.WriteTo**: Uses system `tar` command to verify generated tar file format
3. Both tests need to invoke external tools, depending on Boost::Process
4. When Boost unavailable, tests skipped, reducing test coverage

**Code locations:**
- `unit-test/morphizen/test_graph.cpp:538` - GraphTest skip
- `unit-test/morphizen/test_tar_file.cpp:349` - TarFileTest skip

**Affected tests (2 total):**
- GraphTest.NewConstantInitializer
- TarFileTest.WriteTo

## Solution

**Proposed approaches:**

### Option A: Enable Boost::Process
```cmake
set(morphizen_ENABLE_BOOST ON)
```

**Pros**: Simple and direct
**Cons**: Increases dependency, see Issue #035

### Option B: Use CMake execute_process as Replacement

#### For GraphTest.NewConstantInitializer:
```cpp
// test_graph.cpp
#ifdef MORPHIZEN_ENABLE_BOOST
    // Use Boost::Process
#else
    // Use std::system (not recommended) or skip validation step
    LOG(WARNING) << "ONNX validation skipped (Boost not available)";
    // Still save file, but don't validate
#endif
```

Or use ONNX C++ API directly for validation:
```cpp
// No need to invoke Python
#include <onnx/checker.h>
onnx::checker::check_model(model_proto);  // Directly use ONNX C++ API
```

#### For TarFileTest.WriteTo:
```cpp
// test_tar_file.cpp
#ifdef MORPHIZEN_ENABLE_BOOST
    // Use system tar command for validation
#else
    // Use our own TarFile::read() for validation
    auto tar = TarFile::open(tarFileName);
    ASSERT_NE(tar, nullptr);
    // Verify file list
    auto entries = tar->entries();
    ASSERT_EQ(entries.size(), expected_count);
#endif
```

### Option C: Conditional Compilation - Keep Partial Functionality
```cpp
// Main test functionality doesn't depend on Boost, only validation step depends
// Even without Boost, execute main test, only skip external validation
```

**Recommended:** Option B

## Approach

### For GraphTest.NewConstantInitializer:
1. Investigate if ONNX C++ API can replace Python checker
2. If yes, add ONNX C++ checker invocation
3. If no, add WARNING log, skip validation but keep test

### For TarFileTest.WriteTo:
1. Use our own `TarFile::open()` to read back tar file
2. Verify file list and contents
3. Remove dependency on system `tar` command

## Benefits

- ✅ 2 tests restored (at least partial functionality)
- ✅ Reduce external dependencies
- ✅ Using internal API more reliable
- ✅ Improve test independence

## Evidence

**Test output:**
```
22/85 Test #22: GraphTest.NewConstantInitializer .....Skipped   0.05 sec
69/85 Test #69: TarFileTest.WriteTo ..................Skipped   0.06 sec
```

**Skip reasons:**
- GraphTest: Python + onnx.checker validation requires Boost::Process
- TarFileTest: System tar command validation requires Boost::Process

**Notes:**
- GraphTest.NewConstantInitializer mainly tests adding constant initializer functionality, validation is additional step
- TarFileTest.WriteTo mainly tests writing tar files, system tar validation is just extra confirmation
- Core functionality of both tests doesn't depend on Boost, can use alternative validation approaches

**Related:**
- See Issue #035 for ConstDataTest series (similar Boost dependency issue)
- Consider enabling Boost globally vs. finding alternatives for each test
