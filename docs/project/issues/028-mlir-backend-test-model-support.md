<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #028: Provide ResNet50 MLIR Test Model for MLIR Backend

## Metadata
- **Type:** Feature
- **Priority:** HIGH
- **Created:** 2026-02-03
- **Dependencies:** None

## Description

Current unit tests use `pt_resnet50.onnx` (ONNX format) as test model, but MLIR backend expects MLIR format (`.mlir` or `.mlirbc`) model files. This causes 23 unit tests to fail when using MLIR backend.

## Problem

**Current design/code:**
```cpp
// test_environment.hpp
static const std::filesystem::path RESNET_50_PATH = TEST_CWD / "pt_resnet50.onnx";

// All tests use this ONNX file
auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
```

**Why this is problematic:**
1. MLIR backend's `Model::load()` expects MLIR format files, not ONNX format
2. Attempting to load ONNX files causes parsing error: `error: unexpected character`
3. Results in access violation (SEH exception 0xc0000005)
4. Affects 23 core test cases covering ModelTest, GraphTest, PatternTest, PassContextConfigTest

**Code locations:**
- `unit-test/test_environment.hpp:33` - RESNET_50_PATH definition
- `mlir-imp/src/mlir-model.cpp:108` - MLIR model loading failure
- `unit-test/morphizen/test_model.cpp` - 5 failing tests
- `unit-test/morphizen/test_graph.cpp` - 11 failing tests
- `unit-test/morphizen/test_pattern.cpp` - 2 failing tests
- `unit-test/morphizen/test_pass_context.cpp` - 5 failing tests

**Error output:**
```
loc("D:/ROCm/build/MorphiZen/unit-test\\pt_resnet50.onnx":1:1): error: unexpected character
E20260203 01:59:47.988436 mlir-model.cpp:108] Failed to load MLIR module from: D:/ROCm/build/MorphiZen/unit-test\pt_resnet50.onnx
unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.
```

## Solution

**Proposed approaches:**

### Option A: Provide Pre-generated MLIR Model File
```bash
# Use existing tools to convert ONNX to MLIR
# Add pt_resnet50.mlir or pt_resnet50.mlirbc to test data
```

**Pros**:
- Simplest and most direct
- Fast test startup

**Cons**:
- Need to maintain additional binary files
- May have compatibility issues with MLIR version changes

### Option B: Dynamic Conversion at Test Initialization
```cpp
// In test main() or test fixture SetUp()
if (using_mlir_backend) {
    if (!std::filesystem::exists("pt_resnet50.mlir")) {
        // Call ONNX→MLIR conversion tool
        convert_onnx_to_mlir("pt_resnet50.onnx", "pt_resnet50.mlir");
    }
}
```

**Pros**:
- Always uses latest conversion logic
- No need to maintain MLIR files

**Cons**:
- Increased test startup time
- Requires conversion tool support

### Option C: Backend-Specific Model Selection
```cpp
// test_environment.hpp
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
static const std::filesystem::path RESNET_50_PATH = TEST_CWD / "pt_resnet50.mlir";
#else
static const std::filesystem::path RESNET_50_PATH = TEST_CWD / "pt_resnet50.onnx";
#endif
```

**Pros**:
- Each backend uses its native format
- Compile-time decision, no runtime overhead

**Cons**:
- Need to maintain two sets of model files
- May lead to test behavior differences

**Recommended:** Option C (short-term) + Option B (long-term)

## Approach

1. Short-term: Add conditional compilation to select model based on backend
2. Provide `pt_resnet50.mlir` or `pt_resnet50.mlirbc` file
3. Update test environment to use appropriate model
4. Verify 23 tests pass with MLIR backend

## Benefits

- ✅ 23 core unit tests restored
- ✅ Improved MLIR backend test coverage
- ✅ Better CI/CD test reliability
- ✅ Foundation for adding more MLIR test models

## Evidence

**Affected tests (23 total):**

**ModelTest (5 tests):**
- ModelTest.Load
- ModelTest.Clone
- ModelTest.MainGraph
- ModelTest.SetAndGetMetadata
- ModelTest.ImplicitConversion

**GraphTest (11 tests):**
- GraphTest.LoadAndSave
- GraphTest.CloneAndSave
- GraphTest.FindNodeArgGraphInput
- GraphTest.FindNodeArgGraphOutput
- GraphTest.NodesInTopologicalOrder
- GraphTest.NodeIndex
- GraphTest.FindConsumers
- GraphTest.NodeArgFindProducer
- GraphTest.Fuse
- GraphTest.TryFuse
- GraphTest.VirtualFuse

**PatternTest (2 tests):**
- PatternTest.CommutableNode
- PatternTest.LoadSaveBinary

**PassContextConfigTest (5 tests):**
- PassContextConfigTest.Config
- PassContextConfigTest.ProviderOptions
- PassContextConfigTest.TargetSpecifiedByEndUserNotValid
- PassContextConfigTest.TargetSpecifiedByEndUserValid
- PassContextConfigTest.TargetInConfigFileNotValidTarget
- PassContextConfigTest.TargetInConfigFileValidTarget

**Test results:** 23/85 tests fail with "Failed to load MLIR module" error
