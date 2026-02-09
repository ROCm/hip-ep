<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * @file test-runner-example.md
 * @brief Example of how to run the comprehensive MorphiZen ORT API tests
 *
 * This file demonstrates how to run and analyze the comprehensive test suite
 * for the MorphiZen ORT API implementation.
 */

# MorphiZen ORT API Test Suite

## Overview

The comprehensive test suite includes the following test classes and files:

1. **`TestCoverageWrapperTest`** - Base test class with coverage wrapper setup
2. **`MorphizenOrtApiTest`** - Comprehensive API tests derived from the base class
3. **Coverage Analysis Tools** - Helpers for analyzing API coverage

## Test Categories

### Model API Tests
- `ModelLoadAndDelete` - Tests model loading and deletion operations
- `ModelMetaDataOperations` - Tests model metadata operations and cloning

### Graph API Tests
- `GraphBasicOperations` - Tests basic graph operations (name, nodes, I/O)
- `GraphAdvancedOperations` - Tests advanced operations (DFS, naming, paths)

### Node API Tests
- `NodeOperations` - Tests node inspection and manipulation

### NodeArg API Tests
- `NodeArgOperations` - Tests node argument creation, cloning, and properties

### NodeAttributes API Tests
- `NodeAttributesOperations` - Tests node attribute containers

### AttributeProto API Tests
- `AttributeProtoOperations` - Tests all attribute types (int, float, string, arrays)

### TensorProto API Tests
- `TensorProtoOperations` - Tests all tensor data types and operations

### Extended API Tests
- `ExtendedApiOperations` - Tests library info, proto operations
- `GraphTensorOperations` - Tests graph-level tensor operations

### Coverage Analysis Tests
- `ComprehensiveCoverageReport` - Analyzes and reports API coverage
- `DetailedCoverageAnalysis` - Provides detailed coverage analysis with recommendations

## Running the Tests

### Command Line
```bash
# Run all tests
./ort-bridge-test

# Run only MorphiZen ORT API tests
./ort-bridge-test --gtest_filter="MorphizenOrtApiTest.*"

# Run specific test category
./ort-bridge-test --gtest_filter="MorphizenOrtApiTest.ModelLoadAndDelete"

# Run with verbose logging
./ort-bridge-test --gtest_filter="MorphizenOrtApiTest.*" --v=3
```

### CMake/CTest
```bash
# Build and run tests
cmake --build . --target ort-bridge-test
ctest -R "ort-bridge-test" -V
```

## Interpreting Results

### Coverage Report
The tests will output a comprehensive coverage report showing:

```
=== MorphiZen ORT API Coverage Report ===
Total unique APIs called: 87
API Coverage: 82.3% (87/109)

Model APIs: 5/7 (71.4%)
Graph APIs: 18/23 (78.3%)
Node APIs: 6/10 (60.0%)
NodeArg APIs: 10/13 (76.9%)
NodeAttributes APIs: 5/5 (100.0%)
AttributeProto APIs: 18/20 (90.0%)
TensorProto APIs: 19/20 (95.0%)
Extended APIs: 6/9 (66.7%)

=== Most Frequently Called APIs ===
  tensor_proto_delete: 45 calls
  attr_proto_delete: 12 calls
  node_arg_new: 8 calls
  ...
```

### Missing APIs
The detailed analysis will list any APIs that weren't covered:

```
=== Missing APIs (22) ===
Missing Model APIs (2):
  - model_load
  - graph_save
Missing Graph APIs (5):
  - graph_fuse
  - graph_remove_node
  ...
```

### Recommendations
The tests provide actionable recommendations:

```
=== Coverage Improvement Recommendations ===
Consider adding tests with actual ONNX model files for Model APIs
Consider adding tests with more complex graph operations
Consider adding tests with models that have actual nodes
```

## Test Design Philosophy

### Safe Testing
- Tests are designed to be safe and not require external dependencies
- Many operations are tested through the wrapper even if they fail with actual data
- Focus is on API coverage and call verification rather than functional correctness

### Realistic Testing
- Tests attempt to create realistic scenarios where possible
- Uses temporary files and in-memory operations when feasible
- Handles expected failures gracefully

### Coverage Focused
- Primary goal is to exercise all API functions through the wrapper
- Detailed logging and statistics collection
- Comprehensive reporting for CI/CD integration

## Integration with CI/CD

The tests output metrics that can be parsed by build systems:

```
COVERAGE_METRIC: 82.34
TOTAL_APIS: 109
COVERED_APIS: 87
MISSING_APIS: 22
```

These can be used for:
- Setting coverage thresholds in CI pipelines
- Tracking coverage trends over time
- Identifying areas needing more test development

## Extending the Tests

To add new test cases:

1. **Add new test methods** to `MorphizenOrtApiTest` class
2. **Update API lists** in `test-api-coverage-checker.hpp` if new APIs are added
3. **Follow naming conventions** for test methods
4. **Use the coverage wrapper** to ensure calls are logged and counted
5. **Handle failures gracefully** since some APIs may require specific conditions

## Best Practices

1. **Reset statistics** at the beginning of each test with `reset_morphizen_ort_api_call_statistics()`
2. **Check coverage** after major test sections to verify expected API calls were made
3. **Use try-catch blocks** around operations that may fail without proper setup
4. **Log informative messages** to help with debugging and analysis
5. **Maintain wrapper compatibility** when adding new APIs to the interface

## Example Usage in Your Tests

```cpp
class MyApiTest : public MorphizenOrtApiTest {
protected:
  void SetUp() override {
    MorphizenOrtApiTest::SetUp();
    // Your custom setup
  }
};

TEST_F(MyApiTest, MySpecificOperation) {
  reset_morphizen_ort_api_call_statistics();

  // Perform your operations using wrapped_api_
  auto result = wrapped_api_->some_operation();

  // Verify the API was called
  auto stats = get_morphizen_ort_api_call_statistics();
  EXPECT_GT(stats["some_operation"], 0);
}
```
