# Pass Graph Validation Implementation

## Overview

This document describes the implementation of hipDNN graph building and validation in the Level 1 pass (`level-1-pass-hipdnn/src/pass_main.cpp`), following the pattern established in `external/hipDNNEP/src/kernel.cc`.

## Purpose

The pass now validates that a Conv operation can be successfully represented as a hipDNN graph before committing to fusion. This early validation ensures:

1. **Compatibility Check**: Verifies the operation is compatible with hipDNN before fusion
2. **Static Shape Requirement**: Ensures all tensors have static shapes required by hipDNN
3. **Data Type Support**: Confirms data types are supported (FLOAT, HALF)
4. **Graph Validity**: Validates the graph structure using hipDNN's validation API

## Implementation Details

### Added Includes

```cpp
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <memory>
#include <unordered_map>
```

### Helper Functions

#### 1. ComputeStrides
```cpp
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
```
- Computes tensor strides from shape (NCHW layout)
- Matches the implementation in `kernel.cc`

#### 2. ToHipDNNDataType
```cpp
std::optional<hipdnn_frontend::DataType> ToHipDNNDataType(int32_t onnx_dtype)
```
- Converts ONNX data types to hipDNN data types
- Supports FLOAT (type 1) and HALF (type 10)
- Returns `std::nullopt` for unsupported types

#### 3. GetComputeDataType
```cpp
std::optional<hipdnn_frontend::DataType> GetComputeDataType(
    hipdnn_frontend::DataType x_dtype,
    hipdnn_frontend::DataType w_dtype)
```
- Determines the compute data type for the operation
- Uses FLOAT for compute when inputs are float types

### Main Validation Function

#### BuildAndValidateGraph
```cpp
bool BuildAndValidateGraph(
    const Node& conv_node,
    const NodeArg& input_arg,
    const NodeArg& weight_arg,
    const NodeArg& output_arg)
```

This function performs the following steps:

1. **Create hipDNN Graph**: Instantiates a `hipdnn_frontend::graph::Graph`

2. **Extract Tensor Information**:
   - Gets shapes for input, weight, and output tensors
   - Validates all shapes are static (required by hipDNN)
   - Converts ONNX data types to hipDNN data types

3. **Create Tensor Attributes**:
   - Creates `TensorAttributes` for input and weight
   - Sets UID, name, data type, dimensions, and strides
   - Marks tensors as non-virtual

4. **Extract Conv Attributes**:
   - Reads `pads`, `strides`, and `dilations` from the Conv node
   - Normalizes padding format (converts 2-element to 4-element if needed)
   - Determines compute data type

5. **Build Graph**:
   - Creates `ConvFpropAttributes` with operation parameters
   - Adds convolution operation to graph using `graph->conv_fprop()`
   - Sets output tensor attributes

6. **Validate**:
   - Calls `graph->validate()` to check graph validity
   - Returns `false` if validation fails
   - Returns `true` if graph is valid

### Integration into Pass

The validation is integrated into the pattern matching callback:

```cpp
// Get Conv node inputs (input data and weight)
auto conv_inputs = node_get_inputs(*conv_node);
if (conv_inputs.size() < 2) {
  MY_LOG(1) << "Conv node must have at least 2 inputs (data and weight)";
  return false;
}

// Build and validate hipDNN graph
bool graph_valid = BuildAndValidateGraph(
    *conv_node, 
    *conv_inputs[0].node_arg,  // input data
    *conv_inputs[1].node_arg,  // weight
    *output.node_arg_);

if (!graph_valid) {
  MY_LOG(1) << "hipDNN graph validation failed, skipping fusion";
  return false;
}

MY_LOG(1) << "hipDNN graph validation succeeded, proceeding with fusion";
```

## Differences from kernel.cc

While following the same pattern, there are key differences:

| Aspect | kernel.cc | pass_main.cpp |
|--------|-----------|---------------|
| **Purpose** | Build, compile, and execute graph | Validate graph can be built |
| **Graph Lifecycle** | Stored in Kernel class for execution | Temporary, discarded after validation |
| **Compilation** | Calls `build_operation_graph()`, `create_execution_plans()`, etc. | Only validates structure |
| **Input Source** | Ort::ConstGraph (ORT API) | VAIP pattern binder |
| **Context** | Runtime execution | Compile-time pass optimization |
| **Error Handling** | Returns OrtStatus* | Returns bool |

## Benefits

1. **Early Error Detection**: Catches incompatible operations before fusion
2. **Cleaner Error Messages**: Provides specific validation failures via logging
3. **Safer Fusion**: Only fuses operations that hipDNN can handle
4. **Consistent API**: Uses the same hipDNN frontend API as runtime execution

## Future Enhancements

Potential improvements:

1. **Extended Logging**: Add more detailed validation failure reasons
2. **Caching**: Cache validation results for repeated patterns
3. **Multi-Op Validation**: Extend to validate fused multi-operation graphs
4. **Performance Metrics**: Track validation time and success rates

## References

- Original implementation: `external/hipDNNEP/src/kernel.cc::BuildAndCompile()`
- hipDNN Frontend API: `hipdnn_frontend.hpp`
- VAIP Pattern Matching: `morphizen/vaip.hpp`
