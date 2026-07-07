<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Graph Class Refactoring Summary

## Overview
Successfully refactored and separated the graph classes into dedicated files:
- `OrtGraphWrapper`: Wraps ORT's incomplete graph API (in `ort-graph-wrapper.hpp/cpp`)
- `Graph`: New in-memory topological representation of ONNX GraphProto (in `graph.hpp/cpp`)

## Changes Made

### 1. File Structure Reorganization
- **`ort-graph-wrapper.hpp/cpp`**: Contains the `OrtGraphWrapper` class for ORT API bridging
- **`graph.hpp/cpp`**: Contains the new `Graph` class placeholder for in-memory operations
- **Separation**: Clean separation of concerns with dedicated files for each class

### 2. Renamed ORT Wrapper Class
- **Old**: `Graph` class that wraps ORT's incomplete graph API
- **New**: `OrtGraphWrapper` class with the same functionality
- **Purpose**: Frees up the name `Graph` for the new in-memory implementation

### 2. Updated All References
- **ir-converter.hpp/cpp**: Updated all method signatures to use `OrtGraphWrapper`
- **morphizen-ep.hpp/cpp**: Updated forward declarations and usage
- **api-ptrs.hpp**: Updated forward declarations
- **Documentation**: Updated ARRAY_LIFETIME_MANAGEMENT.md
- **Tests**: Updated test file references

### 3. Added New Graph Class Placeholder
- Added comprehensive placeholder for the new `Graph` class
- Includes planned methods for:
  - Topological sorting
  - Dependency analysis
  - Graph inspection
  - Graph transformations
- Documented with clear purpose and design intent

### 4. Maintained Compatibility
- All existing functionality preserved
- No breaking changes to the API
- Both classes now coexist in the same namespace

## Files Modified

### Core Files
- `ort-bridge/src/ort-graph-wrapper.hpp/cpp` - OrtGraphWrapper class for ORT API bridging
- `ort-bridge/src/graph.hpp/cpp` - New Graph class placeholder for in-memory operations
- `ort-bridge/src/ir-converter.hpp` - Updated to use OrtGraphWrapper explicitly
- `ort-bridge/src/ir-converter.cpp` - Updated all method signatures
- `ort-bridge/src/morphizen-ep.hpp` - Updated forward declaration
- `ort-bridge/src/morphizen-ep.cpp` - Updated usage and comments
- `ort-bridge/src/api-ptrs.hpp` - Updated forward declarations
- `ort-bridge/ort-bridge.cmake` - Updated to include both new files

### Documentation and Tests
- `ort-bridge/doc/ARRAY_LIFETIME_MANAGEMENT.md` - Updated examples
- `ort-bridge/test/CMakeLists.txt` - Added new test file

## Next Steps

### For the New Graph Class Implementation
1. **Constructor**: `explicit Graph(const ONNX_NAMESPACE::GraphProto& graph_proto)`
2. **Topological Operations**:
   - `topological_sort()`
   - `get_dependencies()`
   - `get_dependents()`
3. **Graph Inspection**:
   - `find_node()`, `nodes()`, `inputs()`, `outputs()`, `initializers()`
4. **Graph Transformations**:
   - `remove_node()`, `add_node()`, `replace_node()`

### Private Data Members to Add
- `ONNX_NAMESPACE::GraphProto graph_proto_`
- `std::unordered_map<std::string, std::vector<std::string>> dependencies_`
- `std::unordered_map<std::string, std::vector<std::string>> dependents_`

## Benefits

1. **Clear Separation of Concerns**:
   - `OrtGraphWrapper`: Patches ORT's incomplete API
   - `Graph`: Efficient in-memory ONNX operations

2. **No Breaking Changes**: All existing code continues to work

3. **Future-Proof**: Ready for efficient topological operations on ONNX graphs

4. **Well-Documented**: Clear purpose and usage patterns for both classes

## Usage Examples

### OrtGraphWrapper (ORT API Bridge)
```cpp
#include "./ort-graph-wrapper.hpp"

// Wraps ORT's incomplete graph API
auto graph_wrapper = morphizen::OrtGraphWrapper(api_ptrs, ort_graph);
auto nodes = graph_wrapper.nodes();
auto opset = graph_wrapper.guess_opset();
```

### Graph (Future In-Memory Operations)
```cpp
#include "./graph.hpp"

// TODO: When implemented
auto graph = morphizen::Graph(graph_proto);
auto sorted_nodes = graph.topological_sort();
auto deps = graph.get_dependencies("conv_node");
```
