<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MLIR Implementation of MorphiZen ORT API

This directory contains the MLIR-based implementation of the MorphiZen ORT API bridge, providing an alternative IR representation to the ONNX-based implementation.

## File Structure

### Implementation Files (`src/`)
- `mlir-context-manager.hpp/.cpp` - MLIR context singleton management
- `mlir-model.hpp/.cpp` - MLIR-based model implementation using `mlir::ModuleOp`
- `mlir-graph.hpp/.cpp` - Graph representation using MLIR blocks
- `morphizen-ort-api.cpp` - Main API bridge implementation and function pointer initialization

**Note**: All header files are private implementation details and are located in the `src/` directory alongside their corresponding `.cpp` files.

## Key Features

1. **Modular Design**: Classes are split into separate header and implementation files for better maintainability
2. **MLIR Integration**: Uses MLIR's `ModuleOp` and `Block` for IR representation
3. **API Compatibility**: Implements the same `MorphizenOrtApiExt` interface as the ONNX version
4. **Context Management**: Proper MLIR context setup with dialect loading

## Usage

The implementation provides a factory function `get_morphizen_ort_api_mlir()` that returns the MLIR-specific API implementation:

```cpp
extern "C" MorphizenOrtApiExt* get_morphizen_ort_api_mlir();
```

## Implementation Status

- ✅ **Model Management**: Load/save MLIR modules, metadata handling
- ✅ **Basic Graph Operations**: Name retrieval, model references
- ✅ **Tensor Utilities**: ONNX-compatible tensor creation
- ⚠️ **Graph Traversal**: Placeholder implementations (to be completed)
- ⚠️ **Node Operations**: Placeholder implementations (to be completed)
- ⚠️ **NodeArg Operations**: Placeholder implementations (to be completed)

## Build Integration

Include the CMakeLists-additions.txt content in your main CMakeLists.txt file to build all components together.

## Future Work

1. Complete MLIR-specific node and node argument implementations
2. Add proper MLIR operation traversal and manipulation
3. Implement MLIR-native tensor representations
4. Add MLIR dialect-specific optimizations
5. Extend with custom MLIR dialects for MorphiZen operations
