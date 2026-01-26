# Build Fixes for onnx-hipdnn-ep

This document describes the fixes applied to successfully build the onnx-hipdnn-ep project with hipDNN integration on Windows.

## Date
January 9, 2026

## Overview
The project required several configuration and source code fixes to compile successfully with TheRock ROCm SDK and hipDNN libraries on Windows using Visual Studio 2022.

## Changes Made

### 1. Build Configuration (build.bat)

**File**: `build.bat`

**Changes**:
- Added `THEROCK_DIST` environment variable pointing to `C:\Develop\TheRock`
- Added `HIP_PLATFORM=amd` environment variable
- Passed `-DTHEROCK_DIST=C:/Develop/TheRock` to CMake configuration

**Purpose**: These changes ensure CMake can locate the TheRock ROCm SDK and hipDNN libraries during configuration.

```batch
REM Set TheRock environment
echo Setting TheRock environment...
set THEROCK_DIST=C:\Develop\TheRock
set HIP_PLATFORM=amd
echo THEROCK_DIST=%THEROCK_DIST%
```

### 2. CMake Configuration (level-1-pass-hipdnn/CMakeLists.txt)

**File**: `level-1-pass-hipdnn/CMakeLists.txt`

**Changes**:

#### Include Directories
Added TheRock include directory for HIP headers:
```cmake
target_include_directories(${LIB_NAME} PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/hipDNNEP/include
    ${THEROCK_DIST}/include  # Added for HIP headers
)
```

#### Compile Definitions
Added HIP platform definition:
```cmake
target_compile_definitions(${LIB_NAME} PRIVATE 
    -DOUTPUT_NAME="morphizen-level1-pass-hipdnn"
    -D__HIP_PLATFORM_AMD__  # Required by HIP headers
)
```

#### Compile Options
Added `/utf-8` flag required by fmt library:
```cmake
target_compile_options(${LIB_NAME} PRIVATE /utf-8)
```

**Purpose**: These changes resolve compilation errors related to missing HIP headers, platform definitions, and Unicode support.

### 3. Source Code Fixes (level-1-pass-hipdnn/src/pass_main.cpp)

**File**: `level-1-pass-hipdnn/src/pass_main.cpp`

**Changes**:

#### Namespace Conflict Resolution
The code had conflicts between `onnxruntime::Graph` and `hipdnn_frontend::graph::Graph`. Fixed by:

1. Using type alias for hipDNN graph:
```cpp
using HipDNNGraph = hipdnn_frontend::graph::Graph;
using hipdnn_frontend::graph::TensorAttributes;
using hipdnn_frontend::graph::ConvFpropAttributes;
```

2. Renaming ONNX Runtime graph parameters:
```cpp
// Changed from: onnxruntime::Graph* graph
// To:
onnxruntime::Graph* ort_graph
```

#### hipDNN API Corrections

**Original (incorrect) code**:
```cpp
// Create output tensor separately
auto y_attr = std::make_shared<TensorAttributes>();
y_attr->set_uid(next_uid++)
    .set_name(node_arg_get_name(output_ref))
    // ... set other properties

// Pass as 4th parameter
graph->conv_fprop(x_attr, w_attr, y_attr, conv_attrs);

// Validate with HeurMode (doesn't exist)
auto [error, plan] = graph->create_execution_plans({hipdnn_frontend::HeurMode::A});
```

**Corrected code**:
```cpp
// conv_fprop returns the output tensor (3 parameters, not 4)
auto y_attr = graph->conv_fprop(x_attr, w_attr, conv_attrs);

// Set output properties on returned tensor
y_attr->set_uid(next_uid++)
    .set_name(node_arg_get_name(output_ref))
    // ... set other properties

// Validate by checking return value
if (!y_attr) {
    MY_LOG(1) << "hipDNN graph validation failed: conv_fprop returned null";
    return false;
}
```

**Purpose**: These changes align with the actual hipDNN frontend API provided by TheRock.

## Build Results

After applying these fixes, the build completed successfully:
- All 43 targets compiled without errors
- Generated `morphizen-level1-pass-hipdnn.lib` static library
- Generated `onnxruntime_vitisai_ep.dll` execution provider DLL
- All test executables built successfully

## Dependencies

The successful build requires:
- TheRock ROCm SDK installed at `C:\Develop\TheRock`
- Visual Studio 2022 with C++ toolchain
- CMake 3.25.2+
- Ninja build system
- hipDNN frontend and backend libraries (included in TheRock)

## Related Documentation

- [HIPDNN_WINDOWS_SETUP.md](HIPDNN_WINDOWS_SETUP.md) - Complete setup guide for hipDNN on Windows
- [HIPDNNEP_INSTALLATION.md](HIPDNNEP_INSTALLATION.md) - hipDNN EP installation instructions
- [PASS_GRAPH_VALIDATION.md](PASS_GRAPH_VALIDATION.md) - Graph validation implementation details

## Testing

To verify the build:
```batch
cd C:\Develop\m\build\onnx-hipdnn-ep
ninja
```

Expected output: `Build completed successfully!`

## Future Improvements

1. Make TheRock path configurable via environment variable or CMake cache variable
2. Add automated detection of TheRock installation
3. Consider adding CI/CD pipeline to prevent regression
4. Document minimum required TheRock version
