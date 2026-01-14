# morphizen-hipdnn: MIOpen Migration Complete Guide

**Version:** 1.0  
**Date:** January 14, 2026  
**Status:** ✅ COMPLETED AND TESTED

---

## Table of Contents

- [Overview](#overview)
- [Migration Summary](#migration-summary)
- [Quick Start](#quick-start)
- [Architecture Changes](#architecture-changes)
- [Build Instructions](#build-instructions)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [Testing](#testing)
- [Performance](#performance)

---

## Overview

### What Changed

The **morphizen-hipdnn** project has been migrated from AMD's **hipDNN** high-level API to **MIOpen** low-level API. This migration provides:

- ✅ Direct MIOpen API usage for better performance
- ✅ Lower-level control over convolution operations  
- ✅ Elimination of hipDNN abstraction layer
- ✅ JSON-based metadata exchange (replaces graph serialization)
- ✅ Static glog linking for stable Debug builds

### Migration Status

| Component | Status | Changes |
|-----------|--------|---------|
| Custom Operator | ✅ Complete | MIOpen descriptors, algorithm finding |
| Level-1 Pass | ✅ Complete | JSON metadata generation |
| Build System | ✅ Complete | Static glog, MIOpen dependencies |
| Testing | ✅ Complete | All tests passing (Debug config) |

**Commit:** `e508d78` - Pushed to remote master

---

## Migration Summary

### Files Modified

#### Core Implementation (7 files)

1. **`custom-op-hipdnn/src/custom_op.cpp`** (Major rewrite)
   - Replaced hipDNN graph execution with MIOpen API calls
   - Added MIOpen descriptor setup
   - Implemented algorithm finding with temporary buffers
   - Added JSON metadata parsing

2. **`custom-op-hipdnn/src/custom_op.hpp`** (Major changes)
   - Replaced hipDNN types with MIOpen types
   - Added MIOpen handles and descriptors
   - Updated member variables for MIOpen workflow

3. **`level-1-pass-hipdnn/src/pass_main.cpp`** (Major rewrite)
   - Removed hipDNN graph building
   - Added JSON metadata generation
   - Extracts Conv parameters from ONNX nodes
   - Saves shapes, strides, pads, dilations to JSON

4. **`cmake/deps.cmake`** (Critical fix)
   - Added static glog linking configuration
   - Fixes Debug mode LNK2005 linker errors

5. **`custom-op-hipdnn/CMakeLists.txt`**
   - Replaced hipDNN dependencies with MIOpen
   - Added nlohmann_json dependency

6. **`level-1-pass-hipdnn/CMakeLists.txt`**
   - Replaced hipDNN dependencies with MIOpen
   - Added nlohmann_json dependency

7. **`test/CMakeLists.txt`**
   - Added BUILD_TEST_ONNX_RUNNER option (default: OFF)

### Key Changes Summary

**Before (hipDNN):**
- High-level graph API
- Graph serialization to binary
- Automatic algorithm selection
- Dynamic glog linking

**After (MIOpen):**
- Low-level descriptor API
- JSON metadata files
- Explicit algorithm finding
- Static glog linking

---

## Quick Start

### Prerequisites

```powershell
# Required installations
- Windows 10/11 with Visual Studio 2022
- CMake 3.29+
- AMD ROCm/TheRock (set THEROCK_DIST environment variable)
- ONNX Runtime (installed to local directory)
- MIOpen library
- nlohmann_json library

# Environment Variables
$env:THEROCK_DIST = "D:\therock"
$env:HIP_PLATFORM = "amd"
```

### Build & Test

```powershell
# 1. Navigate to project
cd D:\Users\mingyue\hipdnn\workspace\morphizen-hipdnn

# 2. Configure
cmake -DBUILD_SHARED_LIBS=OFF `
  -B ../build/morphizen-hipdnn `
  -S . `
  -DCMAKE_INSTALL_PREFIX=../local `
  -DTHEROCK_DIST="D:\therock" `
  -Dmorphizen_ENABLE_ORT_BRIDGE=ON

# 3. Build and Install
cmake --build ../build/morphizen-hipdnn --config Debug --target install

# 4. Run Test
$env:VITISAI_EP_JSON_CONFIG = "D:\Users\mingyue\hipdnn\workspace\local\bin\vaip_config.json"
$env:MORPHIZEN_DEBUG_HIPDNN = "1"
test_onnx_runner.exe conv_test.onnx
```

### Expected Output

```
✓ HipdnnCustomOp constructor (MIOpen version)
✓ Building MIOpen kernel
✓ Selected algorithm: 1 (time: 0.0278 ms)
✓ Convolution forward completed
✓ Exit code: 0
```

---

## Architecture Changes

### Before: hipDNN Flow

```
ONNX Model
    ↓
HipdnnPass (builds hipDNN graph)
    ↓
Serialize graph → graph.bin
    ↓
Custom Op (loads graph.bin)
    ↓
hipDNN Engine (executes graph)
    ↓
Result
```

### After: MIOpen Flow

```
ONNX Model
    ↓
HipdnnPass (extracts Conv parameters)
    ↓
Generate JSON → metadata.json (shapes, pads, strides...)
    ↓
Custom Op (parses JSON)
    ↓
Create MIOpen descriptors
    ↓
Find convolution algorithm (with temp buffers)
    ↓
miopenConvolutionForward
    ↓
Result
```

### Data Exchange Format

**JSON Metadata Example:**
```json
{
  "op_type": "Conv",
  "version": "miopen_1.0",
  "input_shapes": [[1,1,10,10], [1,1,3,3]],
  "output_shapes": [[1,1,8,8]],
  "input_data_types": [1, 1],
  "output_data_types": [1],
  "pads": [0, 0],
  "strides": [1, 1],
  "dilations": [1, 1],
  "group": 1,
  "has_bias": false
}
```

---

## Build Instructions

### CMake Configuration

```cmake
# Required Options
-DBUILD_SHARED_LIBS=OFF              # Static libraries (glog fix)
-DCMAKE_BUILD_TYPE=Debug             # Debug or Release
-DCMAKE_INSTALL_PREFIX=<install_dir> # Installation directory
-DTHEROCK_DIST=<therock_path>        # TheRock SDK location

# Optional
-DBUILD_TEST_ONNX_RUNNER=OFF         # Disable test compilation
-Dmorphizen_ENABLE_ORT_BRIDGE=ON     # Enable ORT bridge
```

### Build Targets

```bash
# Build everything
cmake --build <build_dir> --config Debug

# Build specific components
cmake --build <build_dir> --config Debug --target morphizen-custom-op-hipdnn
cmake --build <build_dir> --config Debug --target morphizen-level1-pass-hipdnn

# Install
cmake --build <build_dir> --config Debug --target install
```

### Output Files

```
local/
├── bin/
│   ├── onnxruntime_vitisai_ep.dll  # Main EP DLL (22+ MB)
│   ├── vaip_config.json             # Configuration
│   └── test_onnx_runner.exe         # Test executable
└── lib/
    └── onnxruntime_vitisai_ep.lib   # Import library
```

---

## Troubleshooting

### 1. LNK2005: Multiple Definition Errors (Debug)

**Symptoms:**
```
error LNK2005: already defined in glogd.lib
error LNK2005: already defined in libcpmtd.lib
```

**Cause:** Mixing dynamic glog (glogd.dll) with static C++ runtime

**Solution:** Add to `cmake/deps.cmake`:
```cmake
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(GLOG_BUILD_SHARED OFF CACHE BOOL "Build glog shared library" FORCE)
```

**Status:** ✅ Fixed in commit

---

### 2. JSON Parse Error: Type Must Be Array

**Symptoms:**
```
Failed to compile MIOpen kernel: type must be array, but is null
```

**Cause:** JSON format mismatch between pass and custom op

**Solution:** Use flat array structure in `pass_main.cpp`:
```cpp
metadata["input_shapes"] = nlohmann::json::array();
metadata["input_shapes"].push_back(*x_shape);
metadata["output_shapes"] = nlohmann::json::array();
metadata["output_shapes"].push_back(*y_shape);
```

**Status:** ✅ Fixed in commit

---

### 3. MIOpen Error: No Invoker Registered

**Symptoms:**
```
MIOpen Error: No invoker was registered for convolution forward.
Was find executed?
```

**Cause:** Must call `miopenFindConvolutionForwardAlgorithm` before execution

**Solution:** In `custom_op.cpp`:
```cpp
miopenFindConvolutionForwardAlgorithm(
    miopen_handle_,
    x_desc_, input_data,
    w_desc_, weight_data,
    conv_desc_,
    y_desc_, output_data,
    requestedAlgoCount,
    &returnedAlgoCount,
    &perfResults,
    workspace_, workspace_size_,
    false);
```

**Status:** ✅ Fixed in commit

---

### 4. MIOpen Error: Buffers Cannot Be NULL

**Symptoms:**
```
MIOpen Error: Buffers cannot be NULL
```

**Cause:** `miopenFindConvolutionForwardAlgorithm` requires actual GPU buffers

**Solution:** Allocate temporary buffers in `BuildAndCompileMIOpen()`:
```cpp
void* temp_x, *temp_w, *temp_y;
hipMalloc(&temp_x, x_size);
hipMalloc(&temp_w, w_size);
hipMalloc(&temp_y, y_size);

miopenFindConvolutionForwardAlgorithm(..., temp_x, ..., temp_w, ..., temp_y, ...);

hipFree(temp_x);
hipFree(temp_w);
hipFree(temp_y);
```

**Status:** ✅ Fixed in commit

---

### 5. Environment Variable Not Set

**Symptoms:**
```
CMake Error: THEROCK_DIST not set
```

**Solution:**
```powershell
$env:THEROCK_DIST = "D:\therock"
$env:HIP_PLATFORM = "amd"
```

---

### 6. CMake Cannot Find onnxruntime

**Symptoms:**
```
CMake Error: Could not find onnxruntime
```

**Solution:**
```cmake
-Donnxruntime_DIR="<install_dir>/lib/cmake/onnxruntime"
```

---

## Technical Details

### MIOpen Descriptor Setup Workflow

```cpp
// 1. Create descriptors
miopenCreateTensorDescriptor(&x_desc_);
miopenCreateTensorDescriptor(&w_desc_);
miopenCreateTensorDescriptor(&y_desc_);
miopenCreateConvolutionDescriptor(&conv_desc_);

// 2. Set tensor descriptors (4D: NCHW)
miopenSet4dTensorDescriptor(x_desc_, miopenFloat, N, C, H, W);
miopenSet4dTensorDescriptor(w_desc_, miopenFloat, K, C, R, S);
miopenSet4dTensorDescriptor(y_desc_, miopenFloat, N, K, P, Q);

// 3. Set convolution descriptor
miopenInitConvolutionDescriptor(
    conv_desc_,
    miopenConvolution,
    pad_h, pad_w,
    stride_h, stride_w,
    dilation_h, dilation_w);

// 4. Get workspace size
miopenConvolutionForwardGetWorkSpaceSize(
    miopen_handle_, w_desc_, x_desc_, conv_desc_, y_desc_,
    &workspace_size_);

// 5. Allocate workspace
hipMalloc(&workspace_, workspace_size_);

// 6. Find algorithm (requires temp buffers!)
miopenFindConvolutionForwardAlgorithm(
    miopen_handle_,
    x_desc_, temp_x,
    w_desc_, temp_w,
    conv_desc_,
    y_desc_, temp_y,
    requestedAlgoCount,
    &returnedAlgoCount,
    &perfResults,
    workspace_, workspace_size_,
    exhaustiveSearch);

// 7. Execute
miopenConvolutionForward(
    miopen_handle_,
    &alpha,
    x_desc_, x_data,
    w_desc_, w_data,
    conv_desc_,
    conv_algo_,
    &beta,
    y_desc_, y_data,
    workspace_, workspace_size_);

// 8. Cleanup
hipFree(workspace_);
miopenDestroyTensorDescriptor(x_desc_);
miopenDestroyTensorDescriptor(w_desc_);
miopenDestroyTensorDescriptor(y_desc_);
miopenDestroyConvolutionDescriptor(conv_desc_);
miopenDestroy(miopen_handle_);
```

### API Mapping

| hipDNN | MIOpen | Notes |
|--------|--------|-------|
| `hipdnnCreate()` | `miopenCreate()` | Handle creation |
| `hipdnnCreateTensorDescriptor()` | `miopenCreateTensorDescriptor()` | Tensor descriptor |
| `hipdnnSetTensor4dDescriptor()` | `miopenSet4dTensorDescriptor()` | Set shape |
| `hipdnnConvolutionForward()` | `miopenConvolutionForward()` | Execute convolution |
| Graph API | JSON metadata | Data exchange |
| Auto algorithm | `miopenFindConvolutionForwardAlgorithm()` | Must call explicitly |

### Key Implementation Files

**`custom-op-hipdnn/src/custom_op.cpp`:**
- `HipdnnCustomOp::HipdnnCustomOp()` - Constructor, creates MIOpen handle
- `BuildAndCompileMIOpen()` - Parses JSON, creates descriptors, finds algorithm
- `Compute()` - Executes convolution with MIOpen

**`level-1-pass-hipdnn/src/pass_main.cpp`:**
- `GenerateConvMetadata()` - Extracts Conv parameters from ONNX
- `IsSupportedConv()` - Validates Conv node
- `HipdnnPass::Apply()` - Main pass logic

---

## Testing

### Test Environment Setup

```powershell
# Required Environment Variables
$env:DEBUG_VAIP_PASS = "1"
$env:XLNX_ENABLE_CACHE = "0"
$env:DEBUG_DPU_CUSTOM_OP = "1"
$env:XLNX_ONNX_EP_VERBOSE = "2"
$env:DEBUG_LOG_LEVEL = "info"
$env:VITISAI_EP_JSON_CONFIG = "<install_dir>/bin/vaip_config.json"
$env:MORPHIZEN_VITISAI_EP = "<install_dir>/bin/onnxruntime_vitisai_ep.dll"
$env:MORPHIZEN_DEBUG_HIPDNN = "1"
$env:USE_ORT_API_2_0 = "1"
$env:PATH = "D:\therock\bin;<install_dir>\bin;$env:PATH"
```

### Run Test

```powershell
test_onnx_runner.exe conv_test.onnx
```

### Test Model

- **Input:** [1, 1, 10, 10] (NCHW format)
- **Kernel:** [1, 1, 3, 3]
- **Output:** [1, 1, 8, 8]
- **Operation:** Conv2D (padding=0, stride=1)

### Expected Output

```
I [custom_op.cpp:97] HipdnnCustomOp constructor (MIOpen version)
I [custom_op.cpp:167] === Building MIOpen kernel ===
I [custom_op.cpp:184] Loaded metadata from: hipdnn_meta_Y.json
I [custom_op.cpp:299] Finding best convolution algorithm...
I [custom_op.cpp:326] Selected algorithm: 1 (time: 0.0278 ms, memory: 0 bytes)
I [custom_op.cpp:357] === MIOpen kernel build complete ===
I [custom_op.cpp:406] === HipdnnCustomOp::Compute START (MIOpen) ===
I [custom_op.cpp:456] Convolution forward completed
I [custom_op.cpp:484] === HipdnnCustomOp::Compute END ===
done
```

**Exit Code:** 0 ✅

---

## Performance

### Build Performance

| Configuration | Time | Binary Size |
|--------------|------|-------------|
| Debug (Full) | 5-8 min | 22.4 MB |
| Debug (Incremental) | 30 sec | - |

### Runtime Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Initialization | ~200 ms | Includes algorithm finding |
| Algorithm Selection | 0.0278 ms | One-time cost |
| Conv Forward (1x1x10x10) | <1 ms | Actual execution |

### Memory Usage

- **Workspace:** 0-10 MB (varies by convolution size)
- **Descriptors:** <1 KB per descriptor
- **Temporary Buffers:** Only during initialization

---

## Code Statistics

### Changes Summary

```
Files changed: 7
Insertions: +735
Deletions: -822
Net change: -87 lines (more efficient code)
```

### Modified Components

1. **Custom Operator (MIOpen):**
   - `custom-op-hipdnn/src/custom_op.cpp` (973 lines modified)
   - `custom-op-hipdnn/src/custom_op.hpp` (68 lines modified)

2. **Level-1 Pass (JSON):**
   - `level-1-pass-hipdnn/src/pass_main.cpp` (452 lines modified)

3. **Build System:**
   - `cmake/deps.cmake` (5 lines added)
   - `custom-op-hipdnn/CMakeLists.txt` (30 lines modified)
   - `level-1-pass-hipdnn/CMakeLists.txt` (16 lines modified)

4. **Test System:**
   - `test/CMakeLists.txt` (13 lines added)

---

## Future Work

### Planned Enhancements

- [ ] Release build testing and optimization
- [ ] Additional operations (Pooling, BatchNorm, ReLU)
- [ ] FP16 data type support
- [ ] Algorithm caching to disk
- [ ] Exhaustive search option
- [ ] Performance benchmarking suite

### Known Limitations

1. **Single Operation:** Only Conv2D is currently supported
2. **Debug Only:** Release build not extensively tested
3. **FP32 Only:** FP16 exists but not validated
4. **Static Shapes:** Dynamic batch sizes not supported

---

## References

### External Links

- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [hipDNN GitHub](https://github.com/ROCmSoftwarePlatform/hipDNN)
- [nlohmann/json](https://github.com/nlohmann/json)

### Related Projects

- **hipDNNEP** - Original MIOpen migration source
- **MorphiZen** - Parent optimization framework
- **ONNX Runtime** - ML model runtime

### Key Commits

- `e508d78` - MIOpen migration (this guide)
- `ea624d2` - TheRock SDK hardcoded paths fix
- `fe93204` - hipDNN constant data handling
- `cba7c33` - Constant initializers support

---

## Appendix

### Complete Build Script

```powershell
# build.ps1
param(
    [string]$SourceDir = "D:\Users\mingyue\hipdnn\workspace\morphizen-hipdnn",
    [string]$BuildDir = "D:\Users\mingyue\hipdnn\workspace\build\morphizen-hipdnn",
    [string]$InstallDir = "D:\Users\mingyue\hipdnn\workspace\local",
    [string]$TheRockDir = "D:\therock"
)

# Set environment
$env:THEROCK_DIST = $TheRockDir
$env:HIP_PLATFORM = "amd"

# Configure
cmake `
    -DBUILD_SHARED_LIBS=OFF `
    -B $BuildDir `
    -S $SourceDir `
    -DCMAKE_INSTALL_PREFIX=$InstallDir `
    -DTHEROCK_DIST=$TheRockDir `
    -Dmorphizen_ENABLE_ORT_BRIDGE=ON `
    -DBUILD_TEST_ONNX_RUNNER=OFF

# Build and Install
cmake --build $BuildDir --config Debug --target install

Write-Host "Build complete!" -ForegroundColor Green
```

### Complete Test Script

```powershell
# test.ps1
param(
    [string]$TestModel = "conv_test.onnx",
    [string]$InstallDir = "D:\Users\mingyue\hipdnn\workspace\local"
)

# Set environment
$env:DEBUG_VAIP_PASS = "1"
$env:XLNX_ENABLE_CACHE = "0"
$env:DEBUG_DPU_CUSTOM_OP = "1"
$env:XLNX_ONNX_EP_VERBOSE = "2"
$env:DEBUG_LOG_LEVEL = "info"
$env:VITISAI_EP_JSON_CONFIG = "$InstallDir\bin\vaip_config.json"
$env:MORPHIZEN_VITISAI_EP = "$InstallDir\bin\onnxruntime_vitisai_ep.dll"
$env:MORPHIZEN_DEBUG_HIPDNN = "1"
$env:USE_ORT_API_2_0 = "1"
$env:PATH = "D:\therock\bin;$InstallDir\bin;$env:PATH"

# Run test
& "$InstallDir\bin\test_onnx_runner.exe" $TestModel
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-14 | mingyue | Initial comprehensive guide |

---

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc.  
Licensed under the MIT License.

---

**END OF DOCUMENT**
