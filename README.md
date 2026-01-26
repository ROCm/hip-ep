# morphizen-mlir

A minimal MLIR integration project for the MorphiZen framework.

## Overview

This project provides a level-1 MLIR pass for the MorphiZen/VAIP framework. It serves as a template for MLIR-based graph transformations without pattern matching or protobuf dependencies.

## Project Structure

```
morphizen-mlir/
├── CMakeLists.txt              # Root CMake configuration
├── README.md                   # This file
├── LICENSE                     # Apache 2.0 License
├── .gitignore                  # Git ignore rules
├── cmake/                      # CMake modules
│   ├── deps.cmake             # Dependency management
│   └── llvm.cmake             # LLVM/MLIR configuration
├── level-1-pass-mlir/         # MLIR pass implementation
│   ├── CMakeLists.txt         # Pass build configuration
│   └── src/
│       └── pass_main.cpp      # Main pass implementation with MLIR parsing
├── test/                       # Test infrastructure
│   ├── CMakeLists.txt         # Test build configuration
│   ├── test_ort_integration.cpp  # ORT integration tests
│   ├── gen_conv_model.py      # Conv model generator
│   └── gen_conv_gemm_model.py # Conv+Gemm model generator
├── doc/                        # Documentation
│   └── TESTING.md             # Testing guide with examples
└── etc/                        # Configuration files
    └── vaip_config.json       # VAIP pass configuration
```

## Key Features

- **Minimal Design**: No pattern matching, no protobuf dependencies
- **Level-1 Pass**: Simple VAIP pass structure for MLIR integration
- **Clean Architecture**: Based on morphizen-demo but simplified
- **Ready for Extension**: Template for adding MLIR-based transformations

## Building

### Prerequisites

- CMake 3.29 or later
- Visual Studio 2022
- Python 3 (with onnx package: `pip install onnx`)
- Git

### Build Dependencies

The build process requires the following dependencies to be built in order:

1. **ONNXRuntime** (required - must be built first)
2. **LLVM/MLIR** (optional pre-build - can be auto-fetched via FetchContent or pre-built manually)
3. **MorphiZen** (automatically fetched via CMake FetchContent from ../MorphiZen)

**Note:** LLVM/MLIR pre-build is optional. The build system will automatically fetch and build LLVM via FetchContent if not pre-installed. However, pre-building can save time on subsequent builds.

### Recommended Directory Layout

```
workspace/
├── onnxruntime/           # ONNXRuntime source (cloned from GitHub)
├── onnx-hipdnn-ep/        # This project (morphizen-mlir)
├── build/
│   ├── onnxruntime/       # ONNXRuntime build output
│   └── morphizen-mlir/    # morphizen-mlir build output
└── local/                 # Installation prefix
    ├── bin/               # DLLs and executables
    ├── lib/               # Libraries
    └── include/           # Headers
```

### Step-by-Step Build Instructions

#### Step 1: Build ONNXRuntime

ONNXRuntime must be built first as it's a core dependency:

```bash
# Clone ONNXRuntime (from workspace root)
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime

# Build and install (Release configuration)
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local

# Install
cmake --build ../build/onnxruntime/Release/ --target install
```

#### Step 2: Build morphizen-mlir

Once ONNXRuntime is built, you can build morphizen-mlir. LLVM/MLIR and MorphiZen will be automatically fetched via CMake FetchContent:

**Option A: Using Visual Studio generator (recommended for Windows)**
```bash
cd onnx-hipdnn-ep

# Configure with Visual Studio generator
cmake -G "Visual Studio 17 2022" -A x64 \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
  -S . -B ../build/morphizen-mlir \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DCMAKE_PREFIX_PATH=$PWD/../local

# Build
cmake --build ../build/morphizen-mlir --config Release
```

**Option B: Using the build script**
```bash
./build.bat
```

**Note:** The first build will take a long time (1-3 hours) as LLVM/MLIR is fetched and compiled. Subsequent builds are much faster.

### Quick Build Summary

```bash
# From workspace root directory:

# 1. Build ONNXRuntime
cd onnxruntime
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ../build/onnxruntime/Release/ --target install

# 2. Build morphizen-mlir (LLVM/MLIR and MorphiZen will be auto-fetched)
cd ../onnx-hipdnn-ep
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -S . -B ../build/morphizen-mlir -DCMAKE_INSTALL_PREFIX=../local -DCMAKE_PREFIX_PATH=$PWD/../local
cmake --build ../build/morphizen-mlir --config Release
```

### Optional: Pre-build LLVM/MLIR

If you want to pre-build LLVM/MLIR instead of using FetchContent (which can take several hours during the first build), you can build it manually. The build system will automatically detect and use the pre-built LLVM if available in `../local`.

## Configuration

The VAIP configuration is defined in `etc/vaip_config.json`:

- **Pass Name**: `mlir-pass`
- **Plugin**: `morphizen-level1-pass-mlir`
- **Target**: `mlir-target`

## Development

### Adding MLIR Logic

The main pass implementation is in `level-1-pass-mlir/src/pass_main.cpp`. The `Level1MlirPass::process()` method is where you would add MLIR transformation logic.

### Debug Logging

Set the environment variable `MORPHIZEN_DEBUG_MLIR=1` to enable debug logging.

## MLIR Integration

This project includes full MLIR integration for graph processing:

### Features

- **MLIR Parsing**: Parse ONNX graphs saved in MLIR format to `mlir::ModuleOp`
- **Operation Walking**: Walk and inspect all operations in the MLIR module
- **ModuleOp Printing**: Print complete MLIR IR with detailed flags
- **Dialect Support**: Loaded dialects include func, arith, and unregistered dialects

### MLIR Pass Implementation

The Level-1 MLIR pass (`level-1-pass-mlir/src/pass_main.cpp`) performs:
1. Saves graph to file (`graph_for_mlir.onnx`)
2. Parses MLIR file to `mlir::ModuleOp`
3. Walks all operations in the module
4. Prints ModuleOp to stdout with generic form, debug info, and value users

## Testing

The project includes comprehensive ORT integration tests. For detailed testing instructions, see [doc/TESTING.md](doc/TESTING.md).

### Step 1: Generate Test Models

```bash
# From onnx-hipdnn-ep directory
cd test
python gen_conv_model.py          # Generate conv_model.onnx
python gen_conv_gemm_model.py     # Generate conv_gemm_model.onnx
cd ..
```

### Step 2: Prepare Test Environment

Copy test models and set PATH to include DLL locations:

```bash
# Copy test models to build output directory
cp test/*.onnx ../build/morphizen-mlir/bin/Release/

# Set PATH to include DLL locations
# Windows CMD:
set PATH=..\local\bin;..\build\morphizen-mlir\bin\Release;%PATH%

# Or in Git Bash:
export PATH="../local/bin:../build/morphizen-mlir/bin/Release:$PATH"
```

### Step 3: Run Tests

```bash
# Change to the test executable directory
cd ../build/morphizen-mlir/bin/Release

# Run the test executable
./ort_integration_test.exe
```

### Expected Test Results

```
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from OrtIntegrationTest
[ RUN      ] OrtIntegrationTest.LoadVitisAIProvider
[       OK ] OrtIntegrationTest.LoadVitisAIProvider
[ RUN      ] OrtIntegrationTest.CreateSessionWithVitisAIProvider
[       OK ] OrtIntegrationTest.CreateSessionWithVitisAIProvider
[ RUN      ] OrtIntegrationTest.CreateSessionWithConvGemmModel
[       OK ] OrtIntegrationTest.CreateSessionWithConvGemmModel
[----------] 3 tests from OrtIntegrationTest
[  PASSED  ] 3 tests.
```

### Quick Test Summary (from onnx-hipdnn-ep directory)

```bash
# Generate and copy models
cd test && python gen_conv_model.py && python gen_conv_gemm_model.py && cd ..
cp test/*.onnx ../build/morphizen-mlir/bin/Release/

export PATH="../local/bin:../build/morphizen-mlir/bin/Release:$PATH"

# Run tests
cd ../build/morphizen-mlir/bin/Release
./ort_integration_test.exe
```

### Test Models

| Model | Description | Input Shape | Output Shape |
|-------|-------------|-------------|--------------|
| `conv_model.onnx` | Simple Conv operation | 1x3x8x8 | 1x16x8x8 |
| `conv_gemm_model.onnx` | Conv + Flatten + Gemm | 1x3x8x8 | 1x32 |

### Environment Variables

The MLIR backend is now **enabled by default** in MorphiZen (as of PR #530), so you no longer need to set environment variables for basic usage.

For debug output:
```bash
set MORPHIZEN_DEBUG_MLIR=2
set GLOG_logtostderr=1
set GLOG_minloglevel=0
```

## Differences from morphizen-demo

This project extends morphizen-demo with:

- ✅ **Full MLIR integration** with parsing and operation walking
- ✅ **Test infrastructure** with ORT integration tests
- ✅ **Multiple test models** (Conv, Conv+Gemm)
- ✅ **Test utilities** for easy model generation
- ❌ No pattern matching (no `patterns/` directory)
- ❌ No protobuf definitions (no `proto/` directory)
- ❌ No custom operators (no `custom-op-*/` directory)

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
