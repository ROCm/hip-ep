# onnx-hipdnn-ep

A MLIR integration project for the MorphiZen framework.

## Overview

This project provides a level-1 MLIR pass for the MorphiZen framework. It serves as a template for MLIR-based graph transformations without pattern matching or protobuf dependencies.

## Project Structure

```
onnx-hipdnn-ep/
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
    └── morphizen_config.json  # MorphiZen pass configuration
```

## Key Features

- **Minimal Design**: No pattern matching, no protobuf dependencies
- **Level-1 Pass**: Simple MorphiZen pass structure for MLIR integration
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
├── onnx-hipdnn-ep/        # This project
├── build/
│   ├── onnxruntime/       # ONNXRuntime build output
│   └── onnx-hipdnn-ep/    # onnx-hipdnn-ep build output
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

#### Step 2: Build onnx-hipdnn-ep

Once ONNXRuntime is built, you can build onnx-hipdnn-ep. LLVM/MLIR and MorphiZen will be automatically fetched via CMake FetchContent:

**Using Visual Studio generator (recommended for Windows)**
```bash
cd onnx-hipdnn-ep

# Configure with Visual Studio generator
cmake -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
  -S . -B ../build/onnx-hipdnn-ep \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DCMAKE_PREFIX_PATH=$PWD/../local

# Build
cmake --build ../build/onnx-hipdnn-ep --config Release
```


**Note:** The first build will take a long time (1-3 hours) as LLVM/MLIR is fetched and compiled. Subsequent builds are much faster.

### Quick Build Summary

```bash
# From workspace root directory:

# 1. Build ONNXRuntime
cd onnxruntime
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ../build/onnxruntime/Release/ --target install

# 2. Build onnx-hipdnn-ep (LLVM/MLIR and MorphiZen will be auto-fetched)
cd ../onnx-hipdnn-ep
cmake -A x64 -DBUILD_SHARED_LIBS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -S . -B ../build/onnx-hipdnn-ep -DCMAKE_INSTALL_PREFIX=../local -DCMAKE_PREFIX_PATH=$PWD/../local
cmake --build ../build/onnx-hipdnn-ep --config Release --parallel
```

### Optional: Pre-build LLVM/MLIR

If you want to pre-build LLVM/MLIR instead of using FetchContent (which can take several hours during the first build), you can build it manually. The build system will automatically detect and use the pre-built LLVM if available in `../local`.

## Configuration

The MorphiZen configuration is defined in `etc/morphizen_config.json`:

- **Pass Name**: `mlir-pass`
- **Plugin**: `morphizen-level1-pass-mlir`
- **Target**: `mlir-target`

## Development

### Adding MLIR Logic

The main pass implementation is in `level-1-pass-mlir/src/pass_main.cpp`. The `Level1MlirPass::process()` method is where you would add MLIR transformation logic.


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

This project includes ORT integration tests. For comprehensive testing instructions, troubleshooting, and expected output details, see [doc/TESTING.md](doc/TESTING.md).

### Quick Start

```bash
# From onnx-hipdnn-ep directory

# 1. Generate test models
pip install onnx  （If your environment don't have onnx）
cd test && python gen_conv_model.py && python gen_conv_gemm_model.py

# 2. Copy models to build output
copy /Y *.onnx ..\..\build\onnx-hipdnn-ep\bin\Release\
or
cp *.onnx ../../build/onnx-hipdnn-ep/bin/Release/

# 3. Run tests
cd ..\..\build\onnx-hipdnn-ep\bin\Release\
or
cd ../../build/onnx-hipdnn-ep/bin/Release/

ort_integration_test.exe
```

For comprehensive information including:
- Test case descriptions
- Expected MLIR output examples
- Debug mode configuration
- Troubleshooting guide
- CI/CD integration

See [doc/TESTING.md](doc/TESTING.md).

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
