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
│   └── presets.cmake          # CMake presets configuration
├── level-1-pass-mlir/         # MLIR pass implementation
│   ├── CMakeLists.txt         # Pass build configuration
│   └── src/
│       └── pass_main.cpp      # Main pass implementation
├── etc/                        # Configuration files
│   └── vaip_config.json       # VAIP pass configuration
└── tools/                      # Build tools
    └── initialize-cmake-preset.py  # CMake preset generator
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
- Python 3
- Ninja build system
- Git

### Build Dependencies

The build process requires the following dependencies:

1. **ONNXRuntime** (required - must be built first)
2. **LLVM/MLIR** (automatically fetched via CMake FetchContent)
3. **MorphiZen** (automatically fetched via CMake FetchContent)

### Step-by-Step Build Instructions

#### Step 1: Build ONNXRuntime

ONNXRuntime must be built first as it's a core dependency:

```bash
# Clone ONNXRuntime
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime

# Build and install (Debug configuration)
./build.bat --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local

# Install
cmake --build ../build/onnxruntime/Debug/ --target install
```

#### Step 2: Build morphizen-mlir

Once ONNXRuntime is built, you can build morphizen-mlir. LLVM/MLIR and MorphiZen will be automatically fetched via CMake FetchContent:

**Option A: Using the build script (recommended)**
```bash
./build.bat
```

**Option B: Manual CMake commands**
```bash
# Generate CMake configuration with ORT Bridge and MLIR Backend enabled
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -S . -B build -DCMAKE_INSTALL_PREFIX=../local -DCMAKE_PREFIX_PATH=../local -Dmorphizen_ENABLE_ORT_BRIDGE=ON -Dmorphizen_ENABLE_MLIR_BACKEND=ON

# Build the project
cmake --build build

# Install the project
cmake --install build
```

**Note:** MorphiZen framework will be automatically fetched via CMake FetchContent during this step.

### Quick Build Summary

```bash
# 1. Build ONNXRuntime (in parent directory)
cd ../onnxruntime
./build.bat --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ../build/onnxruntime/Debug/ --target install

# 2. Build morphizen-mlir (LLVM/MLIR and MorphiZen will be auto-fetched)
cd ../morphizen-mlir
./build.bat
```

### Optional: Pre-build LLVM/MLIR

If you want to pre-build LLVM/MLIR instead of using FetchContent (which can take several hours during the first build), you can use the provided script:

```bash
cd morphizen-mlir
./build_llvm.bat
```

This will:
- Clone LLVM project to `../llvm`
- Checkout commit `f8cb7987c64dcffb72414a40560055cb717dbf74`
- Build with Ninja
- Install to `../local`

The build system will automatically detect and use the pre-built LLVM if available.

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

## Differences from morphizen-demo

This project is intentionally simplified compared to morphizen-demo:

- ❌ No pattern matching (no `patterns/` directory)
- ❌ No protobuf definitions (no `proto/` directory)
- ❌ No custom operators (no `custom-op-*/` directory)
- ❌ No test infrastructure (no `test/` directory)
- ✅ Minimal level-1 pass only
- ✅ Clean template for MLIR integration

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
