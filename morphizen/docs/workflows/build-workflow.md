<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Build Workflow

This document describes how to build MorphiZen using CMake.

## Prerequisites

### Required Tools
- **CMake** 3.23 or later
- **Ninja** build system (recommended)
- **C++ Compiler**:
  - Windows: MSVC 2022 (Visual Studio 17 2022)
  - Linux: GCC 9+ or Clang 10+
- **Python** 3.8+ (for build scripts and tools)

### Platform-Specific Setup

#### Windows
- Install Visual Studio 2022 with C++ development tools
- Install Ninja: Download from https://ninja-build.org/ or via Visual Studio Installer
- Run commands from an MSVC Developer Command Prompt (or launch Git Bash from Developer Command Prompt)

#### Linux
- Install build tools:
  ```bash
  sudo apt-get install build-essential cmake ninja-build
  ```

## Build Steps

### Linux

```bash
# Configure
cmake -B build -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dmorphizen_ENABLE_ORT_BRIDGE=ON \
  -Dmorphizen_ENABLE_ONNX_BACKEND=ON \
  -Dmorphizen_ENABLE_MLIR_BACKEND=OFF \
  -DCMAKE_PREFIX_PATH=/path/to/onnxruntime/install

# Build
cmake --build build
```

### Windows

From an MSVC Developer Command Prompt:

```cmd
REM Configure
cmake -G Ninja -B build -S . ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  -Dmorphizen_ENABLE_ORT_BRIDGE=ON ^
  -DCMAKE_PREFIX_PATH=C:\path\to\onnxruntime\install

REM Build
ninja -C build
```

Or using Git Bash (launched from Developer Command Prompt):

```bash
cmake -G Ninja -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
  -Dmorphizen_ENABLE_ORT_BRIDGE=ON \
  -DCMAKE_PREFIX_PATH=/c/path/to/onnxruntime/install

ninja -C build
```

## CMake Options

Key CMake options for MorphiZen:

- **`CMAKE_BUILD_TYPE`**: `Debug`, `Release`, or `RelWithDebInfo`
- **`CMAKE_PREFIX_PATH`**: Path to ONNXRuntime and other dependencies
- **`morphizen_ENABLE_ORT_BRIDGE`**: Enable ORT execution provider bridge (ON/OFF)
- **`morphizen_ENABLE_ONNX_BACKEND`**: Enable ONNX IR backend (ON/OFF)
- **`morphizen_ENABLE_MLIR_BACKEND`**: Enable MLIR backend (ON/OFF)
- **`morphizen_ENABLE_UNIT_TEST`**: Build unit tests (ON/OFF)

## Expected Build Outputs

After a successful build, you should find:

- **Windows**: `build/bin/Release/onnxruntime_vitisai_ep.dll` (main library)
- **Linux**: `build/bin/libonnxruntime_vitisai_ep.so` (main library)
- Various tools and test executables in `build/bin/`

## Build Configurations

- **Debug**: Debug symbols, assertions, no optimization (for development)
- **Release**: Optimized, no debug symbols (for production)
- **RelWithDebInfo**: Optimized with debug symbols (for profiling)

## Dependencies

MorphiZen uses CMake's FetchContent to automatically download and build dependencies:
- **onnxruntime**: Core ONNX Runtime library (via `CMAKE_PREFIX_PATH`)
- **protobuf**: Protocol Buffers
- **glog**: Logging library (optional)
- **MLIR**: Multi-Level Intermediate Representation (if enabled)

Dependencies are downloaded during configure and cached in `build/_deps/`.

## Troubleshooting

### Common Issues

**CMake version too old**
- Upgrade CMake to 3.23 or later

**Missing compiler (Windows)**
- Launch terminal from MSVC Developer Command Prompt

**Missing compiler (Linux)**
- Install build tools: `sudo apt install build-essential`

**Ninja not found**
- Windows: Install via Visual Studio Installer or download from https://ninja-build.org/
- Linux: `sudo apt install ninja-build`

**Out of memory during build**
- Reduce parallelism: `cmake --build build --parallel 4`

**Protobuf version mismatch**
- Clean and reconfigure: `rm -rf build/_deps/protobuf* && cmake -B build -S .`

## Reproducing CI Builds Locally

The CI workflows use direct CMake commands (see .github/workflows/build_and_test_*.yml). To reproduce CI builds:

```bash
# Clean build directory
rm -rf build

# Configure (same as CI)
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dmorphizen_ENABLE_ORT_BRIDGE=ON \
  -DCMAKE_PREFIX_PATH=/path/to/onnxruntime/install

# Build
cmake --build build
```

## Building Specific Targets

Build only what you need:

```bash
# Build only the main library
cmake --build build --target onnxruntime_vitisai_ep

# Build only tests
cmake --build build --target morphizen-unit-tests

# List all available targets
cmake --build build --target help
```

## Related Documentation

- **[Developer Guide](../developer-guide.md)**: Development environment setup and testing
- **[Git Workflow](git-workflow.md)**: Version control workflow
- **[PR Workflow](pr-workflow.md)**: Pull request process
- **[Architecture](../architecture.md)**: System design and components
