<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX HIP DNN Execution Provider

An implementation of HIP DNN operations in the MorphiZen framework.

This project demonstrates the integration of HIP (Heterogeneous-compute Interface for Portability) DNN operations within the MorphiZen optimization framework for ONNX Runtime.

> **Note**: This project was formerly known as `morphizen-hipdnn`. It has been renamed to `onnx-hipdnn-ep` to better reflect its role as an ONNX Runtime execution provider using HIP DNN. All references to the old name have been updated throughout the codebase.

---

## Prerequisites

### System Requirements
- Windows 10/11 with AMD GPU (ROCm support)
- Visual Studio 2022 with C++ workload
- CMake 3.29+
- Git with Git Bash
- Python 3.8+

---

## Directory Structure

After completing the build instructions below, your workspace will have the following structure:

```
workspace/
├── therock/                  # TheRock ROCm SDK (extracted from tarball)
│   ├── bin/                  # Runtime DLLs (MIOpen.dll, hiprtc.dll, etc.)
│   └── lib/llvm/bin/         # LLVM tools (amdgpu-arch.exe)
├── onnxruntime/              # ONNX Runtime source code (git clone)
├── build/
│   ├── onnxruntime/          # ONNX Runtime build artifacts
│   └── onnx-hipdnn-ep/       # onnx-hipdnn-ep build artifacts
├── local/                    # ONNX Runtime installation (CMAKE_PREFIX_PATH)
│   ├── bin/                  # onnxruntime.dll, onnxruntime_morphizen_ep.dll, test_classification.exe
│   └── lib/cmake/            # CMake configuration files
└── onnx-hipdnn-ep/           # This project (git clone)
    ├── test/data/            # Test data (pt_resnet50.onnx, input.bin)
    └── etc/                  # Configuration files (vaip_config.json)
```

---

## Build Instructions

### Step 1: Setup TheRock ROCm SDK

TheRock SDK provides HIP/ROCm runtime for Windows.

**Download:** https://therock-nightly-tarball.s3.amazonaws.com/index.html

1. **Determine your GPU architecture** (before downloading):

   Open **Device Manager** → **Display adapters** to find your AMD GPU model, then select the matching GFX series:

   | GPU Model | GFX Series | TheRock Tarball |
   |-----------|------------|-----------------|
   | Radeon RX 7900/7800/7700/7600 | gfx110X | `therock-dist-windows-gfx110X-all-*.tar.gz` |
   | Radeon RX 6900/6800/6700/6600 | gfx103X | `therock-dist-windows-gfx103X-all-*.tar.gz` |
   | Radeon 880M/780M (Strix Point) | gfx115X | `therock-dist-windows-gfx115X-all-*.tar.gz` |
   | Radeon 890M (Strix Halo) | gfx120X | `therock-dist-windows-gfx120X-all-*.tar.gz` |

2. **Create workspace and extract TheRock**:
   ```bash
   mkdir workspace
   cd workspace
   
   # Extract TheRock tarball to workspace/therock
   tar -xzf /path/to/therock-dist-windows-gfx115X-all-*.tar.gz
   mv dist therock
   ```

3. **Verify installation**:
   ```bash
   # Verify GPU detection
   ./therock/lib/llvm/bin/amdgpu-arch.exe
   # Example output: gfx1150

   # Verify HIP configuration
   ./therock/bin/hipconfig.exe --full
   ```

### Step 2: Build ONNXRuntime

To build ONNX Runtime, follow the [official documentation](https://onnxruntime.ai/docs/build/inferencing.html).

It is recommended to use Git Bash. The commands below have only been tested in Git Bash. They might also work in other shells with slight modifications.

#### Download onnxruntime

```bash
# Create a workspace directory
mkdir workspace
cd workspace

# Clone the ONNX Runtime repository
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

#### Build ONNX Runtime

```bash
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ../build/onnxruntime/Release/ --target install
```

This script is used to build the ONNX Runtime project with specific configurations and options. Below is a breakdown of the command-line arguments used:

- `--config Release`: Specifies the build configuration as Release.
- `--build_shared_lib`: Builds the project as a shared library.
- `--parallel`: Enables parallel compilation for faster build times.
- `--compile_no_warning_as_error`: Prevents warnings from being treated as errors during compilation.
- `--skip_submodule_sync`: Skips the synchronization of submodules, assuming they are already up-to-date.
- `--build_dir ../build/onnxruntime`: Specifies the directory where the build artifacts will be generated.
- `--skip_tests`: Skips the execution of tests after the build process.
- `--cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local`: Passes additional CMake definitions, setting the installation prefix to a local directory relative to the current working directory.

### Step 3: Build onnx-hipdnn-ep

#### Download onnx-hipdnn-ep

```bash
cd workspace
git clone https://github.com/ROCm/onnx-hipdnn-ep.git
```

#### Configure and build

```bash
cd onnx-hipdnn-ep
export THEROCK_DIST=$PWD/../therock
cmake \
  -B ../build/onnx-hipdnn-ep -S . \
  -DTHEROCK_DIST=$THEROCK_DIST \
  -DCMAKE_PREFIX_PATH=$PWD/../local \
  -DONNXRUNTIME_SOURCE_TREE_DIR=$PWD/../onnxruntime \
  -DHIP_PLATFORM=amd

# Build Release version (recommended)
cmake --build ../build/onnx-hipdnn-ep --config Release --target install

# Or build Debug version
cmake --build ../build/onnx-hipdnn-ep --config Debug --target install
```

This script demonstrates the steps to configure, build, and install the `onnx-hipdnn-ep` project.

Steps:
1. Navigate to the `onnx-hipdnn-ep` directory.
2. Run the `cmake` command to configure the project, specify the build directory (`../build/onnx-hipdnn-ep`), and set the installation prefix to a local directory.
3. Build the project in Debug configuration and install the resulting binaries to the specified installation directory.

## Project Design

In this project, we demonstrate how to integrate HIP DNN operations into the MorphiZen framework, including:
- Level-1 pass implementation for HIP DNN operations
- Custom operator kernels for HIP DNN
- Integration with ONNX Runtime execution provider

### Components

- **level-1-pass-hipdnn**: Graph optimization pass for HIP DNN operations
- **custom-op-hipdnn**: Custom operator implementations using HIP
- **proto**: Protocol buffer definitions
- **test**: Test suite for validation
  - **test_classification**: Classification test executable for ONNX models (see [doc/resnet50_e2e_test.md](doc/resnet50_e2e_test.md))

### Environment Variables

Essential HipDNN-specific environment variables:

- **`THEROCK_DIST`**: Path to TheRock SDK installation
- **`HIP_PLATFORM`**: Set to `amd` for AMD GPU support

For additional debugging variables and advanced configuration, see:
- [Linux Build Guide](doc/linux_build_guide.md#environment-variables-reference)
- [Windows Build Guide](doc/windows_build_guide.md#environment-variables-reference)

## Testing

The project includes a ResNet50 classification test that demonstrates end-to-end ONNX model inference with HipDNN execution provider support.

### Quick Test

```bash
# Generate test input
cd test/data && python image_to_bin.py resnet50.jpg -o input.bin && cd ../..

# Build and run
cmake -B build -DBUILD_TEST_CLASSIFICATION=ON
cmake --build build --target test_classification --config Release
export PATH="$THEROCK_DIST/bin;$PATH"
./build/test/test_classification test/data/pt_resnet50.onnx test/data/input.bin
```

### Documentation

- **Complete Testing Guide**: [doc/resnet50_e2e_test.md](doc/resnet50_e2e_test.md)
  - Detailed setup instructions
  - Expected output and validation
  - Command-line options
  - Image preprocessing tool usage

- **Platform-Specific Guides**:
  - [Linux Build & Testing](doc/linux_build_guide.md)
  - [Windows Build & Testing](doc/windows_build_guide.md)

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
