<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MorphiZen HIP DNN

An implementation of HIP DNN operations in the MorphiZen framework.

This project demonstrates the integration of HIP (Heterogeneous-compute Interface for Portability) DNN operations within the MorphiZen optimization framework for ONNX Runtime.

---

## Build Instructions

### Build ONNXRuntime

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
./build.bat --use_vitisai --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ../build/onnxruntime/Debug/ --target install
```

This script is used to build the ONNX Runtime project with specific configurations and options. Below is a breakdown of the command-line arguments used:

- `--use_vitisai`: Enables the use of Vitis AI for acceleration.
- `--config Debug`: Specifies the build configuration as Debug.
- `--build_shared_lib`: Builds the project as a shared library.
- `--parallel`: Enables parallel compilation for faster build times.
- `--compile_no_warning_as_error`: Prevents warnings from being treated as errors during compilation.
- `--skip_submodule_sync`: Skips the synchronization of submodules, assuming they are already up-to-date.
- `--build_dir ../build/onnxruntime`: Specifies the directory where the build artifacts will be generated.
- `--skip_tests`: Skips the execution of tests after the build process.
- `--cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local`: Passes additional CMake definitions, setting the installation prefix to a local directory relative to the current working directory.

#### Build morphizen-hipdnn

#### Download morphizen and morphizen-hipdnn

```bash
cd workspace
git clone git@gitenterprise.xilinx.com:VitisAI/morphizen-hipdnn.git
git clone git@gitenterprise.xilinx.com:VitisAI/MorphiZen.git --recursive
```

#### Configure and build

```bash
cd morphizen-hipdnn
cmake -DBUILD_SHARED_LIBS=OFF -B ../build/morphizen-hipdnn -S . -DCMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ../build/morphizen-hipdnn --config Debug --target install
```

This script demonstrates the steps to configure, build, and install the `morphizen-hipdnn` project.

Steps:
1. Navigate to the `morphizen-hipdnn` directory.
2. Run the `cmake` command to configure the project, specify the build directory (`../build/morphizen-hipdnn`), and set the installation prefix to a local directory.
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

### Environment Variables

To assist in debugging and enhance logging during execution, you can configure the following environment variables:

- **`XLNX_ONNX_EP_VERBOSE`**: Enables verbose logging for the Xilinx ONNX Execution Provider, providing detailed insights into its operations.
- **`DEBUG_LOG_LEVEL`**: Sets the debug log level to control the granularity of logging output for troubleshooting purposes.

Adjust these variables as needed to streamline the debugging process.

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
