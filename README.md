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
cmake -DCMAKE_CXX_FLAGS="/EHsc" -B ../build/morphizen-hipdnn -S . -DCMAKE_INSTALL_PREFIX=$PWD/../local -DTHEROCK_DIST="/path/to/dist/therock" -DHIP_PLATFORM=amd
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
  - **test_classification**: Classification test executable for ONNX models (see [test/README_CLASSIFICATION.md](test/README_CLASSIFICATION.md))

### Environment Variables

To assist in debugging and enhance logging during execution, you can configure the following environment variables:

- **`USE_ORT_API_2_0`**: Enable ONNX Runtime API 2.0 compatibility (set to 1)
- **`XLNX_ONNX_EP_VERBOSE`**: Enables verbose logging for the Xilinx ONNX Execution Provider, providing detailed insights into its operations.
- **`DEBUG_LOG_LEVEL`**: Sets the debug log level to control the granularity of logging output for troubleshooting purposes.
- **`VITISAI_EP_JSON_CONFIG`**: Path to VitisAI EP configuration file
- **`XLNX_USE_CACHE_DIR`**: Directory for cache storage
- **`XLNX_USE_CACHE_KEY`**: Cache key for model compilation

Adjust these variables as needed to streamline the debugging process.

## Testing

### Classification Test

The project includes a classification test executable that demonstrates ONNX model inference with VitisAI EP support.

#### Quick Start

1. **Pull ONNX model** (Git LFS):
   ```bash
   git lfs install
   git lfs pull
   ```

2. **Generate input binary**:
   ```bash
   cd test/data
   python image_to_bin.py resnet50.jpg -o input.bin
   cd ../..
   ```

3. **Build with classification test**:
   ```bash
   cmake -B build -DBUILD_TEST_CLASSIFICATION=ON
   cmake --build build --target test_classification --config Release
   ```

4. **Run the test**:
   ```bash
   # Windows
   set USE_ORT_API_2_0=1
   set DEBUG_LOG_LEVEL=info
   .\build\test\Release\test_classification.exe test\data\pt_resnet50.onnx test\data\input.bin
   
   # Linux/macOS
   export USE_ORT_API_2_0=1
   export DEBUG_LOG_LEVEL=info
   ./build/test/test_classification test/data/pt_resnet50.onnx test/data/input.bin
   ```

#### Expected Output

```
================VitisAIExecutionProviderenable_ep = true
HIP Library Path: C:\Windows\SYSTEM32\amdhip64_7.dll
Running model...
done
batch_index: 0
score[109]  =  0.997308     text: brain coral,,
score[973]  =  0.00116773   text: coral reef,,
score[5]    =  0.000909427  text: electric ray, crampfish, numbfish, torpedo,,
score[397]  =  0.000158035  text: puffer, pufferfish, blowfish, globefish,,
score[955]  =  0.000123078  text: jackfruit, jak, jack,,
```

For detailed instructions, see [doc/resnet50_e2e_test.md](doc/resnet50_e2e_test.md).

#### Test Data

Test data files are located in `test/data/`:
- `pt_resnet50.onnx` (102 MB) - ResNet50 ONNX model (managed by Git LFS)
- `resnet50.jpg` (58 KB) - Test image (managed by Git LFS)
- `input.bin` (602 KB) - Generated input binary (created using image_to_bin.py)

#### Image to Binary Tool

The `image_to_bin.py` tool converts images to binary format for model inference:

```bash
# Basic usage
python test/data/image_to_bin.py input.jpg -o output.bin

# Custom size
python test/data/image_to_bin.py input.jpg --size 256 256 -o output.bin
```

See [test/data/README_image_to_bin.md](test/data/README_image_to_bin.md) for more details.

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
