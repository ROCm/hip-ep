<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Building & Testing Guide

Comprehensive instructions for building, configuring, and testing onnx-hipdnn-ep.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Directory Layout](#directory-layout)
- [Building ONNXRuntime](#building-onnxruntime)
- [Building onnx-hipdnn-ep](#building-onnx-hipdnn-ep)
- [Configuration Options](#configuration-options)
- [Testing](#testing)
- [Platform-Specific Notes](#platform-specific-notes)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Build-Time Requirements

**Required:**
- **CMake** 3.28 or later
- **Visual Studio 2022** (Windows) or equivalent C++17 compiler
- **Python 3** with onnx package: `pip install onnx`
- **Git** (for cloning repositories)
- **Clang and llvm-link** (for HIP runtime bitcode compilation)
  - Download from https://releases.llvm.org/ (LLVM 19+ recommended)
  - Or use system package manager:
    - Windows: `choco install llvm`
    - Linux: `apt install clang llvm` or `yum install clang llvm`
    - macOS: `brew install llvm`
  - Must include both `clang` and `llvm-link` executables in PATH or specified via `CMAKE_PROGRAM_PATH`

**Build Dependencies:**
1. **ONNXRuntime** (required - must be built first)
2. **LLVM/MLIR** (optional pre-build - can be auto-fetched via FetchContent)
3. **MorphiZen** (integrated as parent project)

**Note:** LLVM/MLIR pre-build is optional. The build system automatically fetches and builds LLVM via CMake FetchContent if not pre-installed. However, pre-building can save time on subsequent builds (first auto-build takes 1-3 hours).

### Runtime Requirements (Optional)

Only needed for executing compiled kernels on AMD GPUs:
- **HIP Runtime** (ROCm)
- **MIOpen** (DNN library)
- **hipBLASLt** (BLAS library)
- **ONNX Runtime** (built in previous step)

**Note:** Runtime dependencies are not required for building or testing the MLIR compilation pipeline.

**TheRock ROCm SDK Option:**
This project supports TheRock ROCm SDK (Windows). If using TheRock:
- Install TheRock SDK to `C:/Develop/m/dist/therock` (or custom location)
- Set `THEROCK_DIST` environment variable
- Set `HIP_PLATFORM=amd` environment variable
- Generate `hipblaslt.lib` import library (see TheRock Integration section below)
- Use `-DBUILD_MOCK_RUNTIME=OFF` when configuring CMake

## Directory Layout

Recommended workspace structure:

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

This layout assumes:
- Build directories are siblings to source directories
- Shared installation prefix at `../local` relative to each project
- Clean separation of build artifacts from source code

## Building ONNXRuntime

ONNXRuntime must be built first as a core dependency.

### Step 1: Clone ONNXRuntime

From workspace root:
```bash
git clone https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

### Step 2: Build and Install

**Release configuration (recommended):**

```bash
# If using Visual Studio 18 2026, upgrade CMake to >=4.2 (e.g., 4.2.3)
# and add: --cmake_generator "Visual Studio 18 2026"

./build.bat --config Release \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_submodule_sync \
  --build_dir ../build/onnxruntime \
  --skip_tests \
  --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local \
  --disable_memleak_checker

# Install to ../local
cmake --build ../build/onnxruntime/Release/ --target install
```

**Debug configuration (for development):**

Replace `--config Release` with `--config Debug` and build/install from `../build/onnxruntime/Debug/`.

## Building onnx-hipdnn-ep

### Step 1: Configure with CMake

**Windows (Visual Studio generator - recommended):**

```bash
cmake -S . -B ../build/onnx-hipdnn-ep \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DCMAKE_PREFIX_PATH=$PWD/../local \
  -DCMAKE_PROGRAM_PATH="C:/LLVM20/bin"
```

**Linux/macOS (Ninja or Unix Makefiles):**

```bash
cmake -S . -B ../build/onnx-hipdnn-ep \
  -G Ninja \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DCMAKE_PREFIX_PATH=$PWD/../local \
  -DCMAKE_PROGRAM_PATH="/usr/local/opt/llvm/bin"
```

**Note:** Replace `CMAKE_PROGRAM_PATH` with your actual LLVM installation path:
- Windows: `C:/LLVM20/bin`, `C:/Program Files/LLVM/bin`
- Linux: `/usr/bin`, `/usr/local/bin`
- macOS: `/usr/local/opt/llvm/bin`, `/opt/homebrew/opt/llvm/bin`

If `clang` and `llvm-link` are in your PATH, you can omit `CMAKE_PROGRAM_PATH`.

### Step 3: Build

```bash
cmake --build ../build/onnx-hipdnn-ep --config Release --parallel
```

**Note:** The first build may take 1-3 hours as LLVM/MLIR is fetched and compiled via FetchContent. Subsequent builds are much faster (incremental compilation).

### Quick Build Summary

Complete workflow from workspace root:

```bash
# 1. Build ONNXRuntime
cd onnxruntime
./build.bat --config Release --build_shared_lib --parallel \
  --compile_no_warning_as_error --skip_submodule_sync \
  --build_dir ../build/onnxruntime --skip_tests \
  --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local \
  --disable_memleak_checker
cmake --build ../build/onnxruntime/Release/ --target install

# 2. Build onnx-hipdnn-ep
cd ../onnx-hipdnn-ep
cmake -S . -B ../build/onnx-hipdnn-ep \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DCMAKE_PREFIX_PATH=$PWD/../local
cmake --build ../build/onnx-hipdnn-ep --config Release --parallel
```

## Configuration Options

### CMake Build Options

**BUILD_SHARED_LIBS** (default: OFF)
- Build shared libraries (.dll/.so) instead of static (.lib/.a)
- **Recommended:** OFF (static linking simplifies deployment)

**CMAKE_BUILD_TYPE** (default: Release)
- Build configuration: Release, Debug, RelWithDebInfo, MinSizeRel
- Only used with single-config generators (Ninja, Unix Makefiles)
- For Visual Studio, use `--config` flag at build time

**CMAKE_MSVC_RUNTIME_LIBRARY** (default: MultiThreadedDLL)
- MSVC runtime linkage: MultiThreadedDLL (/MD), MultiThreaded (/MT), MultiThreadedDebugDLL (/MDd), MultiThreadedDebug (/MTd)
- **Note:** Must match ONNXRuntime and LLVM runtime library settings

**CMAKE_PREFIX_PATH**
- Semicolon-separated list of installation prefixes to search for dependencies
- Example: `../local` (finds ONNXRuntime, pre-built LLVM)
- **CRITICAL:** Must be absolute path or CMake-resolvable relative path

**CMAKE_PROGRAM_PATH**
- Directories to search for executables (clang, llvm-link)
- Example: `C:/LLVM20/bin`

### MorphiZen Configuration

MorphiZen pass configuration is defined in `etc/morphizen_config.json`:

```json
{
  "passes": [
    {
      "name": "mlir-pass",
      "plugin": "morphizen-level1-pass-mlir",
      "target": "mlir-target"
    }
  ]
}
```

**Key Fields:**
- **name**: Pass identifier used in MorphiZen pipeline
- **plugin**: Shared library name (without extension)
- **target**: Target architecture identifier

To modify the configuration:
1. Edit `etc/morphizen_config.json`
2. Rebuild the project (CMake automatically copies to build output)
3. Re-run tests to verify changes

### Optional: Pre-building LLVM/MLIR

To avoid long first-time build (1-3 hours), you can pre-build LLVM/MLIR manually. The build system automatically detects and uses pre-built LLVM if available at `CMAKE_PREFIX_PATH`.

**Example (from workspace root):**

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

cmake -S llvm -B ../build/llvm-project \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../local \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DBUILD_SHARED_LIBS=OFF

cmake --build ../build/llvm-project --parallel
cmake --install ../build/llvm-project
```

After installation, onnx-hipdnn-ep will find LLVM at `../local` via `CMAKE_PREFIX_PATH`.

### TheRock ROCm SDK Integration

To build with GPU support using TheRock ROCm SDK:

#### One-Time Setup

1. **Install TheRock SDK** to `C:/Develop/m/dist/therock` (or custom location)

2. **Generate hipBLASLt Import Library**:

   TheRock provides `libhipblaslt.dll` but no MSVC `.lib` import library. Generate it manually:

   ```bash
   # Create .def file
   mkdir -p /c/Temp
   cat > /c/Temp/hipblaslt.def << 'EOF'
   LIBRARY libhipblaslt
   EXPORTS
       hipblasLtCreate
       hipblasLtDestroy
       hipblasLtMatrixLayoutCreate
       hipblasLtMatmulDescCreate
       hipblasLtMatmul
       hipblasLtMatrixLayoutDestroy
       hipblasLtMatmulDescDestroy
   EOF

   # Generate import library (requires MSVC lib.exe in PATH)
   lib.exe /def:C:/Temp/hipblaslt.def \
     /out:C:/Develop/m/dist/therock/lib/hipblaslt.lib \
     /machine:x64
   ```

#### Build Configuration

```bash
# Set environment variables
export THEROCK_DIST=C:/Develop/m/dist/therock
export HIP_PLATFORM=amd
LOCAL_DIR=$(cd ../../local && pwd)

# Configure with real runtime
cmake -S . -B ../../build/onnx-hipdnn-ep \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DONNX_MLIR_BUILD_TESTS=OFF \
  -DBUILD_MLIR_BACKEND=ON \
  -DBUILD_MLIR_HIP_OPT_TOOL=ON \
  -DONNX_HIP_INCLUDE_LIT_TESTS=ON \
  -DBUILD_MOCK_RUNTIME=OFF \
  --fresh

# Build
cmake --build ../../build/onnx-hipdnn-ep --config Debug --parallel
```

#### Verification

Check that real runtime (not mock) was built:

```bash
# Should show "Building REAL runtime" (not "Building mock runtime")
cat cmake_configure.log | grep -E "Building REAL|Building mock"

# Verify runtime.bc contains HIP calls (should be > 0)
C:/Develop/m/local/bin/llvm-dis.exe \
  ../../build/onnx-hipdnn-ep/mlir-compiler/lib/Runtime/runtime.bc -o - | \
  grep -c "hipMalloc\|miopenConvolution"
```

#### GPU Testing

See `GPU_TESTING.md` for complete GPU testing instructions.

## Testing

This project includes ORT integration tests that verify the MLIR compilation pipeline.

### Generating Test Models

From `test/` directory:

```bash
cd test

# Install onnx if not already installed
pip install onnx

# Generate ONNX test models
python gen_conv_model.py
python gen_conv_gemm_model.py
```

This creates:
- `conv_model.onnx` - Single Conv operation
- `conv_gemm_model.onnx` - Conv + Gemm operations

### Copying Models to Build Output

**Windows:**
```cmd
copy /Y *.onnx ..\..\build\onnx-hipdnn-ep\bin\Release\
```

**Linux/macOS:**
```bash
cp *.onnx ../../build/onnx-hipdnn-ep/bin/Release/
```

### Running Tests

Navigate to build output directory:

**Windows:**
```cmd
cd ..\..\build\onnx-hipdnn-ep\bin\Release\
ort_integration_test.exe
```

**Linux/macOS:**
```bash
cd ../../build/onnx-hipdnn-ep/bin/Release/
./ort_integration_test
```

### Expected Output

Successful test execution prints:
```
[INFO] Running test: SimpleConv
[INFO] Model loaded: conv_model.onnx
[INFO] MorphiZen pass executed
[INFO] MLIR module parsed and processed
[PASS] SimpleConv

[INFO] Running test: ConvGemm
[INFO] Model loaded: conv_gemm_model.onnx
[INFO] MorphiZen pass executed
[INFO] MLIR module parsed and processed
[PASS] ConvGemm

=== All tests passed ===
```

### Debug Mode Testing

To enable verbose MLIR output, modify `etc/morphizen_config.json`:

```json
{
  "passes": [
    {
      "name": "mlir-pass",
      "plugin": "morphizen-level1-pass-mlir",
      "target": "mlir-target",
      "debug": true
    }
  ]
}
```

Rebuild and re-run tests to see detailed MLIR IR output.

### Test Descriptions

**SimpleConv:**
- Tests basic ONNX Conv operation parsing
- Verifies MLIR module creation from ONNX graph
- Validates operation walking and printing

**ConvGemm:**
- Tests multi-operation ONNX graph (Conv + Gemm)
- Verifies handling of complex operation sequences
- Validates inter-operation dependencies


## Platform-Specific Notes

### Windows

**MSVC Developer Command Prompt:**
Launch bash/PowerShell from "Developer Command Prompt for VS 2022" to ensure Visual Studio tools are in PATH.

**Path Separators:**
Use forward slashes (`/`) in CMake paths for cross-platform compatibility. CMake automatically converts to backslashes on Windows.

**Clang Installation:**
- Download from https://releases.llvm.org/
- Or install via Chocolatey: `choco install llvm`
- Add to PATH or specify via `CMAKE_PROGRAM_PATH`

**Runtime Library:**
Ensure consistent runtime library across all dependencies:
- `/MD` (MultiThreadedDLL) for Release
- `/MDd` (MultiThreadedDebugDLL) for Debug

### Linux

**Package Manager Installation (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install cmake build-essential clang llvm python3 python3-pip git
pip3 install onnx
```

**Package Manager Installation (RHEL/CentOS):**
```bash
sudo yum install cmake gcc-c++ clang llvm python3 python3-pip git
pip3 install onnx
```

### macOS

**Homebrew Installation:**
```bash
brew install cmake llvm python3 git
pip3 install onnx
```

**LLVM Path:**
Homebrew installs LLVM to `/usr/local/opt/llvm` (Intel) or `/opt/homebrew/opt/llvm` (Apple Silicon). Add to `CMAKE_PROGRAM_PATH`:
```bash
cmake ... -DCMAKE_PROGRAM_PATH=/opt/homebrew/opt/llvm/bin
```

## Troubleshooting

### CMake Cannot Find ONNXRuntime

**Symptom:** `Could not find package onnxruntime`

**Solution:**
- Ensure ONNXRuntime is built and installed to `../local`
- Verify `CMAKE_PREFIX_PATH` points to `../local` (absolute path recommended)
- Check `../local/lib/cmake/onnxruntime/` exists

### Clang or llvm-link Not Found

**Symptom:** `Could not find program: clang` or `llvm-link not found`

**Solution:**
- Install LLVM (see Prerequisites)
- Add LLVM bin directory to PATH, or
- Specify via `CMAKE_PROGRAM_PATH=/path/to/llvm/bin`

### Long First Build Time

**Symptom:** Build takes 1-3 hours

**Solution:**
- This is expected for first build (LLVM/MLIR FetchContent)
- Subsequent builds are incremental (minutes)
- To avoid: Pre-build LLVM/MLIR manually (see Configuration Options)

### Runtime Library Mismatch

**Symptom:** Linker errors about mismatched runtime libraries (e.g., `/MD` vs `/MT`)

**Solution:**
- Ensure `CMAKE_MSVC_RUNTIME_LIBRARY` matches ONNXRuntime and LLVM builds
- For ONNXRuntime Release: use `MultiThreadedDLL` (/MD)
- For ONNXRuntime Debug: use `MultiThreadedDebugDLL` (/MDd)
- Rebuild all dependencies with consistent settings

### Tests Fail to Find Models

**Symptom:** `Cannot open model file: conv_model.onnx`

**Solution:**
- Generate models: `cd test && python gen_conv_model.py && python gen_conv_gemm_model.py`
- Copy to build output: `cp *.onnx ../../build/onnx-hipdnn-ep/bin/Release/`
- Run tests from build output directory: `cd ../../build/onnx-hipdnn-ep/bin/Release/ && ./ort_integration_test`

### MorphiZen Plugin Not Loaded

**Symptom:** `Failed to load plugin: morphizen-level1-pass-mlir`

**Solution:**
- Ensure plugin is built: check `../build/onnx-hipdnn-ep/bin/Release/morphizen-level1-pass-mlir.dll` (Windows) or `.so` (Linux)
- Verify `etc/morphizen_config.json` is copied to build output
- Check plugin name matches library name (without extension)

---

For additional help, see:
- [Architecture Documentation](design/ARCHITECTURE.md)
- [Project README](../README.md)
