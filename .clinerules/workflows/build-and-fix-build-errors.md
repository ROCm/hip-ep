<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.
-->
# Build onnx-hipdnn-ep Project

Please build the onnx-hipdnn-ep project and fix any build errors that occur.

## Build Settings (Remember these):
- **Build Type**: Release (default)
- **Generator**: Visual Studio 17 2022|Visual Studio 18 2026
- **Architecture**: x64
- **Build Directory**: `../build/onnx-hipdnn-ep`
- **Install Prefix**: `../local`
- **Working Directory**: `c:/Develop/m/Source/onnx-hipdnn-ep`

## Prerequisites

Ensure ONNXRuntime is already built and installed in `../local`. If not, build it first:

```powershell
cd ..\onnxruntime
.\build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ..\build\onnxruntime --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/../local
cmake --build ..\build\onnxruntime\Release\ --target install
```

## Build Steps

### Option 1: Using build.bat (Recommended)

```powershell
cd C:\Develop\m\Source\onnx-hipdnn-ep
.\build.bat
```

### Option 2: Manual CMake Configuration

1. Configure the project:
   ```powershell
   cd C:\Develop\m\Source\onnx-hipdnn-ep
   cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -S . -B ..\build\onnx-hipdnn-ep -DCMAKE_INSTALL_PREFIX=..\local -DCMAKE_PREFIX_PATH=$PWD\..\local
   ```

2. Build the project:
   ```powershell
   cmake --build ..\build\onnx-hipdnn-ep --config Release
   ```

## Expected Outputs

Build artifacts are located in `..\build\onnx-hipdnn-ep\bin\Release\`:
- `ort_integration_test.exe` - Integration test executable
- `morphizen-level1-pass-mlir.dll` - MLIR pass plugin
- Test model files (after running test generators)

## Common Build Issues

### Missing ONNXRuntime
**Error:** Cannot find ONNXRuntime package or headers
**Fix:** Build and install ONNXRuntime first (see Prerequisites section)

### LLVM/MLIR Build Takes Too Long
**Note:** First build takes 1-3 hours as LLVM/MLIR is fetched and compiled via FetchContent
**Solution:** This is expected. Subsequent builds are much faster.

### MSVC Environment Issues
If build fails with missing standard headers, initialize MSVC environment:
```cmd
cmd /c "call ""C:\msvsn2022\VC\Auxiliary\Build\vcvars64.bat"" && cd /d C:\Develop\m\Source\onnx-hipdnn-ep && cmake --build ""..\build\onnx-hipdnn-ep"" --config Release"
```

## Verifying Build Outputs

Use PowerShell cmdlets (NOT CMD-style commands) to list build outputs:
```powershell
# CORRECT - use Get-ChildItem
Get-ChildItem "..\build\onnx-hipdnn-ep\bin\Release" -Name

# WRONG - do NOT use CMD-style dir /b
# dir "..\build\onnx-hipdnn-ep\bin\Release" /b
```

## Running Tests After Build

See [doc/TESTING.md](../doc/TESTING.md) for comprehensive testing instructions.

Quick test:
```powershell
# Generate test models
cd test; python gen_conv_model.py; python gen_conv_gemm_model.py; cd ..

# Copy models to build output
Copy-Item test\*.onnx ..\build\onnx-hipdnn-ep\bin\Release\

# Run tests
..\build\onnx-hipdnn-ep\bin\Release\ort_integration_test.exe
