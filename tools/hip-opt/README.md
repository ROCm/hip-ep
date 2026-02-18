# HIP MLIR Dialect Compiler

This directory contains a custom MLIR dialect for HIP (Heterogeneous-compute Interface for Portability) operations and a compiler tool `hip-opt` to lower them to LLVM IR.

## Overview

The HIP dialect provides high-level operations for:
- Creating and destroying HIP runtime handles
- Allocating and freeing device memory
- Matrix multiplication (GEMM) via hipDNN graph API

These operations are lowered to LLVM IR function calls that interface with the HIP runtime library and hipDNN.

## Files

- `HipDialect.td` - Dialect definition
- `HipTypes.td` - Type definitions (e.g., `!hip.handle`)
- `HipOps.td` - Operation definitions
- `HipDialect.h/cpp` - Dialect C++ implementation
- `HipPasses.h` - Pass registration
- `HipToLLVM.cpp` - Conversion pass from HIP dialect to LLVM dialect
- `hip-opt.cpp` - Main compiler driver
- `examples/test.mlir` - Example MLIR file using HIP dialect (memory ops)
- `examples/test_gemm.mlir` - GEMM example using `hip.gemm`
- `examples/test_e2e.mlir` - Self-contained transformer layer (E2E test)
- `examples/model_target_hip.mlir` - Target HIP dialect reference (full transformer)
- `examples/model_hip.mlir` - Generated HIP dialect from Llama-3.2-1B
- `hip_gemm_runtime.cpp` - Runtime wrapper implementing `hip_gemm_f32` via hipBLAS-LT
- `hip_gemm_runtime_hipdnn.cpp` - Runtime wrapper implementing `hip_gemm_f32` via hipDNN graph API
- `main_gemm.cpp` - Main driver for the GEMM end-to-end test
- `scripts/run_full_pipeline_hipblaslt.bat` - Script to compile MLIR through the full pipeline (hipBLAS-LT backend)
- `scripts/run_full_pipeline_hipdnn.bat` - Script to compile MLIR through the full pipeline (hipDNN backend)
- `scripts/run_e2e_pipeline.bat` - E2E transformer pipeline (all ops)

## Prerequisites: Building TheRock (ROCm) on Windows

The GEMM pipeline requires the HIP runtime (`amdhip64.dll`), hipBLAS-LT (`libhipblaslt.dll`), and hipDNN (`hipdnn_backend.dll` + hipBLASLt engine plugin), which are built from source using [TheRock](https://github.com/ROCm/TheRock).


### Install Build Tools

Using `winget` (recommended):

```cmd
winget install Git.Git -e --source winget --custom "/o:PathOption=CmdTools"
winget install cmake ninja-build.ninja ccache python strawberryperl bloodrock.pkg-config-lite
winget install --id Iterative.DVC --silent --accept-source-agreements
```

Configure git for long paths and symlinks:

```cmd
git config --global core.symlinks true
git config --global core.longpaths true
```

###  Clone and Fetch Sources

```cmd
:: Create and activate conda environment (if not already done)
conda create -n llvm python=3.12 pip cmake ninja pkg-config -y
conda activate llvm

git clone https://github.com/ROCm/TheRock.git
cd TheRock

pip install -r requirements.txt

python ./build_tools/fetch_sources.py
```

###  Configure

Open an **x64 Native Tools Command Prompt for VS 2022** (or run `vcvars64.bat`), then:

```cmd
cmake -B build -GNinja --preset windows-release ^
  -DTHEROCK_AMDGPU_TARGETS=gfx1151
```

Replace `gfx1151` with your GPU target. To build only the components needed for this project (HIP runtime + hipBLAS-LT + hipDNN + hipDNN plugins), you can do a minimal build:

```cmd
cmake -B build -GNinja --preset windows-release ^
  -DTHEROCK_AMDGPU_TARGETS=gfx1151 ^
  -DTHEROCK_ENABLE_ALL=OFF ^
  -DTHEROCK_ENABLE_BLAS=ON ^
  -DTHEROCK_ENABLE_HIPDNN=ON ^
  -DTHEROCK_ENABLE_HIPBLASLT_PLUGIN=ON
```

Note: `THEROCK_ENABLE_HIP_RUNTIME` and `THEROCK_ENABLE_COMPILER` are implicitly enabled as dependencies. The flags explained:

| Flag | What it builds |
|------|----------------|
| `THEROCK_ENABLE_BLAS=ON` | hipBLAS-LT library (`libhipblaslt.dll`) |
| `THEROCK_ENABLE_HIPDNN=ON` | hipDNN backend + frontend (`hipdnn_backend.dll`) |
| `THEROCK_ENABLE_HIPBLASLT_PLUGIN=ON` | hipBLASLt engine plugin for hipDNN matmul |

The hipBLASLt plugin is **required** for hipDNN matmul/GEMM to work. hipDNN is a plugin-driven framework -- without a plugin that implements matmul, `graph.build()` will fail with "No engine configurations available."

```
hipDNN Architecture:
  Frontend (graph API)  -- user code builds graphs here
      |
  Backend (hipdnn_backend.dll)  -- dispatch layer
      |
  Engine Plugins (loaded at runtime from hipdnn_plugins/engines/)
    ├── MIOpen plugin (convolution, batchnorm)
    └── hipBLASLt plugin (matmul/GEMM)  <-- needed for GEMM
```

### 5. Build

```cmd
cmake --build build --target therock-dist
```

This will take a long time (1+ hour). Once complete, the ROCm SDK will be at:

```
TheRock\build\dist\rocm\
  ├── bin\
  │   ├── amdhip64_7.dll, libhipblaslt.dll, hipdnn_backend.dll
  │   └── hipdnn_plugins\engines\   (hipBLASLt plugin DLL)
  ├── include\    (hip/, hipblaslt/, hipdnn/, ...)
  └── lib\        (amdhip64.lib, hipdnn_backend.lib, ...)
```

###  Set Environment Variable

Set `THEROCK_DIST` to point to the dist output for use by the build steps below:

```cmd
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
```

### Build Troubleshooting (TheRock)

- **`No engine configurations available for the graph`** (hipDNN matmul runtime error): hipDNN requires a plugin that supports matmul. Rebuild TheRock with `-DTHEROCK_ENABLE_HIPBLASLT_PLUGIN=ON`. The plugin DLL will be installed to `dist/rocm/bin/hipdnn_plugins/engines/`. hipDNN auto-loads plugins from this directory relative to `hipdnn_backend.dll` at `hipdnnCreate()` time.
- **`Could NOT find PkgConfig`**: Install `pkg-config-lite` via `winget install bloodrock.pkg-config-lite`, or via conda: `conda install -c conda-forge pkg-config`
- **`Failed to find ROCm root directory`** or **`does not contain the HIP runtime CMake package`**: A pre-existing ROCm/HIP SDK install (e.g. from conda `rocm-sdk-core`) conflicts with TheRock's build. Uninstall it: `pip uninstall rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-gfx1151`. After uninstalling, delete stale CMake caches in the failed sub-projects (e.g. `build/core/hip-tests/build/CMakeCache.txt`, `build/ml-libs/hipDNN/build/CMakeCache.txt`) and re-run configure.
- **`CMAKE_Fortran_COMPILER gfortran is not a full path and was not found`**: hipBLASLt requires a Fortran compiler. Install via conda: `conda install -n llvm -c conda-forge gfortran`. Then delete the stale cache and rebuild: `del build\math-libs\BLAS\hipBLASLt\build\CMakeCache.txt`.
- **`PAL_CLIENT_INTERFACE_MAJOR_VERSION not supported`**: This is a warning, not an error. Can be safely ignored.
- **`amd_comgr version not compatible`**: The version mismatch warning (requested 2.9, found 3.0.0) is expected and handled by TheRock's dependency provider.
- **Long path errors**: Ensure long paths are enabled in git and Windows registry.

## Building hip-opt

Build LLVM and MLIR first (e.g. from `C:\local\llvm-project`):

```bash
cd C:\local\llvm-project
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host
cmake --build build
```

Then build hip-opt. If LLVM is at `C:\local\llvm-project`, CMake will auto-detect the build under `C:\local\llvm-project\build`:

```bash
mkdir build
cd build
cmake .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
# On Windows with LLVM at C:\local\llvm-project (built in build/):
# cmake .. -DLLVM_DIR=C:/local/llvm-project/build/lib/cmake/llvm -DMLIR_DIR=C:/local/llvm-project/build/lib/cmake/mlir
cmake --build .
```

## Usage

### Input MLIR with HIP Dialect

```mlir
module {
  func.func @example(%N: index) {
    %handle = hip.create_handle() : !hip.handle
    %x = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    hip.free(%handle, %x) : memref<?x128xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
```

### Convert to LLVM Dialect

```bash
./hip-opt examples/test.mlir --convert-hip-to-llvm
```

### Output LLVM Dialect

The conversion will generate LLVM dialect code with function declarations and calls:

```llvm
llvm.func @hipCreateHandle() -> !llvm.ptr
llvm.func @hipDestroyHandle(!llvm.ptr)
llvm.func @hipMalloc(i64) -> !llvm.ptr
llvm.func @hipFree(!llvm.ptr)

// In the function body:
%handle = llvm.call @hipCreateHandle() : () -> !llvm.ptr
%void_x = llvm.call @hipMalloc(%bytes) : (i64) -> !llvm.ptr
%x = llvm.bitcast %void_x : !llvm.ptr to !llvm.ptr<f32>
%free_x = llvm.bitcast %x : !llvm.ptr<f32> to !llvm.ptr
llvm.call @hipFree(%free_x) : (!llvm.ptr) -> ()
llvm.call @hipDestroyHandle(%handle) : (!llvm.ptr) -> ()
```

### Convert to LLVM IR

To get actual LLVM IR, you can further translate the LLVM dialect:

```bash
./hip-opt examples/test.mlir --convert-hip-to-llvm | mlir-translate --mlir-to-llvmir
```

## HIP Dialect Operations

### `hip.create_handle() : !hip.handle`
Creates and returns a HIP runtime handle.

### `hip.destroy_handle(%handle) : !hip.handle`
Destroys a HIP runtime handle.

### `hip.alloc(%handle, %dynamicSizes...) : memref<...>`
Allocates device memory and returns a memref. Supports dynamic dimensions.

### `hip.free(%handle, %memref) : memref<...>`
Frees device memory previously allocated with `hip.alloc`.

### `hip.gemm(%handle, %A, %B, %C, %M, %K, %N) : (...)`
Matrix multiply C = A @ B. A is MxK, B is KxN, C is MxN. All pointers must be device memory. Lowered to a call to `hip_gemm_f32()` which uses the hipDNN graph API.

## GEMM End-to-End Pipeline (hipBLASLt)

The full pipeline from MLIR to a runnable executable:

```
test_gemm.mlir  -->  hip-opt  -->  mlir-translate  -->  llc  -->  gemm.obj
                                                                      |
hip_gemm_runtime.cpp  -->  cl.exe  -->  runtime.obj                   |
main_gemm.cpp         -->  cl.exe  -->  main.obj                      |
                                                                      v
                                              link.exe  -->  gemm_test_hipblaslt.exe
                                                (+ hipblaslt.lib, amdhip64.lib)
```

Requires `THEROCK_DIST` to point to the TheRock dist output (see [Prerequisites](#prerequisites-building-therock-rocm-on-windows)):

```cmd
set THEROCK_DIST=C:\path\to\TheRock\build\dist\rocm
```

### Step 1: Compile MLIR to Object File

```bash
# Run the full pipeline script
scripts\run_full_pipeline_hipblaslt.bat

# Or manually:
hip-opt examples/test_gemm.mlir --convert-hip-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o gemm_lowered.mlir
mlir-translate gemm_lowered.mlir --mlir-to-llvmir -o gemm.ll
llc gemm.ll -filetype=obj -o gemm.obj
```

### Step 2: Compile Runtime and Main

```cmd
cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include hip_gemm_runtime.cpp /Fo:runtime.obj
cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include main_gemm.cpp /Fo:main.obj
```

### Step 3: Link and Run

```cmd
link gemm.obj runtime.obj main.obj /LIBPATH:%THEROCK_DIST%\lib amdhip64.lib hipblaslt.lib /out:gemm_test_hipblaslt.exe

:: Ensure runtime DLLs are in PATH
set PATH=%THEROCK_DIST%\bin;%PATH%
gemm_test_hipblaslt.exe
```

## GEMM End-to-End Pipeline (hipDNN)

The same MLIR code and main driver work with a hipDNN backend by swapping the runtime wrapper at link time. The MLIR lowering emits `llvm.call @hip_gemm_f32(...)` which is backend-agnostic.

**Prerequisites**: TheRock must be built with `-DTHEROCK_ENABLE_HIPBLASLT_PLUGIN=ON` so the matmul engine plugin is available at runtime.

**Note**: The hipDNN runtime wrapper must be compiled with TheRock's **clang++** (not MSVC `cl.exe`), because the hipDNN headers pull in HIP-specific types (`hip_bfloat16`, `__hip_bfloat16`) that use clang-only syntax.

```
test_gemm.mlir  -->  hip-opt  -->  mlir-translate  -->  llc  -->  gemm.obj
                                                                      |
hip_gemm_runtime_hipdnn.cpp  -->  clang++  -->  runtime.obj           |
main_gemm.cpp                -->  cl.exe   -->  main.obj              |
                                                                      v
                                              link.exe  -->  gemm_test_hipdnn.exe
                                                (+ hipdnn_backend.lib, amdhip64.lib)
```

### Using the hipDNN pipeline script

```cmd
scripts\run_full_pipeline_hipdnn.bat
```

### Manual steps

```cmd
set THEROCK_DIST=C:\path\to\TheRock\build\dist\rocm
set THEROCK_CLANG=C:\path\to\TheRock\build\compiler\amd-llvm\dist\lib\llvm\bin

:: Steps 1-3 are identical to the hipBLAS-LT pipeline (compile MLIR to gemm.obj)

:: Compile hipDNN runtime wrapper with clang++ (MSVC cl.exe cannot compile hipDNN headers)
%THEROCK_CLANG%\clang++.exe -c -std=c++17 -fms-extensions -fms-compatibility ^
  -D__HIP_PLATFORM_AMD__ -D__HIPCC__ -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY ^
  -I%THEROCK_DIST%\include -I%THEROCK_DIST%\include\hipdnn\frontend ^
  -I%THEROCK_DIST%\include\hipdnn\backend -I%THEROCK_DIST%\include\hipdnn\data_sdk ^
  -Wno-unused-value -Wno-c++11-narrowing ^
  hip_gemm_runtime_hipdnn.cpp -o runtime.obj

:: Compile main driver (same as hipBLAS-LT pipeline, uses MSVC cl.exe)
cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include main_gemm.cpp /Fo:main.obj

:: Link against hipdnn_backend.lib instead of hipblaslt.lib
link gemm.obj runtime.obj main.obj /LIBPATH:%THEROCK_DIST%\lib amdhip64.lib hipdnn_backend.lib /out:gemm_test_hipdnn.exe

set PATH=%THEROCK_DIST%\bin;%PATH%
gemm_test_hipdnn.exe
```

## Type System

### `!hip.handle`
An opaque type representing a HIP runtime handle. Lowered to `!llvm.ptr` (opaque pointer type in address space 0) in LLVM.

## Pass Pipeline

The `--convert-hip-to-llvm` pass performs the following conversions:

1. **Type Conversion**: `!hip.handle` → `!llvm.ptr` (opaque pointer type in address space 0)
2. **Operation Lowering**: Each HIP operation is converted to LLVM function calls
3. **Function Declaration**: Required HIP runtime functions are declared in the module

## Extending the Dialect

To add new operations:

1. Add the operation definition in `HipOps.td`
2. Add a conversion pattern in `HipToLLVM.cpp`
3. Rebuild the project (TableGen will regenerate the necessary files)
