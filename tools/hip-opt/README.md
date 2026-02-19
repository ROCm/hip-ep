# HIP MLIR Dialect Compiler

This directory contains a custom MLIR dialect for HIP (Heterogeneous-compute Interface for Portability) operations and a compiler tool `hip-opt` to lower them to LLVM IR.

## Overview

The HIP dialect provides high-level operations for:
- Creating and destroying HIP runtime handles
- Allocating and freeing device memory
- Matrix multiplication via hipBLASLt (`hip.hipblaslt.matmul`)
- Normalization, activation, and attention ops via MIOpen
- Custom HIP kernel ops (gather, silu, etc.)

All compute ops use **Destination-Passing Style (DPS)**: arguments are split into
`ins(...)` (read-only inputs) and `outs(...)` (output destinations that the caller provides).

These operations are lowered to LLVM IR function calls that interface with the HIP runtime, hipBLASLt, and MIOpen libraries.

## Files

- `HipDialect.td` - Dialect definition
- `HipTypes.td` - Type definitions (e.g., `!hip.handle`)
- `HipOps.td` - Operation definitions (DPS `ins`/`outs` format)
- `HipDialect.h/cpp` - Dialect C++ implementation
- `HipPasses.h` - Pass registration
- `HipToLLVM.cpp` - Conversion pass from HIP dialect to LLVM dialect
- `hip-opt.cpp` - Main compiler driver
- `examples/test.mlir` - Example MLIR file using HIP dialect (memory ops)
- `examples/test_gemm.mlir` - Two chained matmuls in DPS style
- `examples/test_e2e.mlir` - Self-contained transformer layer (E2E test)
- `examples/model_hip.mlir` - Generated HIP dialect from Llama-3.2-1B
- `ops_runtime/hip_runtime.cpp` - Handle lifecycle + device memory (shared by all tests)
- `ops_runtime/hipblaslt_matmul.cpp` - hipBLASLt matmul runtime (full implementation)
- `ops_runtime/miopen_add.cpp` - MIOpen element-wise add runtime (full implementation)
- `ops_runtime/miopen_mul.cpp` - MIOpen element-wise mul runtime (full implementation)
- `ops_runtime/miopen_rms_norm.cpp` - MIOpen RMS normalization runtime (full implementation)
- `examples/main_gemm.cpp` - Main driver for the matmul test
- `examples/main_add.cpp` - Main driver for the add test
- `examples/main_mul.cpp` - Main driver for the mul test
- `examples/main_rms_norm.cpp` - Main driver for the rms_norm test
- `scripts/run_full_pipeline_hipblaslt.bat` - Matmul pipeline (hipBLASLt)
- `scripts/run_full_pipeline_miopen_add.bat` - Add pipeline (MIOpen)
- `scripts/run_full_pipeline_miopen_mul.bat` - Mul pipeline (MIOpen)
- `scripts/run_full_pipeline_miopen_rms_norm.bat` - RMS Norm pipeline (MIOpen)

## Prerequisites: Building TheRock (ROCm) on Windows

The pipeline requires the HIP runtime (`amdhip64.dll`) and hipBLAS-LT (`libhipblaslt.dll`), which are built from source using [TheRock](https://github.com/ROCm/TheRock).


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

Replace `gfx1151` with your GPU target. To build only the components needed for this project (HIP runtime + hipBLAS-LT), you can do a minimal build:

```cmd
cmake -B build -GNinja --preset windows-release ^
  -DTHEROCK_AMDGPU_TARGETS=gfx1151 ^
  -DTHEROCK_ENABLE_ALL=OFF ^
  -DTHEROCK_ENABLE_BLAS=ON
```

Note: `THEROCK_ENABLE_HIP_RUNTIME` and `THEROCK_ENABLE_COMPILER` are implicitly enabled as dependencies.

| Flag | What it builds |
|------|----------------|
| `THEROCK_ENABLE_BLAS=ON` | hipBLAS-LT library (`libhipblaslt.dll`) |

### Build

```cmd
cmake --build build --target therock-dist
```

This will take a long time (1+ hour). Once complete, the ROCm SDK will be at:

```
TheRock\build\dist\rocm\
  ├── bin\
  │   ├── amdhip64_7.dll, libhipblaslt.dll
  ├── include\    (hip/, hipblaslt/, ...)
  └── lib\        (amdhip64.lib, ...)
```

###  Set Environment Variable

Set `THEROCK_DIST` to point to the dist output for use by the build steps below:

```cmd
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
```

### Build Troubleshooting (TheRock)

- **`Could NOT find PkgConfig`**: Install `pkg-config-lite` via `winget install bloodrock.pkg-config-lite`, or via conda: `conda install -c conda-forge pkg-config`
- **`Failed to find ROCm root directory`** or **`does not contain the HIP runtime CMake package`**: A pre-existing ROCm/HIP SDK install (e.g. from conda `rocm-sdk-core`) conflicts with TheRock's build. Uninstall it: `pip uninstall rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-gfx1151`. After uninstalling, delete stale CMake caches in the failed sub-projects and re-run configure.
- **`CMAKE_Fortran_COMPILER gfortran is not a full path and was not found`**: hipBLASLt requires a Fortran compiler. Install via conda: `conda install -n llvm -c conda-forge gfortran`. Then delete the stale cache and rebuild: `del build\math-libs\BLAS\hipBLASLt\build\CMakeCache.txt`.
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

### Convert to LLVM IR

```bash
./hip-opt examples/test.mlir --convert-hip-to-llvm | mlir-translate --mlir-to-llvmir
```

## HIP Dialect Operations

### Lifecycle and Memory

- **`hip.create_handle() : !hip.handle`** -- Creates a HIP runtime handle.
- **`hip.destroy_handle(%handle) : !hip.handle`** -- Destroys a HIP runtime handle.
- **`hip.alloc(%handle, %dynamicSizes...) : memref<...>`** -- Allocates device memory.
- **`hip.free(%handle, %memref) : memref<...>`** -- Frees device memory.

### Compute Ops (DPS)

All compute ops use Destination-Passing Style: read-only inputs are in `ins(...)`,
output destinations are in `outs(...)`. The handle and scalar params are leading
positional arguments.

- **`hip.hipblaslt.matmul(%handle) ins(%A, %B : ...) outs(%C : ...)`** -- Matrix multiply C = A @ B via hipBLASLt.
- **`hip.miopen.rms_norm(%handle) ins(%input, %weight : ...) outs(%output : ...)`** -- RMS normalization.
- **`hip.miopen.skip_rms_norm(%handle) ins(%x, %skip, %weight : ...) outs(%output, %residual : ...)`** -- Fused Add + RMS normalization.
- **`hip.miopen.rope(%handle, %start_pos) ins(%cos, %sin : ...) outs(%q, %k : ...)`** -- Rotary positional embeddings.
- **`hip.miopen.add(%handle) ins(%A, %B : ...) outs(%C : ...)`** -- Element-wise add.
- **`hip.miopen.mul(%handle) ins(%A, %B : ...) outs(%C : ...)`** -- Element-wise multiply.
- **`hip.gather(%handle) ins(%indices, %table : ...) outs(%output : ...)`** -- Embedding table lookup.
- **`hip.silu(%handle) ins(%input : ...) outs(%output : ...)`** -- SiLU activation.
- **`hip.gqa(%handle, %layer, %start_pos, %seq_len) ins(%q, %k, %v : ...) outs(%kv_cache, %output : ...)`** -- Grouped query attention.

## Matmul End-to-End Pipeline

The full pipeline from MLIR to a runnable executable using two chained matmuls:

```
test_gemm.mlir  -->  hip-opt  -->  mlir-translate  -->  llc  -->  gemm.obj
                                                                      |
ops_runtime/all_runtime.cpp  -->  cl.exe  -->  runtime.obj            |
main_gemm.cpp                -->  cl.exe  -->  main.obj               |
                                                                      v
                                              link.exe  -->  matmul_test.exe
                                                (+ hipblaslt.lib, amdhip64.lib)
```

Requires `THEROCK_DIST` to point to the TheRock dist output (see [Prerequisites](#prerequisites-building-therock-rocm-on-windows)):

```cmd
set THEROCK_DIST=C:\path\to\TheRock\build\dist\rocm
```

### Using the pipeline script

```cmd
scripts\run_full_pipeline_hipblaslt.bat
```

### Manual steps

```cmd
REM Step 1: Compile MLIR to object file
hip-opt examples/test_gemm.mlir --convert-hip-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o gemm_lowered.mlir
mlir-translate gemm_lowered.mlir --mlir-to-llvmir -o gemm.ll
llc gemm.ll -filetype=obj -o gemm.obj

REM Step 2: Compile runtime and main
cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include ops_runtime\hip_runtime.cpp
cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include ops_runtime\hipblaslt_matmul.cpp
cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include main_gemm.cpp /Fo:main.obj

REM Step 3: Link and run
link gemm.obj hip_runtime.obj hipblaslt_matmul.obj main.obj /LIBPATH:%THEROCK_DIST%\lib amdhip64.lib hipblaslt.lib /out:matmul_test.exe

set PATH=%THEROCK_DIST%\bin;%PATH%
matmul_test.exe
```

## Type System

### `!hip.handle`
An opaque type representing a HIP runtime handle. Lowered to `!llvm.ptr` (opaque pointer type in address space 0) in LLVM.

## Pass Pipeline

The `--convert-hip-to-llvm` pass performs the following conversions:

1. **Type Conversion**: `!hip.handle` -> `!llvm.ptr` (opaque pointer type in address space 0)
2. **Operation Lowering**: Each HIP operation is converted to LLVM function calls
3. **Function Declaration**: Required HIP runtime functions are declared in the module

## Extending the Dialect

To add new operations:

1. Add the operation definition in `HipOps.td` with `ins`/`outs` DPS assembly format
2. Add a conversion pattern in `HipToLLVM.cpp`
3. Add a runtime implementation in `ops_runtime/`
4. Rebuild the project (TableGen will regenerate the necessary files)
