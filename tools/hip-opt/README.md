# HIP MLIR Dialect Compiler

This directory contains a custom MLIR dialect for HIP (Heterogeneous-compute Interface for Portability) operations and a compiler tool `hip-opt` to lower them to LLVM IR.

## Overview

The HIP dialect provides high-level operations for GPU inference on AMD ROCm, targeting LLM workloads with 3D tensors `[B, S, D]` (batch, sequence, model_dim):

- **Lifecycle & memory**: `hip.create_handle`, `hip.destroy_handle`, `hip.alloc`, `hip.free`
- **hipBLASLt**: `hip.hipblaslt.matmul` -- batched GEMM with weight broadcast (3D x 2D)
- **MIOpen**: `hip.miopen.add`, `hip.miopen.mul`, `hip.miopen.rms_norm`, `hip.miopen.softmax`
- **Custom kernels**: `hip.transpose` (N-D with dim params)

All compute ops use **Destination-Passing Style (DPS)**: arguments are split into
`ins(...)` (read-only inputs) and `outs(...)` (output destinations that the caller provides).

All lowerings are **rank-generic**: ops detect tensor rank at compile time and pass appropriate shape metadata to the runtime (batched GEMM for 3D matmul, flattened dims for MIOpen ops, etc.).

## Files

- `HipDialect.td` - Dialect definition
- `HipTypes.td` - Type definitions (e.g., `!hip.handle`)
- `HipOps.td` - Operation definitions (DPS `ins`/`outs` format)
- `HipDialect.h/cpp` - Dialect C++ implementation
- `HipPasses.h` - Pass registration
- `HipToLLVM.cpp` - Conversion pass from HIP dialect to LLVM dialect
- `hip-opt.cpp` - Main compiler driver
- `CMakeLists.txt` - Build configuration

### Runtime (per-op implementations)

- `ops_runtime/hip_runtime.cpp` - Handle lifecycle + device memory (shared by all tests)
- `ops_runtime/hipblaslt_matmul.cpp` - hipBLASLt matmul with batched GEMM + broadcast
- `ops_runtime/miopen_add.cpp` - MIOpen element-wise add
- `ops_runtime/miopen_mul.cpp` - MIOpen element-wise mul
- `ops_runtime/miopen_rms_norm.cpp` - MIOpen RMS normalization (`MIOPEN_BETA_API` required)
- `ops_runtime/miopen_softmax.cpp` - MIOpen softmax
- `ops_runtime/transpose.cpp` - N-D transpose (pure C++, swaps two specified dims)

### Examples and Tests

- `examples/test_gemm.mlir` + `main_gemm.cpp` - 3D matmul with 2D weight broadcast
- `examples/test_add.mlir` + `main_add.cpp` - 3D element-wise add
- `examples/test_mul.mlir` + `main_mul.cpp` - 3D element-wise mul
- `examples/test_rms_norm.mlir` + `main_rms_norm.cpp` - 3D RMS normalization
- `examples/test_softmax.mlir` + `main_softmax.cpp` - 3D softmax
- `examples/test_attention.mlir` + `main_attention.cpp` - Full single-head attention (B=2)
- `examples/test.mlir` - Basic memory ops example
- `examples/model_hip.mlir` - Generated HIP dialect from Llama-3.2-1B

### Pipeline Scripts

- `scripts/run_full_pipeline_hipblaslt.bat` - Matmul test (hipBLASLt)
- `scripts/run_full_pipeline_miopen_add.bat` - Add test (MIOpen)
- `scripts/run_full_pipeline_miopen_mul.bat` - Mul test (MIOpen)
- `scripts/run_full_pipeline_miopen_rms_norm.bat` - RMS Norm test (MIOpen)
- `scripts/run_full_pipeline_miopen_softmax.bat` - Softmax test (MIOpen)
- `scripts/run_full_pipeline_attention.bat` - Attention test (hipBLASLt + MIOpen + custom)

## Prerequisites: Building TheRock (ROCm) on Windows

The pipeline requires the HIP runtime (`amdhip64.dll`), hipBLAS-LT (`libhipblaslt.dll`), and MIOpen (`MIOpen.dll`), built from source using [TheRock](https://github.com/ROCm/TheRock).

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

Replace `gfx1151` with your GPU target. For a minimal build with hipBLASLt + MIOpen:

```cmd
cmake -B build -GNinja --preset windows-release ^
  -DTHEROCK_AMDGPU_TARGETS=gfx1151 ^
  -DTHEROCK_ENABLE_ALL=OFF ^
  -DTHEROCK_ENABLE_BLAS=ON ^
  -DTHEROCK_ENABLE_MIOPEN=ON
```

| Flag | What it builds |
|------|----------------|
| `THEROCK_ENABLE_BLAS=ON` | hipBLAS-LT library (`libhipblaslt.dll`) |
| `THEROCK_ENABLE_MIOPEN=ON` | MIOpen library (`MIOpen.dll`) |

Note: `THEROCK_ENABLE_HIP_RUNTIME` and `THEROCK_ENABLE_COMPILER` are implicitly enabled.

### Build

```cmd
cmake --build build --target therock-dist
```

This will take a long time (1+ hour). Once complete, the ROCm SDK will be at:

```
TheRock\build\dist\rocm\
  ├── bin\      (amdhip64_7.dll, libhipblaslt.dll, MIOpen.dll)
  ├── include\  (hip/, hipblaslt/, miopen/, ...)
  └── lib\      (amdhip64.lib, ...)
```

###  Set Environment Variable

```cmd
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
```

### Build Troubleshooting (TheRock)

- **`Could NOT find PkgConfig`**: `winget install bloodrock.pkg-config-lite`
- **`Failed to find ROCm root directory`**: Uninstall conflicting ROCm packages: `pip uninstall rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-gfx1151`
- **`CMAKE_Fortran_COMPILER gfortran is not a full path`**: `conda install -n llvm -c conda-forge gfortran`
- **Long path errors**: Enable long paths in git and Windows registry.

## Building hip-opt

Build LLVM and MLIR first:

```bash
cd C:\local\llvm-project
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host
cmake --build build
```

Then build hip-opt:

```bash
mkdir build && cd build
cmake .. -DLLVM_DIR=C:/local/llvm-project/build/lib/cmake/llvm -DMLIR_DIR=C:/local/llvm-project/build/lib/cmake/mlir
cmake --build .
```

## Usage

### Input MLIR with HIP Dialect (3D tensors)

```mlir
module {
  func.func @example(
      %A: memref<?x?x?xf32, 1>,   // [B, S, D]
      %W: memref<?x?xf32, 1>,     // [D, D]  (2D weight, broadcast)
      %C: memref<?x?x?xf32, 1>) { // [B, S, D]
    %handle = hip.create_handle() : !hip.handle
    hip.hipblaslt.matmul(%handle)
        ins(%A, %W : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%C : memref<?x?x?xf32, 1>)
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
```

### Compile Pipeline

```bash
hip-opt input.mlir \
    --convert-hip-to-llvm \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir -o output.ll
llc output.ll -filetype=obj -o output.obj
```

## HIP Dialect Operations

### Lifecycle and Memory

- **`hip.create_handle() : !hip.handle`** -- Creates a HIP runtime handle.
- **`hip.destroy_handle(%handle) : !hip.handle`** -- Destroys a HIP runtime handle.
- **`hip.alloc(%handle, %dynamicSizes...) : memref<...>`** -- Allocates device memory.
- **`hip.free(%handle, %memref) : memref<...>`** -- Frees device memory.

### Compute Ops (DPS, rank-generic)

All compute ops use Destination-Passing Style with rank-generic lowerings. For LLM workloads, tensors are typically `[B, S, D]` (batch, sequence, model_dim).

**hipBLASLt:**
- **`hip.hipblaslt.matmul(%handle) ins(%A, %B : ...) outs(%C : ...)`** -- Matrix multiply. Supports batched GEMM (3D) and weight broadcast (3D x 2D: `stride_B=0`).

**MIOpen:**
- **`hip.miopen.add(%handle) ins(%A, %B : ...) outs(%C : ...)`** -- Element-wise add.
- **`hip.miopen.mul(%handle) ins(%A, %B : ...) outs(%C : ...)`** -- Element-wise multiply.
- **`hip.miopen.rms_norm(%handle) ins(%input, %weight : ...) outs(%output : ...)`** -- RMS normalization. For 3D input, flattens `B*S` as rows.
- **`hip.miopen.softmax(%handle) ins(%input : ...) outs(%output : ...)`** -- Row-wise softmax over last dim. For 3D input, flattens `B*S` as rows.
- **`hip.miopen.skip_rms_norm(%handle) ins(%x, %skip, %weight : ...) outs(%output, %residual : ...)`** -- Fused Add + RMS normalization (stub).
- **`hip.miopen.rope(%handle, %start_pos) ins(%cos, %sin : ...) outs(%q, %k : ...)`** -- Rotary positional embeddings (stub).

**Custom kernels:**
- **`hip.transpose(%handle, %dim0, %dim1) ins(%input : ...) outs(%output : ...)`** -- N-D transpose swapping two specified dims.
- **`hip.gather(%handle) ins(%indices, %table : ...) outs(%output : ...)`** -- Embedding table lookup (stub).
- **`hip.silu(%handle) ins(%input : ...) outs(%output : ...)`** -- SiLU activation (stub).
- **`hip.gqa(%handle, %layer, %start_pos, %seq_len) ins(%q, %k, %v : ...) outs(%kv_cache, %output : ...)`** -- Grouped query attention (stub).

## End-to-End Pipeline Example

The matmul test: `A[B,S,K] @ B0[K,N] -> tmp -> tmp @ B1[N,P] -> C[B,S,P]`

```
test_gemm.mlir  -->  hip-opt  -->  mlir-translate  -->  llc  -->  gemm.obj
                                                                      |
ops_runtime/hip_runtime.cpp      -->  cl.exe  -->  hip_runtime.obj    |
ops_runtime/hipblaslt_matmul.cpp -->  cl.exe  -->  hipblaslt_matmul.obj
examples/main_gemm.cpp           -->  cl.exe  -->  main.obj           |
                                                                      v
                                              link.exe  -->  matmul_test.exe
                                                (+ hipblaslt.lib, amdhip64.lib)
```

```cmd
set THEROCK_DIST=C:\path\to\TheRock\build\dist\rocm
scripts\run_full_pipeline_hipblaslt.bat
```

## Type System

### `!hip.handle`
An opaque type representing a HIP runtime handle. Lowered to `!llvm.ptr` in LLVM.

## Pass Pipeline

The `--convert-hip-to-llvm` pass converts all HIP ops to LLVM function calls. Additional standard passes are needed:

```
--convert-hip-to-llvm          # HIP ops -> llvm.call @hip_*()
--finalize-memref-to-llvm      # memref.dim -> LLVM
--convert-arith-to-llvm        # arith.constant -> LLVM
--convert-func-to-llvm         # func.func -> LLVM
--reconcile-unrealized-casts   # Clean up type casts
```

## Extending the Dialect

To add new operations:

1. Add the op definition in `HipOps.td` with `ins`/`outs` DPS assembly format
2. Add a lowering pattern in `HipToLLVM.cpp` (extract shape from memref descriptors rank-generically)
3. Add a runtime implementation in `ops_runtime/` with the matching C function signature
4. Create a test: `examples/test_<op>.mlir` + `examples/main_<op>.cpp` + `scripts/run_full_pipeline_<op>.bat`
5. Rebuild hip-opt (TableGen regenerates from .td files)
