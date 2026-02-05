# HIP MLIR Dialect Compiler

This directory contains a custom MLIR dialect for HIP (Heterogeneous-compute Interface for Portability) operations and a compiler tool `hip-opt` to lower them to LLVM IR.

## Overview

The HIP dialect provides high-level operations for:
- Creating and destroying HIP runtime handles
- Allocating and freeing device memory

These operations are lowered to LLVM IR function calls that interface with the HIP runtime library.

## Files

- `HipDialect.td` - Dialect definition
- `HipTypes.td` - Type definitions (e.g., `!hip.handle`)
- `HipOps.td` - Operation definitions
- `HipDialect.h/cpp` - Dialect C++ implementation
- `HipPasses.h` - Pass registration
- `HipToLLVM.cpp` - Conversion pass from HIP dialect to LLVM dialect
- `hip-opt.cpp` - Main compiler driver
- `test.mlir` - Example MLIR file using HIP dialect

## Building

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
./hip-opt test.mlir --convert-hip-to-llvm
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
./hip-opt test.mlir --convert-hip-to-llvm | mlir-translate --mlir-to-llvmir
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
