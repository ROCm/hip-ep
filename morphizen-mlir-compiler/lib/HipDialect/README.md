<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP MLIR Dialect

This directory contains the HIP (Heterogeneous-compute Interface for Portability) MLIR dialect for AMD ROCm GPU operations.

## Operations

### Memory Management

- `hip.alloc` - Allocate device memory (requires context parameter)
- `hip.free` - Free device memory (requires context parameter)

**Note:** Context lifecycle is managed externally via `inference_init()` and `inference_cleanup()`. Generated functions receive context as a parameter.

### MIOpen DNN Operations

#### Convolution

```mlir
%output = hip.conv(%handle, %input, %weights, %bias)
          {kernel_shape = [3, 3], strides = [1, 1],
           pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
          : (memref<?x?x?x?xf32, 1>, memref<?x?x?x?xf32, 1>, memref<?xf32, 1>)
          -> memref<?x?x?x?xf32, 1>
```

Attributes:
- `kernel_shape`: Array of kernel dimensions [H, W]
- `strides`: Array of stride values [H, W]
- `pads`: Array of padding values [top, left, bottom, right]
- `dilations`: Array of dilation values [H, W]
- `group`: Number of groups for grouped convolution

#### Matrix Multiplication (GEMM)

```mlir
%C = hip.gemm(%handle, %A, %B)
     {transA = 0, transB = 0, alpha = 1.0, beta = 0.0}
     : (memref<?x?xf32, 1>, memref<?x?xf32, 1>)
     -> memref<?x?xf32, 1>
```

Performs: C = alpha * A * B + beta * C

Attributes:
- `transA`: Whether to transpose matrix A (0 or 1)
- `transB`: Whether to transpose matrix B (0 or 1)
- `alpha`: Scalar multiplier for A*B (default 1.0)
- `beta`: Scalar multiplier for C (default 1.0)

#### Pooling Operations

**Max Pooling:**
```mlir
%output = hip.maxpool(%handle, %input)
          {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
          : memref<?x?x?x?xf32, 1> -> memref<?x?x?x?xf32, 1>
```

**Average Pooling:**
```mlir
%output = hip.avgpool(%handle, %input)
          {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
          : memref<?x?x?x?xf32, 1> -> memref<?x?x?x?xf32, 1>
```

Attributes:
- `kernel_shape`: Array of pooling window dimensions [H, W]
- `strides`: Array of stride values [H, W]
- `pads`: Array of padding values [top, left, bottom, right]

## Backend Lowering

The HIP dialect operations lower to LLVM dialect via the conversion pass in `HipToLLVM.cpp`. At runtime, these operations call the corresponding MIOpen or hipBLASLt library functions.

## Build

The dialect is built as a static library `HipDialect.lib` and is included by:
1. Level-1 Pass MLIR compiler (for AOT compilation)
2. hip-opt tool (optional, for testing and debugging)

## Files

- `HipDialect.td` - Dialect definition
- `HipTypes.td` - Type definitions
- `HipOps.td` - Operation definitions
- `HipDialect.h/cpp` - Dialect implementation
- `HipToLLVM.cpp` - Conversion pass to LLVM dialect
- `CMakeLists.txt` - Build configuration
