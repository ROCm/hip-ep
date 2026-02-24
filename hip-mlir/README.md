# HIP MLIR Dialect

An MLIR dialect for HIP (Heterogeneous-compute Interface for Portability)
runtime operations. It provides a set of ops for device memory management and
compute kernels dispatched through HIP libraries (hipBLASLt, MIOpen) or custom
HIP kernels.

## Dialect overview

| Namespace | `hip` |
|-----------|-------|
| C++ namespace | `::mlir::hip` |
| Custom type | `!hip.handle` — opaque runtime handle |

### Operations

**Lifecycle & memory**

| Op | Description |
|----|-------------|
| `hip.create_handle` | Create a HIP runtime handle |
| `hip.destroy_handle` | Destroy a HIP runtime handle |
| `hip.alloc` | Allocate device memory (wraps `hipMalloc`) |
| `hip.free` | Free device memory (wraps `hipFree`) |

**hipBLASLt**

| Op | Description |
|----|-------------|
| `hip.hipblaslt.matmul` | Matrix multiply C = A @ B |
| `hip.hipblaslt.graph` | Structural region for hipBLASLt dispatch |

**MIOpen**

| Op | Description |
|----|-------------|
| `hip.miopen.add` | Element-wise add |
| `hip.miopen.mul` | Element-wise multiply |
| `hip.miopen.softmax` | Softmax (row-wise, last dim) |
| `hip.miopen.graph` | Structural region for MIOpen dispatch |

**Custom HIP kernels**

| Op | Description |
|----|-------------|
| `hip.transpose` | N-D transpose swapping two dimensions |

All compute ops use **destination-passing style** (DPS) and work in both
tensor mode (pre-bufferization) and memref mode (post-bufferization).

## File layout

```
hip-mlir/
├── CMakeLists.txt        # Build: TableGen + HipDialect static library
├── HipDialect.td         # Dialect definition
├── HipTypes.td           # Type definitions (!hip.handle)
├── HipOps.td             # Operation definitions
├── HipDialect.h          # Dialect C++ header
├── HipDialect.cpp        # Dialect + op implementation (parse/print/verify)
├── HipBufferize.h        # DPS bufferization interface
└── README.md
```

## Building

Requires an LLVM/MLIR build tree.

```bash
cmake -B build -G Ninja \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build
```

This produces a `HipDialect` static library that downstream tools can link
against.

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
