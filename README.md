<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX HIP DNN Execution Provider

An implementation of HIP DNN operations in the MorphiZen framework.

This project demonstrates the integration of HIP (Heterogeneous-compute Interface for Portability) DNN operations within the MorphiZen optimization framework for ONNX Runtime.

---

## Features

- **MLIR Compiler Pipeline**: ONNX dialect → HIP dialect → LLVM IR → native DLL
- **MIOpen Integration**: Conv, Softmax, RMS Norm, Mul, CausalConvWithState via MIOpen library
- **hipBLASLt MatMul**: High-performance matrix multiplication
- **Custom HIP Kernels**: GQA, RoPE, Cast, Sub, Gather, ReduceSum, Reciprocal
- **Memory Pool Optimization**: `hip-pool-allocs` pass packs allocations into a single grow-on-demand buffer
- **Constant Externalization**: Large model weights stored in sidecar `.constants.bin` files
- **Mock Runtime**: GPU-free development and testing with `BUILD_MOCK_RUNTIME=ON`
- **MorphiZen EP Integration**: Plugs into ONNX Runtime as the MorphiZen Execution Provider

---

## Supported Operations

| Operation | Backend |
|-----------|---------|
| Conv | MIOpen |
| MatMul | hipBLASLt |
| Mul | MIOpen |
| Add | MIOpen |
| Sigmoid | MIOpen |
| Reciprocal | Custom HIP kernel |
| Sqrt | MIOpen |
| Sub | Custom HIP Kernel |
| Cast | Custom HIP Kernel |
| ReduceSum | Custom HIP Kernel |
| Gather | Custom HIP Kernel |
| SimplifiedLayerNormalization | MIOpen |
| SkipSimplifiedLayerNormalization (com.microsoft) | MIOpen |
| RotaryEmbedding (com.microsoft) | Custom HIP Kernel |
| GroupQueryAttention (com.microsoft) | Custom HIP Kernel |
| MatMulNBits (com.microsoft) | Custom HIP Kernel |
| QMoE (com.microsoft) | Custom HIP Kernel |
| CausalConvWithState (com.microsoft) | MIOpen |

### Compiler-Optimized Operations

These operations are handled through standard MLIR transformations without requiring GPU backend support:

| Operation | Implementation | Notes |
|-----------|----------------|-------|
| Reshape | tensor.expand_shape / tensor.collapse_shape | Zero-cost metadata operation, no data movement |
| Constant | arith.constant or externalized to .constants.bin | ONNX Constant nodes: small values inlined, large tensors externalized |

---

## Project Design

### Components

- **HIP MLIR Dialect** (`lib/Dialect/`) — custom dialect: `hip.conv`, `hip.matmul`, `hip.gqa`, etc.
- **ONNX → HIP Conversion** (`lib/Conversion/OnnxToHip/`) — pattern-based ONNX op lowering
- **HIP → LLVM Lowering** (`lib/Conversion/HipToLLVM/`) — lowers HIP ops to runtime C API calls
- **Compiler Driver** (`lib/Compiler/`) — orchestrates the full compilation pipeline
- **Runtime** (`lib/Runtime/`) — real (GPU) and mock (CPU) backends
- **Custom HIP Kernels** (`3rd-party/custom_kernels/`) — handwritten `.hip` kernels for GQA, RoPE, etc.
- **Compiler DLL** (`dll/`) — `hip-compiler.dll` exposing the C API
- **Schemas** (`schemas/`) — FlatBuffers definitions for model metadata and compilation options
- **Tools** — `hip-mlir-opt`, `hip-compiler`, `hip-test-dll`, `hip-inspect-dll`, `hip-onnx-runner`
- **Backend Integration** (`backend-mlir-compiler/`) — bridges to MorphiZen Execution Provider

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        COMPILE-TIME                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ONNX Model (.onnx)                                         │
│      ↓                                                       │
│  onnx-to-hip-pipeline                                        │
│      hip-add-context-arg                                     │
│      convert-onnx-to-hip ── constants → .constants.bin       │
│      one-shot-bufferize (tensor → memref)                    │
│      buffer-deallocation                                     │
│      hip-optimize-memrefs (liveness-based buffer reuse)      │
│      hip-pool-allocs (single grow-on-demand pool)            │
│      hip-lower-allocs (memref.alloc → hip.alloc/free)        │
│      hip-resolve-extern-constants                            │
│      ↓                                                       │
│  hip-to-llvm-pipeline                                        │
│      convert-hip-to-llvm (HIP ops → runtime C API calls)    │
│      generate-interface (inference_init/compute/cleanup)     │
│      ↓                                                       │
│  MLIR → LLVM IR → merge runtime.bc → optimize → link        │
│      ↓                          ↑                            │
│  model.dll              amdhip64 / MIOpen / hipblaslt /      │
│  + constants.bin         hip_custom_kernels.lib               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                          RUNTIME                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Load model.dll + constants.bin                              │
│      ↓                                                       │
│  inference_init   → GPU handles, upload constants, alloc pool│
│  inference_compute → execute ops (MIOpen/hipBLASLt/kernels)  │
│  inference_cleanup → free GPU resources                      │
│                                                              │
│  Dependencies: amdhip64.dll, MIOpen.dll, hipblaslt.dll       │
│  No LLVM/MLIR needed at inference time                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Building

For prerequisites, environment setup, and step-by-step build instructions, see
[docs/quick_start.md](docs/quick_start.md).

---

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
