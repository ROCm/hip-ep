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
- **MIOpen Integration**: Conv, Softmax, RMS Norm, Mul, Sigmoid, Softplus, CausalConvWithState via MIOpen library
- **hipBLASLt MatMul**: High-performance matrix multiplication
- **Custom HIP Kernels**: GQA, RoPE, Cast, Sub, Gather, ReduceSum, Reciprocal, Sqrt, GELU, Range, LinearAttention
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
| Softplus | MIOpen |
| Gelu | Custom HIP kernel |
| Reciprocal | Custom HIP kernel |
| Sqrt | Custom HIP kernel |
| Sub | Custom HIP Kernel |
| Cast | Custom HIP Kernel |
| CastLike | Decomposed → Cast |
| Neg | Custom HIP Kernel |
| Equal | Custom HIP Kernel |
| Not | Custom HIP Kernel |
| Cos | Custom HIP Kernel |
| Sin | Custom HIP Kernel |
| Div | Custom HIP Kernel |
| Mod | Custom HIP Kernel |
| Sign | Custom HIP Kernel |
| Less | Custom HIP Kernel |
| Min | MIOpen |
| ReduceSum | Custom HIP Kernel |
| ReduceMax | Custom HIP Kernel |
| ReduceProd | Custom HIP Kernel |
| CumSum | Custom HIP Kernel |
| Pad | Custom HIP Kernel |
| Tile | Custom HIP Kernel |
| Expand | Custom HIP Kernel |
| GatherND | Custom HIP Kernel |
| Range | Custom HIP kernel |
| NonZero | Runtime stub (not yet GPU-accelerated) |
| Gather | Custom HIP Kernel |
| LayerNormalization | MIOpen |
| SimplifiedLayerNormalization | MIOpen |
| SkipSimplifiedLayerNormalization (com.microsoft) | MIOpen |
| RotaryEmbedding (com.microsoft) | Custom HIP Kernel |
| GroupQueryAttention (com.microsoft) | Custom HIP Kernel |
| MultiHeadAttention (com.microsoft) | Runtime stub (not yet GPU-accelerated) |
| MatMulNBits (com.microsoft) | Custom HIP Kernel |
| QMoE (com.microsoft) | Custom HIP Kernel |
| LinearAttention (com.microsoft) | Custom HIP Kernel |
| CausalConvWithState (com.microsoft) | MIOpen |
| Resize | Custom HIP Kernel |

### Compiler-Optimized Operations

These operations are handled through standard MLIR transformations without requiring GPU backend support:

| Operation | Implementation | Notes |
|-----------|----------------|-------|
| Reshape | tensor.expand_shape / tensor.collapse_shape | Zero-cost metadata operation, no data movement |
| Unsqueeze | tensor.expand_shape | Inserts size-1 axes; shape/stride reinterpretation only |
| Squeeze | tensor.collapse_shape | Removes size-1 axes; shape/stride reinterpretation only |
| Split | tensor.extract_slice | Zero-copy tensor partitioning; creates views without data movement |
| Slice (constant params, positive stride) | tensor.extract_slice (compile-time decompose) | Most common case — constant starts/ends/axes/steps with positive unit/N stride. Lowers to zero-copy `memref.subview` after bufferization. Non-constant indices or negative steps fall through to a native `hip.slice` op with a runtime stub (logs only, no kernel today). |
| Concat | tensor.empty + tensor.insert_slice | Variadic-input concatenation along any axis. Rewritten to one `tensor.empty` plus N `tensor.insert_slice` ops; each insert bufferizes to a `memref.subview` + `memref.copy` against the pooled output buffer. Static, dynamic and mixed shapes are all handled via mixed `OpFoldResult` offsets/sizes; no Concat-specific runtime kernel. |
| ScatterND | `hip.scatter_nd` (runtime stub) | Native DPS op carrying the ONNX `reduction` attribute (`none` / `add` / `mul` / `min` / `max`). Runtime currently logs its parameters only — models exercising ScatterND will produce uninitialised output until the kernel is implemented. |
| Constant | arith.constant or externalized to .constants.bin | ONNX Constant nodes: small values inlined, large tensors externalized |
| ConstantOfShape | arith.constant (compile-time fold) | Folds to a splat constant when the shape input is itself constant; honours optional `value` attribute |
| Identity | SSA value forwarding | Pass-through op; the input value is wired directly to every user (equivalent to a full-range `memref.subview` view, but cheaper — no view op is materialised in the IR) |

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

## Contributing

This repo follows the LLVM project's
[Incremental Development](https://llvm.org/docs/DeveloperPolicy.html#incremental-development)
and [AI Tool Use](https://llvm.org/docs/AIToolPolicy.html) policies. Before
opening a PR, please read [CONTRIBUTING.md](CONTRIBUTING.md) for the local
specifics: PR sizing, AI-assistance disclosure, CODEOWNERS routing, and
commit message trailers.

---

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
