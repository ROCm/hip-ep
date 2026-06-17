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
- **MIOpen Integration**: Conv, ConvTranspose, RMS Norm, Sigmoid, Tanh, Softplus; fp elementwise Mul/Add/Min/Max; CausalConvWithState uses MIOpen only on the general fallback path
- **hipBLASLt MatMul**: High-performance matrix multiplication
- **Custom HIP Kernels**: GQA, RoPE, Cast, Sub, Gather, ReduceSum, Reciprocal, Sqrt, GELU, Range, LinearAttention, Softmax, CausalConvWithState(fast path)
- **Memory Pool Optimization**: `hip-pool-allocs` pass packs allocations into a single grow-on-demand buffer
- **Constant Externalization**: Large model weights stored in sidecar `.constants.bin` files
- **Mock Runtime**: GPU-free development and testing with `BUILD_MOCK_RUNTIME=ON`
- **MorphiZen EP Integration**: Plugs into ONNX Runtime as the MorphiZen Execution Provider

---

## Supported Operations

| Operation | Backend |
|-----------|---------|
| Conv | MIOpen |
| ConvTranspose | MIOpen |
| MatMul | hipBLASLt |
| Gemm | hipBLASLt |
| Transpose | Custom HIP Kernel |
| Mul | MIOpen |
| Add | MIOpen |
| Softmax | Custom HIP Kernel |
| Sigmoid | MIOpen |
| Tanh | MIOpen |
| Softplus | MIOpen |
| Gelu | Custom HIP kernel |
| Reciprocal | Custom HIP kernel |
| Sqrt | Custom HIP kernel |
| Exp | Custom HIP Kernel |
| Pow | Decomposed → Mul / Sqrt / Reciprocal for constant scalar exponents |
| Sub | Custom HIP Kernel |
| Cast | Custom HIP Kernel |
| CastLike | Decomposed → Cast |
| Neg | Custom HIP Kernel |
| Equal | Custom HIP Kernel |
| Not | Custom HIP Kernel |
| And | Custom HIP Kernel |
| Cos | Custom HIP Kernel |
| Sin | Custom HIP Kernel |
| Div | Custom HIP Kernel |
| Mod | Custom HIP Kernel |
| Sign | Custom HIP Kernel |
| Where | Custom HIP Kernel |
| Less | Custom HIP Kernel |
| GreaterOrEqual | Decomposed (Not(Less(A, B))) |
| LessOrEqual | Decomposed (Not(Less(B, A))) |
| Min | MIOpen |
| Max | MIOpen |
| ReduceSum | Custom HIP Kernel |
| ReduceMax | Custom HIP Kernel |
| ReduceProd | Custom HIP Kernel |
| ReduceMean | Custom HIP Kernel |
| CumSum | Custom HIP Kernel |
| Pad | Custom HIP Kernel |
| Tile | Custom HIP Kernel |
| Expand | Custom HIP Kernel |
| GatherND | Custom HIP Kernel |
| ScatterND | Custom HIP Kernel (reductions: none / add / mul / min / max) |
| Range | Custom HIP kernel |
| Size | Custom HIP Kernel (folds to a constant for static shapes) |
| NonZero | Custom HIP Kernel |
| Gather | Custom HIP Kernel |
| LayerNormalization | Custom HIP Kernel |
| SkipLayerNormalization (com.microsoft) | Decomposed → Add (MIOpen) + LayerNormalization (Custom HIP Kernel) |
| SimplifiedLayerNormalization | MIOpen |
| SkipSimplifiedLayerNormalization (com.microsoft) | MIOpen |
| LpNormalization | Decomposed → Mul / ReduceSum / Sqrt / Div |
| RotaryEmbedding (com.microsoft) | Custom HIP Kernel |
| GroupQueryAttention (com.microsoft) | Custom HIP Kernel |
| MultiHeadAttention (com.microsoft) | hipBLASLt + Custom HIP Kernels (encoder–decoder attention also lowers to GroupQueryAttention) |
| Attention (com.microsoft) | Custom HIP Kernel (fused QKV split, lowered to GroupQueryAttention) |
| MatMulNBits (com.microsoft) | Custom HIP Kernel |
| QMoE (com.microsoft) | Custom HIP Kernel |
| GatherBlockQuantized (com.microsoft) | Custom HIP Kernel |
| LinearAttention (com.microsoft) | Custom HIP Kernel |
| CausalConvWithState (com.microsoft) | Custom HIP Kernel (decode / prefill fast paths) + MIOpen (general fallback) |
| Relu | Decomposed → Max (MIOpen) |
| LeakyRelu | Custom HIP Kernel |
| Clip | Decomposed → Max + Min (MIOpen) |
| MaxPool | Custom HIP Kernel |
| AveragePool | Custom HIP Kernel |
| LpPool | Custom HIP Kernel |
| Resize | Custom HIP Kernel |
| GlobalAveragePool | Custom HIP Kernel |
| GlobalMaxPool | Custom HIP Kernel |
| GlobalLpPool | Custom HIP Kernel |

### Compiler-Optimized Operations

These operations are handled through standard MLIR transformations without requiring GPU backend support:

| Operation | Implementation | Notes |
|-----------|----------------|-------|
| Reshape | tensor.expand_shape / tensor.collapse_shape | Zero-cost metadata operation, no data movement |
| Unsqueeze | tensor.expand_shape | Inserts size-1 axes; shape/stride reinterpretation only |
| Squeeze | tensor.collapse_shape | Removes size-1 axes; shape/stride reinterpretation only |
| Split | tensor.extract_slice | Zero-copy tensor partitioning; creates views without data movement |
| Slice (constant params, positive stride) | tensor.extract_slice (compile-time decompose) | Most common case — constant starts/ends/axes/steps with positive unit/N stride. Lowers to zero-copy `memref.subview` after bufferization. Non-constant indices or negative steps fall through to a native `hip.slice` op backed by a runtime kernel (`hip_slice`); index tensors are assumed INT64. |
| Concat | tensor.empty + tensor.insert_slice | Variadic-input concatenation along any axis. Rewritten to one `tensor.empty` plus N `tensor.insert_slice` ops; each insert bufferizes to a `memref.subview` + `memref.copy` against the pooled output buffer. Static, dynamic and mixed shapes are all handled via mixed `OpFoldResult` offsets/sizes; no Concat-specific runtime kernel. |
| Shape | tensor.dim + tensor.from_elements | Fully static shapes fold to a compile-time constant (placed in the constants blob). Dynamic dims lower to `tensor.dim` queries assembled via `tensor.from_elements`; honours optional `start`/`end` sub-range attributes. |
| Constant | arith.constant or externalized to .constants.bin | ONNX Constant nodes: small values inlined, large tensors externalized |
| ConstantOfShape | arith.constant (compile-time fold) | Folds to a splat constant when the shape input is itself constant; honours optional `value` attribute |
| Identity | SSA value forwarding | Pass-through op; the input value is wired directly to every user (equivalent to a full-range `memref.subview` view, but cheaper — no view op is materialised in the IR) |
| Flatten | tensor.collapse_shape (+ tensor.expand_shape for axis = 0 / axis = r) | Reshapes rank-r input to rank-2; pure metadata reinterpretation. Dynamic dims supported. |

---

## Project Design

### Components

- **HIP MLIR Dialect** (`lib/Dialect/`) — custom dialect: `hip.conv`, `hip.matmul`, `hip.gqa`, etc.
- **ONNX → HIP Conversion** (`lib/Conversion/OnnxToHip/`) — pattern-based ONNX op lowering
- **HIP → LLVM Lowering** (`lib/Conversion/HipToLLVM/`) — lowers HIP ops to runtime C API calls
- **Compiler Driver** (`lib/Compiler/`) — orchestrates the full compilation pipeline
- **Runtime** (`lib/Runtime/`) — real (GPU) and mock (CPU) backends
- **Custom HIP Kernels** (`lib/Runtime/Kernels/`) — handwritten `.hip` kernels for GQA, RoPE, etc.
- **Compiler DLL** (`dll/`) — `hip-compiler.dll` exposing the C API
- **Schemas** (`schemas/`) — FlatBuffers definitions for model metadata and compilation options
- **Tools** — `hip-mlir-opt`, `hip-compiler`, `hip-onnx-runner`, `hip-inspect`, `hip-test`
- **Backend Integration** (`backend-mlir-compiler/`) — bridges to MorphiZen Execution Provider

### Architecture

```
=== COMPILE-TIME ===

ONNX Model (.onnx)
  │
  ▼
onnx-to-hip-pipeline
    simplify-onnx                  (CastLike → Cast, drop dead type-donor args)
    hip-add-context-arg
    onnx-loop-outline              (+ hip-infer-loop-body-shapes)
    convert-onnx-to-hip            constants → .constants.bin
    hip-infer-shapes               (refine dynamic result dims)
    hip-resolve-tensor-dims
    one-shot-bufferize             (tensor → memref)
    hip-loop-body-to-out-params    (outlined onnx.Loop bodies → out-param ABI)
    buffer-deallocation            (ownership-based)
    hip-use-output-allocator       (default ABI; outputs allocated in-graph via
                                    EP callback. classic builds instead run
                                    buffer-results-to-out-params before dealloc)
    hip-fix-loop-accumulator-offset (onnx.Loop growing-accumulator offsets)
    convert-linalg-to-loops        (lower any residual linalg.* to scf + stores)
    hip-optimize-memrefs           (liveness-based buffer reuse)
    hip-promote-strided-operands
    hip-materialize-host-scalars   (host-mapped scratch for shape scalars)
    hip-hoist-alloc-size-arith
    hip-pool-allocs                (single grow-on-demand GPU pool)
    convert-bufferization-to-memref (lower residual bufferization.* ops)
    hip-lower-allocs               (memref.alloc → hip.alloc/free)
    hip-resolve-extern-constants
  │  (canonicalize / CSE cleanup runs interleaved between stages)
  ▼
hip-to-llvm-pipeline
    hip-relax-multi-dyn-expand-shape
    expand-strided-metadata        (+ lower-affine)
    scf-to-control-flow            (+ reconcile-unrealized-casts)
    convert-hip-to-llvm            (HIP ops → runtime C API calls)
    generate-interface             (inference_init / compute / cleanup)
  │
  ▼
MLIR → LLVM IR → optimize
  │
  ▼
artifact_format?
  ├─ LLVM_IR (default): emit model.bc  (runtime.bc kept separate;
  │                     OS-portable, no linker)
  └─ NATIVE  (opt-in) : merge runtime.bc → link model.dll/.so
  │
  ▼
model.bc | model.dll   +   constants.bin


=== RUNTIME ===

Load artifact + constants.bin   (loader = artifact_format)
  ├─ LLVM_IR: LlvmIrJit JIT-links model.bc + the runtime.bc
  │           embedded in the EP, in-process (LLVM ORC)
  └─ NATIVE : LoadLibrary / dlopen model.dll/.so
  │
  ▼
inference_init    → GPU handles, upload constants, alloc pool
inference_compute → execute ops (MIOpen / hipBLASLt / kernels)
inference_cleanup → free GPU resources

Dependencies: amdhip64, MIOpen, hipblaslt, custom_kernels
No OS toolchain at inference time (LLVM_IR JITs in-process;
the LLVM ORC engine ships inside the EP)
```

> **Output-allocator ABI.** By default the EP compiles every model to the
> 2-arg `inference_compute(state, inputs)` form: graph outputs are allocated
> *in-graph* at runtime via the EP's output-allocator callback (`hip.alloc_output`),
> selected by the `hip-use-output-allocator` pass stamping a module attribute.
> The classic 3-arg out-param ABI remains for the `hip-compiler` CLI / LIT tests.

> **Artifact format.** Selected by the single compile option `artifact_format`
> (`LLVM_IR` default, `NATIVE` opt-in) and recorded in the EPContext metadata,
> which the EP reads at load time. The default ships **OS-portable LLVM IR** (`.bc`,
> JIT-loaded in-process); native `.dll`/`.so` stays a first-class opt-in for
> benchmarking/parity — see
> [docs/native-vs-ir-comparison.md](docs/native-vs-ir-comparison.md).

---

## Building

For prerequisites, environment setup, and step-by-step build instructions, see
[docs/quick_start.md](docs/quick_start.md).

---

## Plugin extension API

`hip-compiler` ships a public plugin loader (`HIP_EP_PLUGINS` env var,
semicolon-separated paths) that lets a down-stream shared library contribute
MLIR passes, runtime LLVM bitcode, and external libraries without forking
this repo. The design and downstream-usage guide lives in
[docs/design/plugin-interface.md](docs/design/plugin-interface.md); the
practical authoring guide is
[docs/plugin_authoring.md](docs/plugin_authoring.md). A working in-tree
example sits under `test/plugin/sample_plugin/`.

The ABI is not yet frozen; treat `HIP_EP_PLUGIN_API_VERSION` as provisional.

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
