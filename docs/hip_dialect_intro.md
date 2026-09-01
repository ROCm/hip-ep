<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP Dialect Reference

The `hip` dialect provides MLIR operations for GPU inference on AMD ROCm.
All ops live under the `hip` namespace. Ops are grouped by the backend library
that implements them at runtime.

All compute ops use **Destination-Passing Style (DPS)**: arguments are split
into `ins(...)` (read-only inputs) and `outs(...)` (output destinations
provided by the caller). Handle and scalar parameters are leading positional
arguments.

All compute ops support **dual-mode operation**:
- **Memref mode** (default for hand-written tests): operands are `memref<...>`, no results, writes in-place.
- **Tensor mode** (used by `--convert-onnx-to-hip` and bufferization): operands are `tensor<...>`, returns results. Standard `--one-shot-bufferize` can then lower tensor mode to memref mode automatically.
Every compute DPS operation declares an explicit same-shape, broadcast,
reduction, semantic, payload, or outs-authoritative contract. A classification
does not by itself change the runtime ABI; behavioral changes remain in the
lowest layer that owns the corresponding verifier, reifier, or destination.


---

## Types

| Type | Description |
|---|---|
| `!hip.context` | Opaque execution context. Passed as function arg 0. Lowered to `!llvm.ptr`. |

---

## Context & Memory

The `!hip.context` type is passed as the first function argument. The `hip-add-context-arg` pass injects it automatically.

| Op | Signature | Runtime |
|---|---|---|
| `hip.alloc` | `(ctx, dyn_sizes...) -> memref<...>` | `hip_device_malloc(size)` + memref descriptor |
| `hip.free` | `(ctx, memref) -> ()` | `hip_device_free(ptr)` |

---

## hipBLASLt Ops

Matrix multiplication backed by the hipBLASLt library (`hipblasLtMatmul`).

| Op | DPS Syntax | Runtime | Status |
|---|---|---|---|
| `hip.hipblaslt.matmul` | `(%ctx) ins(%A, %B : ...) outs(%C : ...)` | `hip_hipblaslt_matmul(handle, A, B, C, rankA, rankB, batch, M, K, N)` | Full impl |

Rank-generic: batch is determined from A's rank (3D -> batched, 2D -> single).
If B has fewer dims than A (e.g. `X[B,S,D] @ W[D,D]`), B is broadcast across
batches (`stride_B = 0`). Supports strided batched GEMM via hipBLASLt.

---


### Normalization

The norm ops are backed by custom HIP kernels.

| Op | DPS Syntax | Backend |
|---|---|---|
| `hip.rms_norm` | `(%ctx) ins(%input, %weight : ...) outs(%output : ...)` | `rms_norm_kernel.hip` |
| `hip.skip_rms_norm` | `(%ctx) ins(%x, %skip, %weight : ...) outs(%output, %residual : ...)` | `skip_rms_norm_kernel.hip` |

Rank-generic: for 3D input `[B,S,D]`, the lowering flattens `N = B*S, D = D` and
passes them to the runtime: `wrap_rms_norm(state, input, weight, output, ...)`.

`skip_rms_norm` fuses Add + RMSNorm into a single kernel:
`residual = x + skip; output = RMSNorm(residual) * weight`.

### Rotary Positional Embeddings

| Op | DPS Syntax | Runtime entry |
|---|---|---|
| `hip.miopen.rope` | `(%ctx, %start_pos) ins(%cos, %sin : ...) outs(%q, %k : ...)` | `wrap_rotary_embedding` |

### Element-wise Tensor Ops

| Op | DPS Syntax | Runtime entry |
|---|---|---|
| `hip.add` | `(%ctx) ins(%lhs, %rhs : ...) outs(%output : ...)` | `wrap_elementwise` |
| `hip.mul` | `(%ctx) ins(%lhs, %rhs : ...) outs(%output : ...)` | `wrap_elementwise` |
| `hip.min` | `(%ctx) ins(%lhs, %rhs : ...) outs(%output : ...)` | `wrap_elementwise` |
| `hip.max` | `(%ctx) ins(%lhs, %rhs : ...) outs(%output : ...)` | `wrap_elementwise` |
| `hip.sub` | `(%ctx) ins(%lhs, %rhs : ...) outs(%output : ...)` | `wrap_elementwise_sub` |

The lowering passes full 4D shapes for both operands and the result, so the
runtime broadcasts per axis. `hip.add(memref<1x128x32xf16>, memref<32xf16>)`
passes `lhs=[1,1,128,32]`, `rhs=[1,1,1,32]`, `out=[1,1,128,32]`; a dim of 1
broadcasts against the corresponding larger dim. `hip.sub` requires equal
shapes -- broadcasting is resolved before conversion.

### Softmax

| Op | DPS Syntax | Runtime entry |
|---|---|---|
| `hip.miopen.softmax` | `(%ctx) ins(%input : ...) outs(%output : ...)` | `hip_miopen_softmax` |

Row-wise softmax over the last dimension. Rank-generic: for 3D `[B,S,D]`, the
lowering flattens `rows = B*S, cols = D` and passes `elem_size_bytes` (2 for
fp16, 4 for fp32) so the runtime dispatches the matching dtype.

---

## Custom HIP Kernel Ops

Ops with no hipBLASLt equivalent. Implemented as pure C++ kernels.

| Op | DPS Syntax | Purpose | Status |
|---|---|---|---|
| `hip.transpose` | `(%ctx) ins(%input : ...) outs(%output : ...) {perm = [...]}` | N-D transpose with arbitrary permutation | Full impl |
| `hip.gather` | `(%ctx) ins(%indices, %table : ...) outs(%output : ...)` | Embedding table lookup | Stub |
| `hip.silu` | `(%ctx) ins(%input : ...) outs(%output : ...)` | SiLU activation: `x * sigmoid(x)` | Stub |
| `hip.gqa` | `(%ctx, %layer, %start_pos, %seq_len) ins(%q, %k, %v : ...) outs(%kv_cache, %output : ...)` | Grouped query attention with KV cache | Stub |

---

## Region Ops (Structural)

Grouping ops that mark which backend library handles a block of compute ops.
No runtime -- inlined and erased during lowering to LLVM.

| Op | Syntax | Purpose |
|---|---|---|
| `hip.miopen.graph` | `hip.miopen.graph { ... }` | Groups custom-kernel-dispatched ops (name predates the MIOpen removal) |
| `hip.hipblaslt.graph` | `hip.hipblaslt.graph { ... }` | Groups hipBLASLt-dispatched ops |

---

## Example: 3D Matmul with Weight Broadcast (tensor DPS)

```mlir
func.func @two_matmuls(
    %ctx: !hip.context,
    %A:  tensor<?x?x?xf32>,   // [B, S, K]
    %B0: tensor<?x?xf32>,     // [K, N]  (2D, broadcast across batch)
    %B1: tensor<?x?xf32>,     // [N, P]
    %C:  tensor<?x?x?xf32>)   // [B, S, P]  (output init)
    -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %B = tensor.dim %A, %c0 : tensor<?x?x?xf32>
  %S = tensor.dim %A, %c1 : tensor<?x?x?xf32>
  %N = tensor.dim %B0, %c1 : tensor<?x?xf32>
  %tmp_init = tensor.empty(%B, %S, %N) : tensor<?x?x?xf32>

  %tmp = hip.hipblaslt.matmul(%ctx)
      ins(%A, %B0 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%tmp_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  %C_out = hip.hipblaslt.matmul(%ctx)
      ins(%tmp, %B1 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  return %C_out : tensor<?x?x?xf32>
}
```

`%A` is 3D `[B,S,K]`, `%B0` is 2D `[K,N]` -- the matmul broadcasts B0 across
all batches (stride_B=0). `%tmp_init` is created via `tensor.empty` and
automatically becomes a device allocation after bufferization.

---

## Lowering Pipeline

The `hip-compiler` accepts pre-bufferized `.hip.mlir` input (memref format).
Bufferization is handled upstream. The `hip-compiler` pipeline is:

1. **`--convert-hip-to-llvm`** — lowers all HIP ops to LLVM dialect calls.
2. Standard LLVM lowering passes: `--finalize-memref-to-llvm`, `--convert-arith-to-llvm`,
   `--convert-func-to-llvm`, `--reconcile-unrealized-casts`.

For reference, the upstream bufferization passes (not part of `hip-compiler`) are:
- **`--one-shot-bufferize`** — converts tensor DPS ops to memref in-place ops.
- **`--hip-optimize-memrefs`** — liveness-based buffer reuse.
- **`--hip-lower-allocs`** — converts `memref.alloc` to `hip.alloc` + `hip.free`.

Details of `--convert-hip-to-llvm`:

- **Compute ops** lower to `llvm.call @hip_<op_name>(...)`. Memref arguments are
  converted to raw pointers via `MemRefDescriptor(...).alignedPtr()` (using
  `alignedPtr`, not `allocatedPtr`, so that `memref.view` offsets into a memory
  pool are correctly preserved).
  Shape metadata is extracted rank-generically from memref descriptors.
- **Binary ops** (`hip.add`, `hip.mul`, `hip.min`, `hip.max`) pass full 4D shapes
  for both operands and the result, so the runtime broadcasts per axis.
- **`memref.alloc` / `memref.dealloc`** (if any remain) are converted
  to device allocation calls (`hip_device_malloc` / `hip_device_free`).
- **Region ops** are inlined: body ops moved to parent block, region op erased.
- **`!hip.context`** is converted to `!llvm.ptr`.

The `hip-compiler` tool runs this pipeline automatically, then translates
the resulting LLVM dialect to LLVM IR, generates a native `.obj`, and links it
with `runtime.bc` (pre-compiled bitcode) and external libraries to produce a `.dll`.

For manual debugging, `hip-mlir-opt` can run the pass pipeline in isolation:
`--one-shot-bufferize="bufferize-function-boundaries"`,
`--hip-optimize-memrefs`, `--hip-lower-allocs`,
`--convert-hip-to-llvm`, `--finalize-memref-to-llvm`, `--convert-arith-to-llvm`,
`--convert-func-to-llvm`, `--reconcile-unrealized-casts`.

---

## File Structure

```
include/hip/Dialect/IR/
  HipDialect.td            Dialect definition (namespace "hip")
  HipTypes.td              Type definitions (!hip.context)
  HipOps.td                All op definitions (DPS ins/outs format)
  HipDialect.h             Dialect C++ header
  HipBufferize.h           Bufferization interface models

include/hip/Dialect/Transforms/
  Passes.td                Pass definitions via TableGen
  Passes.h                 Pass declarations (auto-generated)

lib/Dialect/IR/
  HipDialect.cpp           Dialect registration + DPS interface implementations
lib/Dialect/Transforms/
  OptimizeMemRefs.cpp      Buffer reuse pass (--hip-optimize-memrefs)
  PoolAllocs.cpp           Memory pooling pass (--hip-pool-allocs)
  LowerAllocs.cpp          memref.alloc -> hip.alloc pass (--hip-lower-allocs)
lib/Conversion/HipToLLVM/
  HipToLLVM.cpp            Lowering pass (hip -> llvm.call)
lib/Conversion/OnnxToHip/
  OnnxToHip.cpp            [optional] ONNX-to-HIP conversion pass (requires onnx-mlir)

tools/hip-mlir-opt/
  hip-mlir-opt.cpp         MLIR pass pipeline tool (for debugging)
tools/hip-compiler/
  hip-compiler.cpp         One-stop MLIR-to-DLL compiler

lib/Runtime/                   (compiled to runtime.bc bitcode by CMake)
  hipdnn_ep_runtime.h     Runtime C API header
  hipdnn_ep_runtime_state.cpp   Runtime state management
  hipdnn_ep_runtime_tensor.cpp  Tensor descriptor helpers
  real/                   Real GPU runtime (HIP / hipBLASLt / custom kernels)
    hip.cpp               HIP device management
    memory.cpp            Device memory allocation
    matmul.cpp            hipBLASLt matmul
    elementwise.cpp       add / mul / min / max / sub
    simplified_layer_norm.cpp     RMS norm
    skip_simplified_layer_norm.cpp  skip + RMS norm
    rotary_embedding.cpp  RoPE
    activation.cpp        SiLU activation
    cast.cpp              Type casting
    gather.cpp            Embedding gather
    gqa.cpp               Grouped query attention
    hipblas.cpp           hipBLAS utilities
    reduce_sum.cpp        Reduction sum
  mock/                   Mock CPU runtime (no GPU required)
    mock_gpu.cpp          CPU fallback implementations
    memory.cpp            Host memory allocation

examples/
  gemm.hip.mlir           Two chained matmuls (hipBLASLt)
  add.hip.mlir            Two chained adds
  mul.hip.mlir            Two chained muls
  rms_norm.hip.mlir       Two chained RMS norms
  softmax.hip.mlir        Two chained softmaxes
  attention.hip.mlir      Attention (memref pool + embedded weights)
  main_gemm.cpp           C++ driver for gemm.hip.mlir
  main_add.cpp            C++ driver for add.hip.mlir
  main_mul.cpp            C++ driver for mul.hip.mlir
  main_rms_norm.cpp       C++ driver for rms_norm.hip.mlir
  main_softmax.cpp        C++ driver for softmax.hip.mlir
  main_attention.cpp      C++ driver for attention.hip.mlir
  main_e2e.cpp            C++ driver for end-to-end test

scripts/
  env.bat                          Shared environment config (edit paths here)
  run_full_pipeline_hipblaslt.bat        Matmul: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_add.bat       Add: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_mul.bat       Mul: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_rms_norm.bat  RMS Norm: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_softmax.bat   Softmax: hip-compiler + cl.exe driver
  run_full_pipeline_attention.bat        Attention: hip-compiler + ORT reference
```
