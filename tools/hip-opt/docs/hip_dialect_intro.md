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

---

## Types

| Type | Description |
|---|---|
| `!hip.handle` | Opaque runtime handle for managing HIP library state |

---

## Runtime Lifecycle & Memory

These ops manage the HIP runtime and device memory.

| Op | Signature | Runtime |
|---|---|---|
| `hip.create_handle` | `() -> !hip.handle` | `hipCreateHandle()` |
| `hip.destroy_handle` | `(!hip.handle) -> ()` | `hipDestroyHandle(handle)` |
| `hip.alloc` | `(handle, dyn_sizes...) -> memref<...>` | `hip_device_malloc(size)` + memref descriptor |
| `hip.free` | `(handle, memref) -> ()` | `hip_device_free(ptr)` |

---

## hipBLASLt Ops

Matrix multiplication backed by the hipBLASLt library (`hipblasLtMatmul`).

| Op | DPS Syntax | Runtime | Status |
|---|---|---|---|
| `hip.hipblaslt.matmul` | `(%handle) ins(%A, %B : ...) outs(%C : ...)` | `hip_hipblaslt_matmul(handle, A, B, C, rankA, rankB, batch, M, K, N)` | Full impl |

Rank-generic: batch is determined from A's rank (3D -> batched, 2D -> single).
If B has fewer dims than A (e.g. `X[B,S,D] @ W[D,D]`), B is broadcast across
batches (`stride_B = 0`). Supports strided batched GEMM via hipBLASLt.

---

## MIOpen Ops

Ops backed by the MIOpen library. Each maps to a specific MIOpen C API call.

### Normalization

| Op | DPS Syntax | MIOpen API | Status |
|---|---|---|---|
| `hip.miopen.rms_norm` | `(%handle) ins(%input, %weight : ...) outs(%output : ...)` | `miopenT5LayerNormForward` | Full impl |
| `hip.miopen.skip_rms_norm` | `(%handle) ins(%x, %skip, %weight : ...) outs(%output, %residual : ...)` | `miopenAddLayerNormForward` (T5 mode) | Stub |

Rank-generic: for 3D input `[B,S,D]`, the lowering flattens `N = B*S, D = D` and
passes them to the runtime: `hip_miopen_rms_norm(handle, input, weight, output, N, D)`.

`skip_rms_norm` fuses Add + RMSNorm into a single kernel:
`residual = x + skip; output = RMSNorm(residual) * weight`.
Uses `MIOPEN_ELEMENTWISE_AFFINE_T5` normalization mode.

### Rotary Positional Embeddings

| Op | DPS Syntax | MIOpen API |
|---|---|---|
| `hip.miopen.rope` | `(%handle, %start_pos) ins(%cos, %sin : ...) outs(%q, %k : ...)` | `miopenRotaryPositionalEmbeddings` (experimental) |

### Element-wise Tensor Ops

| Op | DPS Syntax | MIOpen API | Status |
|---|---|---|---|
| `hip.miopen.add` | `(%handle) ins(%A, %B : ...) outs(%C : ...)` | `miopenOpTensor(miopenTensorOpAdd)` | Full impl |
| `hip.miopen.mul` | `(%handle) ins(%A, %B : ...) outs(%C : ...)` | `miopenOpTensor(miopenTensorOpMul)` | Full impl |

The binary op lowering computes `numElements` (product of all memref dimensions) and
passes it to the runtime: `hip_miopen_{add,mul}(handle, A, B, C, numElements)`.

### Softmax

| Op | DPS Syntax | MIOpen API | Status |
|---|---|---|---|
| `hip.miopen.softmax` | `(%handle) ins(%input : ...) outs(%output : ...)` | `miopenSoftmaxForward_V2` | Full impl |

Row-wise softmax over the last dimension. Rank-generic: for 3D `[B,S,S]`, the
lowering flattens `rows = B*S, cols = S`. The runtime uses a 4D descriptor
`[rows, cols, 1, 1]` with `MIOPEN_SOFTMAX_MODE_CHANNEL` to normalize over cols.

---

## Custom HIP Kernel Ops

Ops with no MIOpen or hipBLASLt equivalent. Implemented as pure C++ kernels.

| Op | DPS Syntax | Purpose | Status |
|---|---|---|---|
| `hip.transpose` | `(%handle, %dim0, %dim1) ins(%input : ...) outs(%output : ...)` | N-D transpose swapping two dims | Full impl |
| `hip.gather` | `(%handle) ins(%indices, %table : ...) outs(%output : ...)` | Embedding table lookup | Stub |
| `hip.silu` | `(%handle) ins(%input : ...) outs(%output : ...)` | SiLU activation: `x * sigmoid(x)` | Stub |
| `hip.gqa` | `(%handle, %layer, %start_pos, %seq_len) ins(%q, %k, %v : ...) outs(%kv_cache, %output : ...)` | Grouped query attention with KV cache | Stub |

---

## Region Ops (Structural)

Grouping ops that mark which backend library handles a block of compute ops.
No runtime -- inlined and erased during lowering to LLVM.

| Op | Syntax | Purpose |
|---|---|---|
| `hip.miopen.graph` | `hip.miopen.graph { ... }` | Groups MIOpen-dispatched ops |
| `hip.hipblaslt.graph` | `hip.hipblaslt.graph { ... }` | Groups hipBLASLt-dispatched ops |

---

## Example: 3D Matmul with Weight Broadcast (tensor DPS)

```mlir
func.func @two_matmuls(
    %A:  tensor<?x?x?xf32>,   // [B, S, K]
    %B0: tensor<?x?xf32>,     // [K, N]  (2D, broadcast across batch)
    %B1: tensor<?x?xf32>,     // [N, P]
    %C:  tensor<?x?x?xf32>)   // [B, S, P]  (output init)
    -> tensor<?x?x?xf32> {
  %handle = hip.create_handle() : !hip.handle

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %B = tensor.dim %A, %c0 : tensor<?x?x?xf32>
  %S = tensor.dim %A, %c1 : tensor<?x?x?xf32>
  %N = tensor.dim %B0, %c1 : tensor<?x?xf32>
  %tmp_init = tensor.empty(%B, %S, %N) : tensor<?x?x?xf32>

  %tmp = hip.hipblaslt.matmul(%handle)
      ins(%A, %B0 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%tmp_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  %C_out = hip.hipblaslt.matmul(%handle)
      ins(%tmp, %B1 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  hip.destroy_handle(%handle) : !hip.handle
  return %C_out : tensor<?x?x?xf32>
}
```

`%A` is 3D `[B,S,K]`, `%B0` is 2D `[K,N]` -- the matmul broadcasts B0 across
all batches (stride_B=0). `%tmp_init` is created via `tensor.empty` and
automatically becomes a device allocation after bufferization.

---

## Lowering Pipeline

The canonical HIP dialect uses tensor types. The lowering pipeline is:

1. **`--one-shot-bufferize`** — converts tensor DPS ops to memref in-place ops.
   `tensor.empty` becomes `memref.alloc`; function tensor args become memref args.
2. **`--convert-hip-to-llvm`** — lowers all HIP ops and `memref.alloc`/`dealloc`
   to LLVM dialect calls. `memref.alloc` is converted to `hip_device_malloc`
   (device memory), not standard `malloc`.
3. Standard LLVM lowering passes: `--finalize-memref-to-llvm`, `--convert-arith-to-llvm`,
   `--convert-func-to-llvm`, `--reconcile-unrealized-casts`.

Details of `--convert-hip-to-llvm`:

- **Compute ops** lower to `llvm.call @hip_<op_name>(...)`. Memref arguments are
  converted to raw pointers via `MemRefDescriptor(...).allocatedPtr()`.
  Shape metadata is extracted rank-generically from memref descriptors.
- **`memref.alloc` / `memref.dealloc`** (produced by bufferization) are converted
  to device allocation calls (`hip_device_malloc` / `hip_device_free`).
- **Region ops** are inlined: body ops moved to parent block, region op erased.
- **`!hip.handle`** is converted to `!llvm.ptr`.

The `hip-compiler` tool runs this full pipeline automatically, then translates
the resulting LLVM dialect to LLVM IR, generates a native `.obj`, and links it
with `hip_runtime_static.lib` and external libraries to produce a `.dll`.

For manual debugging, `hip-opt` can run the pass pipeline in isolation:
`--one-shot-bufferize="bufferize-function-boundaries"`,
`--convert-hip-to-llvm`, `--finalize-memref-to-llvm`, `--convert-arith-to-llvm`,
`--convert-func-to-llvm`, `--reconcile-unrealized-casts`.

---

## File Structure

```
hip-opt.cpp              MLIR pass pipeline tool (for debugging)
hip-compiler.cpp         One-stop MLIR-to-DLL compiler
HipDialect.td            Dialect definition (namespace "hip")
HipTypes.td              Type definitions (!hip.handle)
HipOps.td                All op definitions (DPS ins/outs format)
HipDialect.h / .cpp      C++ dialect registration + DPS interface implementations
HipPasses.td             Pass definitions via TableGen (convert-hip-to-llvm, convert-onnx-to-hip)
HipPasses.h              Pass declarations (auto-generated from HipPasses.td)
HipToLLVM.cpp            Lowering pass (hip -> llvm.call)
OnnxToHip.cpp            [optional] ONNX-to-HIP conversion pass (requires onnx-mlir)
CMakeLists.txt           Builds hip-opt, hip-compiler, and hip_runtime_static

ops_runtime/                    (compiled into hip_runtime_static.lib by CMake)
  hip_runtime.cpp         Handle lifecycle + device memory (shared by all tests)
  hipblaslt_matmul.cpp    hipBLASLt matmul (full impl)
  miopen_add.cpp          MIOpen add (full impl)
  miopen_mul.cpp          MIOpen mul (full impl)
  miopen_rms_norm.cpp     MIOpen RMS norm (full impl)
  miopen_softmax.cpp      MIOpen softmax (full impl)
  miopen_skip_rms_norm.cpp  Stub
  miopen_rope.cpp         Stub
  transpose.cpp           N-D transpose (full impl, pure C++)
  gather.cpp              Stub
  silu.cpp                Stub
  gqa.cpp                 Stub

examples/
  test_gemm.mlir          Two chained matmuls (DPS, hipBLASLt)
  test_add.mlir           Two chained adds (DPS, MIOpen)
  test_mul.mlir           Two chained muls (DPS, MIOpen)
  test_rms_norm.mlir      Two chained RMS norms (DPS, MIOpen)
  test_softmax.mlir       Two chained softmaxes (DPS, MIOpen)
  test_attention.mlir     Single-head attention from composed ops
  test_e2e.mlir           Self-contained transformer layer
  model_hip.mlir          Generated HIP dialect from Llama-3.2-1B
  main_gemm.cpp           C++ driver for test_gemm (links gemm.lib)
  main_add.cpp            C++ driver for test_add (links add.lib)
  main_mul.cpp            C++ driver for test_mul (links mul.lib)
  main_rms_norm.cpp       C++ driver for test_rms_norm (links rms_norm.lib)
  main_softmax.cpp        C++ driver for test_softmax (links softmax.lib)
  main_attention.cpp      C++ driver for test_attention (links attention.lib)
  main_e2e.cpp            C++ driver for test_e2e

scripts/
  env.bat                          Shared environment config (edit paths here)
  run_full_pipeline_hipblaslt.bat        Matmul: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_add.bat       Add: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_mul.bat       Mul: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_rms_norm.bat  RMS Norm: hip-compiler + cl.exe driver
  run_full_pipeline_miopen_softmax.bat   Softmax: hip-compiler + cl.exe driver
  run_full_pipeline_attention.bat        Attention: hip-compiler + cl.exe driver
```
