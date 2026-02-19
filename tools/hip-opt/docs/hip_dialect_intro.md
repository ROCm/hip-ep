# HIP Dialect Reference

The `hip` dialect provides MLIR operations for GPU inference on AMD ROCm.
All ops live under the `hip` namespace. Ops are grouped by the backend library
that implements them at runtime.

All compute ops use **Destination-Passing Style (DPS)**: arguments are split
into `ins(...)` (read-only inputs) and `outs(...)` (output destinations
provided by the caller). Handle and scalar parameters are leading positional
arguments.

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

## Example: 3D Matmul with Weight Broadcast (DPS)

```mlir
func.func @two_matmuls(
    %A:  memref<?x?x?xf32, 1>,   // [B, S, K]
    %B0: memref<?x?xf32, 1>,     // [K, N]  (2D, broadcast across batch)
    %B1: memref<?x?xf32, 1>,     // [N, P]
    %C:  memref<?x?x?xf32, 1>) { // [B, S, P]
  %handle = hip.create_handle() : !hip.handle

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %B = memref.dim %A, %c0 : memref<?x?x?xf32, 1>
  %S = memref.dim %A, %c1 : memref<?x?x?xf32, 1>
  %N = memref.dim %B0, %c1 : memref<?x?xf32, 1>
  %tmp = hip.alloc(%handle, %B, %S, %N) : memref<?x?x?xf32, 1>

  hip.hipblaslt.matmul(%handle)
      ins(%A, %B0 : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
      outs(%tmp : memref<?x?x?xf32, 1>)

  hip.hipblaslt.matmul(%handle)
      ins(%tmp, %B1 : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
      outs(%C : memref<?x?x?xf32, 1>)

  hip.free(%handle, %tmp) : memref<?x?x?xf32, 1>
  hip.destroy_handle(%handle) : !hip.handle
  return
}
```

`%A` is 3D `[B,S,K]`, `%B0` is 2D `[K,N]` -- the matmul broadcasts B0 across
all batches (stride_B=0). `%tmp` is the internally managed intermediate.

---

## Lowering

All ops are lowered to LLVM IR by the `--convert-hip-to-llvm` pass in
`HipToLLVM.cpp`:

- **Compute ops** lower to `llvm.call @hip_<op_name>(...)`. Memref arguments are
  converted to raw pointers via `MemRefDescriptor(...).allocatedPtr()`.
  Shape metadata is extracted rank-generically from memref descriptors:
  `rankA/rankB/batch/M/K/N` for matmul, `numElements` for add/mul,
  `rows/cols` (flattened) for softmax and rms_norm,
  `rank/dim0/dim1/shape` for transpose.
- **Region ops** are inlined: body ops moved to parent block, region op erased.
- **`!hip.handle`** is converted to `!llvm.ptr`.

Additional standard passes needed for a full lowering pipeline:
`--finalize-memref-to-llvm`, `--convert-arith-to-llvm`, `--convert-func-to-llvm`,
`--reconcile-unrealized-casts`.

---

## File Structure

```
HipDialect.td            Dialect definition (namespace "hip")
HipTypes.td              Type definitions (!hip.handle)
HipOps.td                All op definitions (DPS ins/outs format)
HipDialect.h / .cpp      C++ dialect registration
HipToLLVM.cpp            Lowering pass (hip -> llvm.call)
HipPasses.h              Pass registration header
hip-opt.cpp              Compiler driver

ops_runtime/
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
  test_attention.mlir     Single-head attention from composed ops
  test_e2e.mlir           Self-contained transformer layer
  model_hip.mlir          Generated HIP dialect from Llama-3.2-1B
  main_gemm.cpp           C++ driver for test_gemm
  main_add.cpp            C++ driver for test_add
  main_mul.cpp            C++ driver for test_mul
  main_rms_norm.cpp       C++ driver for test_rms_norm
  main_attention.cpp      C++ driver for test_attention
  main_e2e.cpp            C++ driver for test_e2e

scripts/
  run_full_pipeline_hipblaslt.bat        Matmul pipeline (hipBLASLt)
  run_full_pipeline_miopen_add.bat       Add pipeline (MIOpen)
  run_full_pipeline_miopen_mul.bat       Mul pipeline (MIOpen)
  run_full_pipeline_miopen_rms_norm.bat  RMS Norm pipeline (MIOpen)
  run_full_pipeline_attention.bat       Attention pipeline (composed ops)
```
