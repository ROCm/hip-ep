# HIP Dialect Reference

The `hip` dialect provides MLIR operations for GPU inference on AMD ROCm.
All ops live under the `hip` namespace. Ops are grouped by the backend library
that implements them at runtime.

---

## Types

| Type | Description |
|---|---|
| `!hip.handle` | Opaque runtime handle for managing HIP library state |

---

## Runtime Lifecycle & Memory

These ops manage the HIP runtime and device memory. They have full runtime
implementations in `hip_gemm_runtime.cpp`.

| Op | Signature | Runtime |
|---|---|---|
| `hip.create_handle` | `() -> !hip.handle` | `hipCreateHandle()` |
| `hip.destroy_handle` | `(!hip.handle) -> ()` | `hipDestroyHandle(handle)` |
| `hip.alloc` | `(handle, dyn_sizes...) -> memref<...x, 1>` | `hipMalloc(size)` + memref descriptor |
| `hip.free` | `(handle, memref) -> ()` | `hipFree(ptr)` |

---

## hipBLASLt Ops

Matrix multiplication backed by the hipBLASLt library (`hipblasLtMatmul`).

| Op | Signature | Runtime | Status |
|---|---|---|---|
| `hip.hipblaslt.matmul` | `(handle, A, B, C) -> ()` | `hipblasLtMatmul` | Stub (`ops_runtime/hipblaslt_matmul.cpp`) |
| `hip.gemm` | `(handle, A, B, C, M, K, N) -> ()` | `hip_gemm_f32` | Full impl (`hip_gemm_runtime.cpp`) |

`hip.gemm` is the original pointer-based GEMM op. `hip.hipblaslt.matmul` is the
newer memref-based version that will replace it.

---

## MIOpen Ops

Ops backed by the MIOpen library. Each maps to a specific MIOpen C API call.

### Normalization

| Op | MIOpen API | Runtime |
|---|---|---|
| `hip.miopen.rms_norm(handle, input, weight, output)` | `miopenT5LayerNormForward` | Stub (`ops_runtime/miopen_rms_norm.cpp`) |
| `hip.miopen.skip_rms_norm(handle, x, skip, weight, output, residual)` | `miopenAddLayerNormForward` (T5 mode) | Stub (`ops_runtime/miopen_skip_rms_norm.cpp`) |

`skip_rms_norm` fuses Add + RMSNorm into a single kernel:
`residual = x + skip; output = RMSNorm(residual) * weight`.
Uses `MIOPEN_ELEMENTWISE_AFFINE_T5` normalization mode.

### Rotary Positional Embeddings

| Op | MIOpen API | Runtime |
|---|---|---|
| `hip.miopen.rope(handle, q, k, cos, sin, start_pos)` | `miopenRotaryPositionalEmbeddings` (experimental) | Stub (`ops_runtime/miopen_rope.cpp`) |

### Element-wise Tensor Ops

| Op | MIOpen API | Runtime |
|---|---|---|
| `hip.miopen.add(handle, A, B, C)` | `miopenOpTensor(miopenTensorOpAdd)` | Stub (`ops_runtime/miopen_add.cpp`) |
| `hip.miopen.mul(handle, A, B, C)` | `miopenOpTensor(miopenTensorOpMul)` | Stub (`ops_runtime/miopen_mul.cpp`) |

---

## Custom HIP Kernel Ops

Ops with no MIOpen or hipBLASLt equivalent. These require custom HIP kernels.
Currently implemented as empty stubs (log the call, output is undefined).

| Op | Purpose | Runtime |
|---|---|---|
| `hip.gather(handle, indices, table, output)` | Embedding table lookup | Stub (`ops_runtime/gather.cpp`) |
| `hip.silu(handle, input, output)` | SiLU activation: `x * sigmoid(x)` | Stub (`ops_runtime/silu.cpp`) |
| `hip.gqa(handle, q, k, v, kv_cache, output, layer, start_pos, seq_len)` | Grouped query attention with KV cache | Stub (`ops_runtime/gqa.cpp`) |

---

## Region Ops (Structural)

Grouping ops that mark which backend library handles a block of compute ops.
No runtime -- inlined and erased during lowering to LLVM.

| Op | Syntax | Purpose |
|---|---|---|
| `hip.miopen.graph` | `hip.miopen.graph { ... }` | Groups MIOpen-dispatched ops |
| `hip.hipblaslt.graph` | `hip.hipblaslt.graph { ... }` | Groups hipBLASLt-dispatched ops |

---

## Lowering

All ops are lowered to LLVM IR by the `--convert-hip-to-llvm` pass in
`HipToLLVM.cpp`:

- **Compute ops** lower to `llvm.call @hip_<op_name>(...)`. Memref arguments are
  converted to raw pointers via `MemRefDescriptor(...).allocatedPtr()`.
- **Region ops** are inlined: body ops moved to parent block, region op erased.
- **`!hip.handle`** is converted to `!llvm.ptr`.

---

## File Structure

```
HipDialect.td          Dialect definition (namespace "hip")
HipTypes.td            Type definitions (!hip.handle)
HipOps.td              All 16 op definitions
HipDialect.h / .cpp    C++ dialect registration
HipToLLVM.cpp          Lowering pass (hip -> llvm.call)
HipPasses.h            Pass registration header
hip_gemm_runtime.cpp   Full runtime for hip.gemm + handle lifecycle
ops_runtime/
  hipblaslt_matmul.cpp
  miopen_rms_norm.cpp
  miopen_skip_rms_norm.cpp
  miopen_rope.cpp
  miopen_add.cpp
  miopen_mul.cpp
  gather.cpp
  silu.cpp
  gqa.cpp
```
