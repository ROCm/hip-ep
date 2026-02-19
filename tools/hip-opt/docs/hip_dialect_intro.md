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
| `hip.hipblaslt.matmul` | `(%handle) ins(%A, %B : ...) outs(%C : ...)` | `hip_hipblaslt_matmul(handle, A, B, C, M, K, N)` | Full impl |

The lowering extracts M, K, N from the memref descriptors (A is [M,K], B is [K,N])
and passes them to the runtime alongside the device pointers.

---

## MIOpen Ops

Ops backed by the MIOpen library. Each maps to a specific MIOpen C API call.

### Normalization

| Op | DPS Syntax | MIOpen API | Status |
|---|---|---|---|
| `hip.miopen.rms_norm` | `(%handle) ins(%input, %weight : ...) outs(%output : ...)` | `miopenT5LayerNormForward` | Full impl |
| `hip.miopen.skip_rms_norm` | `(%handle) ins(%x, %skip, %weight : ...) outs(%output, %residual : ...)` | `miopenAddLayerNormForward` (T5 mode) | Stub |

The `rms_norm` lowering extracts N and D from the input memref (input is [N,D]) and
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

---

## Custom HIP Kernel Ops

Ops with no MIOpen or hipBLASLt equivalent. These require custom HIP kernels.

| Op | DPS Syntax | Purpose |
|---|---|---|
| `hip.gather` | `(%handle) ins(%indices, %table : ...) outs(%output : ...)` | Embedding table lookup |
| `hip.silu` | `(%handle) ins(%input : ...) outs(%output : ...)` | SiLU activation: `x * sigmoid(x)` |
| `hip.gqa` | `(%handle, %layer, %start_pos, %seq_len) ins(%q, %k, %v : ...) outs(%kv_cache, %output : ...)` | Grouped query attention with KV cache |

---

## Region Ops (Structural)

Grouping ops that mark which backend library handles a block of compute ops.
No runtime -- inlined and erased during lowering to LLVM.

| Op | Syntax | Purpose |
|---|---|---|
| `hip.miopen.graph` | `hip.miopen.graph { ... }` | Groups MIOpen-dispatched ops |
| `hip.hipblaslt.graph` | `hip.hipblaslt.graph { ... }` | Groups hipBLASLt-dispatched ops |

---

## Example: Two Chained Matmuls (DPS)

```mlir
func.func @two_matmuls(
    %A:  memref<?x?xf32, 1>,
    %B0: memref<?x?xf32, 1>,
    %B1: memref<?x?xf32, 1>,
    %C:  memref<?x?xf32, 1>) {
  %handle = hip.create_handle() : !hip.handle

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32, 1>
  %N = memref.dim %B0, %c1 : memref<?x?xf32, 1>
  %tmp = hip.alloc(%handle, %M, %N) : memref<?x?xf32, 1>

  hip.hipblaslt.matmul(%handle)
      ins(%A, %B0 : memref<?x?xf32, 1>, memref<?x?xf32, 1>)
      outs(%tmp : memref<?x?xf32, 1>)

  hip.hipblaslt.matmul(%handle)
      ins(%tmp, %B1 : memref<?x?xf32, 1>, memref<?x?xf32, 1>)
      outs(%C : memref<?x?xf32, 1>)

  hip.free(%handle, %tmp) : memref<?x?xf32, 1>
  hip.destroy_handle(%handle) : !hip.handle
  return
}
```

`%tmp` is the internally managed intermediate buffer (allocated and freed
within the function). `%C` is the caller-provided output destination.

---

## Lowering

All ops are lowered to LLVM IR by the `--convert-hip-to-llvm` pass in
`HipToLLVM.cpp`:

- **Compute ops** lower to `llvm.call @hip_<op_name>(...)`. Memref arguments are
  converted to raw pointers via `MemRefDescriptor(...).allocatedPtr()`.
  Shape metadata is extracted from memref descriptors and passed to the runtime:
  M/K/N for matmul, numElements for add/mul, N/D for rms_norm.
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
  miopen_skip_rms_norm.cpp  Stub
  miopen_rope.cpp         Stub
  gather.cpp              Stub
  silu.cpp                Stub
  gqa.cpp                 Stub

examples/
  test_gemm.mlir          Two chained matmuls (DPS, hipBLASLt)
  test_add.mlir           Two chained adds (DPS, MIOpen)
  test_mul.mlir           Two chained muls (DPS, MIOpen)
  test_rms_norm.mlir      Two chained RMS norms (DPS, MIOpen)
  test_e2e.mlir           Self-contained transformer layer
  model_hip.mlir          Generated HIP dialect from Llama-3.2-1B
  main_gemm.cpp           C++ driver for test_gemm
  main_add.cpp            C++ driver for test_add
  main_mul.cpp            C++ driver for test_mul
  main_rms_norm.cpp       C++ driver for test_rms_norm
  main_e2e.cpp            C++ driver for test_e2e

scripts/
  run_full_pipeline_hipblaslt.bat        Matmul pipeline (hipBLASLt)
  run_full_pipeline_miopen_add.bat       Add pipeline (MIOpen)
  run_full_pipeline_miopen_mul.bat       Mul pipeline (MIOpen)
  run_full_pipeline_miopen_rms_norm.bat  RMS Norm pipeline (MIOpen)
```
