<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX-to-HIP Frontend

Converts ONNX dialect IR (produced by onnx-mlir) into HIP dialect IR at the MLIR
level using destination-passing style (DPS) with tensor types. Bufferization to memref
is handled by a separate `--one-shot-bufferize` pass.

```
model.onnx  -->  onnx-mlir  -->  onnx_dialect.mlir  -->  hip-mlir-opt --convert-onnx-to-hip
                                                           --one-shot-bufferize
                                                           --convert-hip-to-llvm ...
```

## Build Requirements

For general build instructions, see [quick_start.md](quick_start.md).

This feature is **optional** and disabled by default. To enable it, add
`-DONNX_MLIR_SRC=/path/to/onnx-mlir` (and optionally `-DONNX_MLIR_BUILD`)
to your CMake configure command. Without this flag, hip-mlir-opt and
hip-compiler build normally; the `--convert-onnx-to-hip` pass is simply
not registered.

**Known limitation:** onnx-mlir pins to an LLVM 22-dev commit (`0c2701fe7fa0`,
Nov 2025) and is not compatible with LLVM 23. If you are using LLVM 23, leave
`ONNX_MLIR_SRC` unset.

---

## Pass Pipeline

The `--convert-onnx-to-hip` pass performs the following transformations:

1. **Weight extraction** -- `onnx.Constant` ops are promoted to new function arguments
2. **Op mapping** -- ONNX ops are rewritten to HIP dialect ops (tensor DPS):
   - `ONNXMatMulOp` → `hip.hipblaslt.matmul`
   - `ONNXTransposeOp` → `hip.transpose`
   - `ONNXMulOp` → `hip.miopen.mul`
   - `ONNXSoftmaxOp` → `hip.miopen.softmax`
3. **Cleanup** -- `onnx.EntryPoint` is erased

The `hip-add-context-arg` pass must run before this pass to inject `!hip.context`
as function argument 0.

After this pass, the IR is in HIP dialect with tensor types. The standard
`--one-shot-bufferize` pass then converts tensors to memrefs, followed by the
existing `--convert-hip-to-llvm` pipeline.

---

## Op Mapping

### Compute

| ONNX | HIP | Backend |
|---|---|---|
| `MatMul`, `Gemm` | `hip.hipblaslt.matmul` | hipBLASLt |

### Normalization

| ONNX / ORT Contrib | HIP | Backend |
|---|---|---|
| `LayerNormalization` | `hip.layer_norm` | `layer_norm_kernel.hip` |
| `InstanceNormalization` | `hip.instance_norm` | custom HIP kernel |
| `RMSNormalization` | `hip.rms_norm` | `rms_norm_kernel.hip` |
| `SimplifiedLayerNormalization` | `hip.rms_norm` | `rms_norm_kernel.hip` |
| `GridSample` | `hip.grid_sample` | custom HIP kernel |
| `SkipLayerNormalization` | `hip.add` + `hip.layer_norm` | `layer_norm_kernel.hip` |
| `SkipSimplifiedLayerNormalization` | `hip.skip_rms_norm` | `skip_rms_norm_kernel.hip` |
| LpNorm+Mul pattern (fused) | `hip.rms_norm` | `rms_norm_kernel.hip` |

`SkipSimplifiedLayerNormalization` fuses Add + RMSNorm into one kernel:
`residual = x + skip [+ bias]; output = RMSNorm(residual) * weight`.

All four norm kernels are block-per-row with FP32 accumulation, and take a
packed `__half2` path for fp16 rows of even width.

Note that `SkipLayerNormalization` is the mean-subtracting variant and is
decomposed rather than fused; only the Simplified (RMS) variant has a fused op.

### Attention

| ONNX / ORT Contrib | HIP |
|---|---|
| `GroupQueryAttention` | `hip.gqa` |

### Quantization (placeholder, no-op for now)

| ONNX | HIP |
|---|---|
| `QuantizeLinear` | `hip.quantize_linear` |
| `DequantizeLinear` | `hip.dequantize_linear` |

### MIOpen: Activation (`miopenActivationForward`)

| ONNX | HIP | MIOpen mode |
|---|---|---|
| `Relu` | `hip.miopen.relu` | `miopenActivationRELU` |

### MIOpen: Softmax (`miopenSoftmaxForward`)

| ONNX | HIP | MIOpen API |
|---|---|---|
| `Softmax` | `hip.miopen.softmax` | `miopenSoftmaxForward` |

### MIOpen: Element-wise Tensor Ops (`miopenOpTensor`)

| ONNX | HIP | MIOpen mode |
|---|---|---|
| `Add` | `hip.miopen.add` | `miopenTensorOpAdd` |
| `Mul` | `hip.miopen.mul` | `miopenTensorOpMul` |

### MIOpen: Reduction (`miopenReduceTensor`)

| ONNX | HIP | MIOpen mode |
|---|---|---|
| `ReduceMean` | `hip.miopen.reduce_mean` | `MIOPEN_REDUCE_TENSOR_AVG` |

### MIOpen: Concat (`miopenCatForward`, experimental)

| ONNX | HIP | MIOpen API |
|---|---|---|
| `Concat` | `hip.miopen.cat` | `miopenCatForward` |

### Zero-cost Metadata Ops (no kernel needed)

| ONNX | HIP | Notes |
|---|---|---|
| `Reshape` | `tensor.expand_shape` / `tensor.collapse_shape` | Zero-cost standard MLIR shape reinterpretation, no custom HIP op needed |
| `Unsqueeze` | `hip.unsqueeze` | Shape/stride reinterpretation only |
| `Squeeze` | `hip.squeeze` | Shape/stride reinterpretation only |

### Custom HIP Kernels (no MIOpen equivalent)

| ONNX | HIP | Notes |
|---|---|---|
| `Transpose` | `hip.transpose` | ND data permutation |
| `Gather` | `hip.gather` | Embedding lookup / index select |
| `Cast` | `hip.cast` | Type conversion |
| `Div` | `hip.div` | Element-wise division |
| `Pow` | `hip.pow` | Element-wise power |
| `Sqrt` | `hip.sqrt` | Element-wise square root |

Unmapped ops default to `hip.<OpType>`.

---

## Validation (Llama-3.2-1B)

Tested on `Llama-3.2-1B-Instruct` quantized ONNX model:

| HIP Op | Count |
|---|---|
| `hip.hipblaslt.matmul` | 80 |
| `hip.miopen.rms_norm` | 33 |
| `hip.miopen.add` | 32 |
| `hip.miopen.mul` | 16 |
| `hip.gqa` | 16 |
| `hip.silu` | 16 |
| `hip.quantize_linear` | 193 |
| `hip.dequantize_linear` | 274 |

Remaining `onnx.*` / `com.microsoft.*` ops: **0**.
