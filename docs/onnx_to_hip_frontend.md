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
| `NhwcConv` (`com.microsoft`) | transpose + `hip.conv` | rank-4 NHWC, weights `[M, kH, kW, C/group]` |
| `Einsum` | transpose / reshape + `hip.matmul` | binary contraction, static shapes, hipBLASLt |

### Normalization

| ONNX / ORT Contrib | HIP | Backend |
|---|---|---|
| `LayerNormalization` | `hip.layer_norm` | custom HIP kernel |
| `InstanceNormalization` | `hip.instance_norm` | custom HIP kernel |
| `BatchNormalization` | `hip.batch_norm` | custom HIP kernel; inference only |
| `GroupNorm` (`com.microsoft`) | `hip.group_norm` | `group_norm_kernel.hip` |
| `RMSNormalization` | `hip.rms_norm` | `rms_norm_kernel.hip` |
| `SimplifiedLayerNormalization` | `hip.rms_norm` | `rms_norm_kernel.hip` |
| `GridSample` | `hip.grid_sample` | custom HIP kernel |
| `SkipLayerNormalization` | `hip.add` + `hip.layer_norm` | decomposed, including the optional input bias |
| `SkipSimplifiedLayerNormalization` | `hip.skip_rms_norm` | `skip_rms_norm_kernel.hip` |
| LpNorm+Mul pattern (fused) | `hip.rms_norm` | `rms_norm_kernel.hip` |

`SkipSimplifiedLayerNormalization` fuses Add + RMSNorm into one kernel:
`residual = x + skip [+ bias]; output = RMSNorm(residual) * weight`.

Both RMS-norm kernels are block-per-row with FP32 accumulation, and take a
packed `__half2` path for fp16 rows of even width.

### Attention

| ONNX / ORT Contrib | HIP |
|---|---|
| `GroupQueryAttention` | `hip.gqa` |

### Quantization

| ONNX | HIP | Backend |
|---|---|---|
| `QuantizeLinear` | `hip.quantize_linear` | `qdq_kernel.hip` |
| `DequantizeLinear` | `hip.dequantize_linear` | `qdq_kernel.hip` |
| `QLinearConv` | `hip.qlinear_conv` | `qlinear_conv_kernel.hip` |
| `QLinearAdd` (`com.microsoft`) | `hip.qadd` | decomposed, then QDQ fusion |
| `QLinearMul` (`com.microsoft`) | `hip.qmul` | decomposed, then QDQ fusion |
| `QLinearConcat` (`com.microsoft`) | DQ + Concat + Q | no fused kernel |
| `QLinearGlobalAveragePool` (`com.microsoft`) | DQ + `hip.global_pool` + Q | `channels_last` transposed around the pool |
| `QLinearMatMul` | `hip.qlinear_matmul` | `qlinear_matmul_kernel.hip` |

`QLinearMatMul` is the native ONNX op: rank-2 8-bit `a` of shape `[M, K]` times
`b` of shape `[K, N]`, with per-tensor f32 scales and zero points. `hip.qmatmul`
is a different op, the DequantizeLinear + MatMul + QuantizeLinear fusion, and
does not accept `QLinearMatMul`. Rank other than 2, f16 or bf16 scales, and
per-row or per-column quantization, stay `onnx.QLinearMatMul`.

`QLinearConv` is the native ONNX op: 8-bit activations and weights, grouped 2D
windows, and an optional int32 bias. Input and output quantization is
per-tensor. Weight quantization is per-tensor or per output channel. `auto_pad`
must be `NOTSET`. `hip.qconv` is a different op, the W4A16 1x1 QDQ fusion, and
does not accept `QLinearConv`.

QuantizeLinear and DequantizeLinear storage is int8/uint8/int16/uint16 plus
int4/uint4. Granularity comes from the shape of `scale` rather than a flag: a
single element is per-tensor, a 1-D tensor is per-axis along `axis`, and
`block_size > 0` is blocked.

int4/uint4 imports as an 8-bit element type at the logical element count, two
values per byte, so the width travels as a `packed_int4` marker rather than in
the type. Constant lowering is the only producer of that marker and can only
reach `DequantizeLinear`, where the halved backing size makes the packing
observable; a `QuantizeLinear` writing 4-bit output looks identical in the IR to
one writing 8-bit. The quantize direction is implemented through the dialect,
lowering, and kernel, but nothing reaches it today, so its packed kernel has no
runtime coverage.

### Activation

| ONNX | HIP | Backend |
|---|---|---|
| `Relu` | `hip.max` against a 0-D zero | `wrap_elementwise` |

### Softmax

| ONNX | HIP | Backend |
|---|---|---|
| `Softmax` | `hip.miopen.softmax` | `hip_miopen_softmax`, custom HIP kernel |

### Element-wise Tensor Ops

| ONNX | HIP | Backend |
|---|---|---|
| `Add` | `hip.add` | `wrap_elementwise` |
| `Mul` | `hip.mul` | `wrap_elementwise` |
| `Sub` | `hip.sub` | `wrap_elementwise_sub` |

### Reduction

| ONNX | HIP | Backend |
|---|---|---|
| `ReduceMean` | `hip.reduce_mean` | custom HIP kernel |

### Concat

| ONNX | HIP | Backend |
|---|---|---|
| `Concat` | `tensor.empty` + `tensor.insert_slice` | bufferizes to destination subviews and copies |

### Zero-cost Metadata Ops (no kernel needed)

| ONNX | HIP | Notes |
|---|---|---|
| `Reshape` | `tensor.expand_shape` / `tensor.collapse_shape` | Zero-cost standard MLIR shape reinterpretation, no custom HIP op needed |
| `Unsqueeze` | `hip.unsqueeze` | Shape/stride reinterpretation only |
| `Squeeze` | `hip.squeeze` | Shape/stride reinterpretation only |

### Custom HIP Kernels (no vendor-library equivalent)

| ONNX | HIP | Notes |
|---|---|---|
| `Transpose` | `hip.transpose` | ND data permutation |
| `Gather` | `hip.gather` | Embedding lookup / index select |
| `Cast` | `hip.cast` | Type conversion |
| `Div` | `hip.div` | Element-wise division |
| `Pow` | `hip.pow` | Element-wise power |
| `Sqrt` | `hip.sqrt` | Element-wise square root |
| `IsNaN` | `hip.isnan` | Float input, 1-byte boolean output |
| `Upsample` | `hip.resize` | Schema 9; asymmetric coordinates, nearest uses floor |
| `HardSigmoid` | `hip.mul` + `hip.add` + `hip.max` + `hip.min` | Clip(alpha*x + beta, 0, 1); decomposed pre-lowering; f16/f32 only |

Unmapped ops default to `hip.<OpType>`.

---

## Validation (Llama-3.2-1B)

Tested on `Llama-3.2-1B-Instruct` quantized ONNX model:

| HIP Op | Count |
|---|---|
| `hip.hipblaslt.matmul` | 80 |
| `hip.rms_norm` | 33 |
| `hip.add` | 32 |
| `hip.mul` | 16 |
| `hip.gqa` | 16 |
| `hip.silu` | 16 |
| `hip.quantize_linear` | 193 |
| `hip.dequantize_linear` | 274 |

Remaining `onnx.*` / `com.microsoft.*` ops: **0**.
