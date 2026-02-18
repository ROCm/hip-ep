# ONNX-to-HIP Frontend

Converts `.onnx` models directly into HIP dialect MLIR via a single Python script.

```
model.onnx  -->  onnx_to_hip.py --all  -->  model_hip.mlir  -->  hip-opt  -->  ...
```

## Usage

```bash
pip install onnx numpy

python onnx_to_hip.py model.onnx --all -o model_hip.mlir
```

| Flag | Effect |
|---|---|
| *(default)* | Remap all ops from `onnx.*` / `com.microsoft.*` to `hip.*` |
| `--extract-weights` | Move weights from inline constants to function arguments |
| `--fuse` | Fuse RMSNorm (LpNorm+Mul) and SiLU (Sigmoid+Mul) patterns |
| `--memref` | Convert `tensor<...>` to `memref<..., 1>` (device memory) |
| `--lifecycle` | Add `hip.create_handle` / `hip.destroy_handle` wrapping |
| `--all` | All of the above |

---

## Op Mapping

### Compute

| ONNX | HIP | Backend |
|---|---|---|
| `MatMul`, `Gemm` | `hip.hipblaslt.matmul` | hipBLASLt |

### Normalization

| ONNX / ORT Contrib | HIP | MIOpen C API |
|---|---|---|
| `LayerNormalization` | `hip.miopen.layer_norm` | `miopenLayerNormForward` |
| `SimplifiedLayerNormalization` | `hip.miopen.t5_layer_norm` | `miopenT5LayerNormForward` |
| `SkipLayerNormalization` | `hip.miopen.skip_layer_norm` | `miopenAddLayerNormForward` |
| `SkipSimplifiedLayerNormalization` | `hip.miopen.skip_rms_norm` | `miopenAddLayerNormForward` (T5 mode) |
| LpNorm+Mul pattern (fused) | `hip.miopen.rms_norm` | `miopenT5LayerNormForward` |

`SkipSimplifiedLayerNormalization` fuses Add + RMSNorm into one kernel:
`residual = x + skip; output = RMSNorm(residual) * weight`. In MIOpen this is
`miopenAddLayerNormForward` with mode `MIOPEN_ELEMENTWISE_AFFINE_T5`.

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
| `Sigmoid` | `hip.miopen.sigmoid` | `miopenActivationLOGISTIC` |
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
| `Reshape` | `hip.reshape` | Shape/stride reinterpretation only |
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

## Pattern Fusion (`--fuse`)

Detects multi-op patterns and replaces them with single fused ops.
Q/DQ pairs between component ops are skipped during pattern matching.

**RMSNorm**: `LpNormalization(p=2, axis=-1)` -> [Q/DQ] -> `Mul(_, weight)` => `hip.miopen.rms_norm(x, weight)`

**SiLU**: `Sigmoid(x)` -> [Q/DQ] -> `Mul(x, sigmoid_out)` => `hip.silu(x)`

---

## Backend Graph Grouping

Consecutive ops targeting the same backend are wrapped in graph regions:

- `hip.hipblaslt.*` ops -> `hip.hipblaslt.graph { ... }`
- `hip.miopen.*` ops -> `hip.miopen.graph { ... }`
- Other ops remain ungrouped

---

## Validation (Llama-3.2-1B)

Tested on `Llama-3.2-1B-Instruct` quantized ONNX model with `--all`:

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
