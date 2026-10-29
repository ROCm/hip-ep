<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Supported operations

This reference summarizes ONNX operations handled by the built-in ONNX-to-HIP pipeline. Support may be provided by a vendor library, a custom HIP kernel, decomposition into other operations, or standard MLIR tensor transformations.

The conversion registrations in `lib/Conversion/OnnxToHip/OnnxToHip.cpp` and the corresponding LIT/numeric tests are the source of truth. Individual operations may have data-type, rank, attribute, or dynamic-shape restrictions that are documented in their conversion and test files.

## Runtime-backed operations

| Operation | Backend or lowering |
|---|---|
| Conv | Custom HIP kernel. Rank-3 (NCL) is rewritten to a unit-height 2D conv; rank-4 (NCHW) and rank-5 (NCDHW) lower directly |
| NhwcConv (`com.microsoft`) | Rank-4 NHWC activations and `[M, kH, kW, C/group]` weights are transposed into `hip.conv` and the result is transposed back. Static weights are permuted while lowering |
| ConvTranspose | Custom HIP kernel |
| MatMul | hipBLASLt |
| Einsum | Static binary contraction decomposed to Transpose + MatMul (hipBLASLt). An equation with no contracted index, such as `bc,d->bcd`, is a reshape plus `hip.mul` when the element type is f16 or f32 and the rank is at most 4 |
| Gemm | hipBLASLt |
| Transpose | Custom HIP kernel |
| Mul | Custom HIP kernel |
| Add | Custom HIP kernel |
| ImageScaler | Decomposed to `hip.mul`, or `hip.add` then `hip.mul` when the per-channel bias is non-zero. NCHW only; bias broadcasts as `1xCx1x1`. |
| Sum | Decomposed to Add. Variadic elementwise sum with NumPy broadcast; same rank and dtype limits as Add |
| Softmax | Custom HIP kernel |
| Sigmoid | Custom HIP kernel |
| Tanh | Custom HIP kernel |
| Softplus | Custom HIP kernel (f32/f16) |
| Gelu | Custom HIP kernel; `com.microsoft` Gelu is the erf form of `onnx.Gelu` |
| Swish | Custom HIP kernel; `com.microsoft` QuickGelu is this formula with alpha default 1.702 |
| ScaledSiLU (`com.microsoft`) | Decomposed to Mul + Swish (alpha 1). f16/f32 |
| GatedAdd (`com.microsoft`) | Decomposed to Mul + Add. f16/f32, rank 1-4 |
| SwiGLU (`hip.sigmoid` followed by two `hip.mul`) | Fused by `hip-fusion-transform` into one custom HIP kernel when the tensors are identical f16/f32/bf16/f64, each dynamic axis traces to the same runtime extent, and the intermediates have no other value uses |
| BiasGelu (`com.microsoft`) | Custom HIP kernel |
| FastGelu (`com.microsoft`) | Custom HIP kernel |
| Reciprocal | Custom HIP kernel |
| Sqrt | Custom HIP kernel |
| Exp | Custom HIP kernel |
| Log | Custom HIP kernel |
| Pow | Decomposed to Mul / Sqrt / Reciprocal for supported constant scalar exponents. Other constant scalar exponents use a custom HIP kernel |
| Sub | Custom HIP kernel |
| Cast | Custom HIP kernel |
| CastLike | Simplified to Cast |
| Ceil | Custom HIP kernel |
| Round | Custom HIP kernel |
| Floor | Custom HIP kernel |
| Neg | Custom HIP kernel |
| Equal | Custom HIP kernel |
| Not | Custom HIP kernel |
| IsNaN | Custom HIP kernel |
| And | Custom HIP kernel |
| Or | Custom HIP kernel |
| Abs | Custom HIP kernel |
| Cos | Custom HIP kernel |
| Erf | Custom HIP kernel |
| Sin | Custom HIP kernel |
| Atan | Custom HIP kernel |
| Div | Custom HIP kernel |
| Mod | Custom HIP kernel |
| Sign | Custom HIP kernel |
| Where | Custom HIP kernel |
| Less | Custom HIP kernel |
| Greater | Decomposed to `Less(B, A)` |
| GreaterOrEqual | Decomposed to `Not(Less(A, B))` |
| LessOrEqual | Decomposed to `Not(Less(B, A))` |
| Min | Custom HIP kernel |
| Max | Custom HIP kernel |
| ReduceSum | Custom HIP kernel |
| ReduceMax | Custom HIP kernel |
| ReduceMin | Custom HIP kernel |
| ReduceProd | Custom HIP kernel |
| ReduceMean | Custom HIP kernel |
| CumSum | Custom HIP kernel |
| Pad | Custom HIP kernel. The opset<11 form, with `pads` and the fill `value` as attributes, is rewritten to the operand form first |
| Tile | Custom HIP kernel |
| Expand | Custom HIP kernel |
| GatherND | Custom HIP kernel |
| ScatterND | Custom HIP kernel (reductions: none / add / mul / min / max) |
| ScatterElements | Custom HIP kernel (reductions: none / add / mul / min / max) |
| Range | Custom HIP kernel |
| Size | Custom HIP kernel; folds to a constant for static shapes |
| NonZero | Custom HIP kernel |
| Gather | Custom HIP kernel |
| GatherElements | Custom HIP kernel |
| TopK | Custom HIP kernel |
| RandomNormalLike | Custom HIP kernel. Output f16, bf16, f32, and f64; rank 0–8. The input contributes only its shape, including dynamic dimensions. `seed` makes the fill reproducible inside hip-ep; the generator is not bit-identical to another runtime. An absent seed comes from the clock. |
| ArgMax | Custom HIP kernel. f16, bf16, f32, f64, i8, ui8, i16, ui16, i32, and i64; rank 1–8. ui32 and ui64 are not supported. An empty reduction axis fails. NaN outranks every number; the first NaN wins unless select_last_index keeps the last. |
| Compress | Custom HIP kernel; a dynamic selected extent is scanned and read back before allocation |
| OneHot | Custom HIP kernel |
| LayerNormalization | Custom HIP kernel |
| InstanceNormalization | Custom HIP kernel |
| BatchNormalization | Custom HIP kernel; inference only (`training_mode` stays on ONNX) |
| GroupNorm (`com.microsoft`) | Custom HIP kernel; optional SiLU, NCHW or NHWC |
| SkipLayerNormalization (`com.microsoft`) | Decomposed to Add + LayerNormalization, including the optional input bias |
| RMSNormalization | Custom HIP kernel |
| SimplifiedLayerNormalization | Custom HIP kernel |
| SkipSimplifiedLayerNormalization (`com.microsoft`) | Custom HIP kernel, add and norm fused |
| BranchwiseRMSNorm (`com.microsoft`) | Lowered to RMSNormalization over the branch axis. f16/f32; the scale input is required |
| GatedRMSNorm (`com.microsoft`) | Custom HIP kernel; silu, swish or sigmoid gate. f16/f32 |
| EngramGate (`com.microsoft`) | Custom HIP kernel, including the optional `gated_value_normed` output. f16/f32 |
| HyperConnectionPreMix (`com.microsoft`) | Custom HIP kernel; folds a preceding Sigmoid. f16/f32 |
| HyperConnectionPostMix (`com.microsoft`) | Custom HIP kernel; optional stream mix, fused with a following BranchwiseRMSNorm. f16/f32 |
| LpNormalization | Decomposed to Mul / ReduceSum / Sqrt / Div |
| RotaryEmbedding (`com.microsoft`) | Custom HIP kernel |
| RotaryEmbedding (`ai.onnx`) | Custom HIP kernel |
| MRotaryEmbedding (`com.microsoft`) | Custom HIP kernel. Rank-3 input, `interleaved=0`, `is_packed_batching=0`, f16/f32 |
| GroupQueryAttention (`com.microsoft`) | Custom HIP kernels and hipBLASLt |
| SparseAttentionIndexer (`com.microsoft`) | Custom HIP kernels, wave32 (RDNA) only. `qsa` policy with packed QK, head size 128, up to 8 heads, `token_budget / compress_ratio` ≤ 512, f16/f32; no `past_sequence_length` |
| DynamicSparseAttention (`com.microsoft`) | Custom HIP kernel (WMMA), wave32 (RDNA) only. `selected_only` over the main KV cache, head size 128 or 256, up to 16 query heads per KV head, f16; no rotary, q/k norm or `head_sink` |
| MultiHeadAttention (`com.microsoft`) | Lowered to GroupQueryAttention or decomposed hipBLASLt/custom-kernel paths |
| Attention (`com.microsoft`) | Fused QKV split into GroupQueryAttention. Equal Q/K/V sizes may be inferred from a `[H, 3H]` weight. Bidirectional and causal, with static or dynamic batch and sequence. A rank-1 length or rank-2 padding mask becomes an additive attention bias. |
| Attention (`ai.onnx`, opset 23/24) | Lowered to GroupQueryAttention for supported rank-3/rank-4, causal/masked, output, and KV-cache forms |
| MatMulNBits (`com.microsoft`) | Custom HIP kernel |
| MatMulBnb4 (`com.microsoft`) | HIP dequant of FP4 (`quant_type` 0) or NF4 (`quant_type` 1) into `[N, K]`, then `hip.matmul` with `transB = 1`. The first packed value is the high nibble. Static shapes pull that dequant into the rocMLIR matmul kernel. A dynamic activation leaves the dequant on HIP and the GEMM on hipBLASLt. Block size 16, 32, 64, 128, or 256. `transB` must be 1 and `training_mode` must be 0 |
| QMoE (`com.microsoft`) | Custom HIP kernel |
| GatherBlockQuantized (`com.microsoft`) | Custom HIP kernel |
| QuantizeLinear | Custom HIP kernel |
| DequantizeLinear | Custom HIP kernel |
| QLinearConv | Custom HIP kernel. 8-bit grouped NCHW, optional int32 bias. `NOTSET` uses the pads attribute. `VALID`, `SAME_UPPER`, and `SAME_LOWER` are resolved to explicit pads. Separate from the W4A16 `hip.qconv` fusion |
| QLinearAdd (`com.microsoft`) | Decomposed to DequantizeLinear + Add + QuantizeLinear; per-tensor scales fuse to `hip.qadd` |
| QLinearMul (`com.microsoft`) | Decomposed to DequantizeLinear + Mul + QuantizeLinear; per-tensor scales fuse to `hip.qmul` |
| QLinearConcat (`com.microsoft`) | Decomposed to DequantizeLinear + Concat + QuantizeLinear |
| QLinearGlobalAveragePool (`com.microsoft`) | Decomposed to DequantizeLinear + GlobalAveragePool + QuantizeLinear. `channels_last` is transposed around the pool |
| QLinearAveragePool (`com.microsoft`) | Decomposed to DequantizeLinear + AveragePool + QuantizeLinear. `channels_last` is transposed around the pool. Rank 3–5, int8 or uint8. Omitted strides are 1. `y_zero_point` may be omitted |
| QLinearLeakyRelu (`com.microsoft`) | Decomposed to DequantizeLinear + LeakyRelu + QuantizeLinear. int8 or uint8. `alpha` defaults to 0.01. `Y_zero_point` may be omitted |
| QLinearMatMul | Custom HIP kernel. Rank-2 8-bit. A and Y scales and zero points are per-tensor. B's scale and zero point are per-tensor or one value per column. Separate from the `hip.qmatmul` QDQ fusion |
| LinearAttention (`com.microsoft`) | Custom HIP kernel |
| GatedDeltaNet (`com.microsoft`) | Custom HIP kernel. Packed QKV, `gate_activation=qwen`, `beta_activation=sigmoid`, `qk_l2_norm=1`, f16/f32; no `cu_seqlens` or state capture |
| CausalConvWithState (`com.microsoft`) | Custom HIP kernel |
| NGramHashMapping (`com.microsoft`) | Custom HIP kernel. int32/int64 |
| Relu | Decomposed to Max |
| LeakyRelu | Custom HIP kernel |
| PRelu | Custom HIP kernel. Slope broadcasts onto X; f16/bf16/f32/f64 |
| LRN | Custom HIP kernel. Channel window, rank >= 2, f16/bf16/f32/f64 |
| Trilu | Custom HIP kernel. Keeps the upper or lower triangle of the last two dimensions; `k` must be a constant scalar |
| Clip | Decomposed to Max + Min |
| HardSigmoid | Decomposed to Mul + Add + Clip (`alpha` defaults to 0.2, `beta` to 0.5). f16 and f32 only; bf16 and f64 are left to another EP, since the emitted elementwise ops have no runtime path for them |
| MaxPool | Custom HIP kernel (f16/bf16/f32/f64, and i8/ui8) |
| AveragePool | Custom HIP kernel |
| LpPool | Custom HIP kernel |
| Resize | Custom HIP kernel. nearest and linear on f16/bf16/f32/f64 and on int8/uint8. Integer linear interpolates the stored codes and truncates toward zero. A dynamic extent comes from a constant `scales` vector, or from a `sizes` vector read back to a host index when `scales` is absent |
| Upsample | Lowered through Resize (`asymmetric`, nearest `floor`). Schema 9 passes `scales` as an input; opset 7–8 stores them as an f32 attribute |
| GridSample | Custom HIP kernel |
| GlobalAveragePool | Custom HIP kernel |
| GlobalMaxPool | Custom HIP kernel |
| GlobalLpPool | Custom HIP kernel |

## Control flow

| Operation | Implementation |
|---|---|
| If | Outlined then/else functions with runtime branch dispatch. The condition is a single-element `i1`, `i8`, or `ui8` tensor, including a unit shape such as `tensor<1xui8>` |
| Loop | Outlined loop body with runtime loop dispatch |

## Compiler-optimized operations

These operations are handled through standard MLIR transformations and generally do not require operation-specific runtime backend support.

| Operation | Implementation | Notes |
|---|---|---|
| Reshape | `tensor.expand_shape` / `tensor.collapse_shape` | Metadata-only where representable; dynamic dimensions are supported |
| Unsqueeze | `tensor.expand_shape` | Inserts size-one axes |
| Squeeze | `tensor.collapse_shape` | Removes size-one axes |
| Split | `tensor.extract_slice` | Produces tensor slices that bufferize to views |
| Slice | `tensor.extract_slice` or `hip.slice` | Constant positive-stride forms decompose to tensor slices; runtime indices or negative steps use the runtime path. The opset<10 form, with starts/ends/axes as attributes, is rewritten to the operand form first |
| Concat | `tensor.empty` + `tensor.insert_slice` | Bufferizes to destination subviews and copies |
| Shape | `tensor.dim` + `tensor.from_elements` | Static shapes fold to constants; dynamic dimensions remain runtime SSA |
| Constant | `arith.constant` or external constants file | Large values are externalized |
| ConstantOfShape | `arith.constant` when foldable | Produces a splat constant for constant shape inputs |
| Identity | SSA value forwarding | No runtime operation |
| Flatten | `tensor.collapse_shape` and, where needed, `tensor.expand_shape` | Metadata-only where representable |
| DepthToSpace | `tensor.expand_shape`, `hip.transpose`, `tensor.collapse_shape` | Rank-4 NCHW. `blocksize >= 1`, `C` divisible by `blocksize^2`, and output `[N, C / blocksize^2, H * blocksize, W * blocksize]`. Modes `DCR` (default) and `CRD`. `blocksize == 1` is the input |
| SpaceToDepth | `tensor.expand_shape`, `hip.transpose`, `tensor.collapse_shape` | Rank-4 NCHW. `blocksize >= 1`, `H` and `W` divisible by `blocksize`, and output `[N, C * blocksize^2, H / blocksize, W / blocksize]`. Inverse of DepthToSpace `DCR`. `blocksize == 1` is the input |

## Fusion and preprocessing

The frontend also recognizes or simplifies selected patterns before operation conversion, including approximate/FastGelu chains, Erf-based Gelu chains, supported Pow forms, LpNormalization decompositions, CastLike simplification, and model-exporter cleanup patterns.
