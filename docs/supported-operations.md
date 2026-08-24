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
| Conv | MIOpen |
| ConvTranspose | MIOpen |
| MatMul | hipBLASLt |
| Gemm | hipBLASLt |
| Transpose | Custom HIP kernel |
| Mul | Custom HIP broadcast kernel through rank 4; static rank > 4 when broadcast-safe packing to rank ≤ 4 is possible |
| Add | Custom HIP broadcast kernel through rank 4; MIOpen fallback through rank 4; static rank > 4 when broadcast-safe packing to rank ≤ 4 is possible |
| Softmax | Custom HIP kernel |
| Sigmoid | MIOpen |
| Tanh | MIOpen |
| Softplus | MIOpen |
| Gelu | Custom HIP kernel |
| BiasGelu (`com.microsoft`) | Custom HIP kernel |
| FastGelu (`com.microsoft`) | Custom HIP kernel |
| Reciprocal | Custom HIP kernel |
| Sqrt | Custom HIP kernel |
| Exp | Custom HIP kernel |
| Log | Custom HIP kernel |
| Pow | Decomposed to Mul / Sqrt / Reciprocal for supported constant scalar exponents |
| Sub | Custom HIP kernel through rank 4; static rank > 4 when broadcast-safe packing to rank ≤ 4 is possible |
| Cast | Custom HIP kernel |
| CastLike | Simplified to Cast |
| Ceil | Custom HIP kernel |
| Neg | Custom HIP kernel |
| Equal | Custom HIP kernel |
| Not | Custom HIP kernel |
| And | Custom HIP kernel |
| Or | Custom HIP kernel |
| Abs | Custom HIP kernel |
| Cos | Custom HIP kernel |
| Sin | Custom HIP kernel |
| Div | Custom HIP kernel through rank 4; static rank > 4 when broadcast-safe packing to rank ≤ 4 is possible |
| Mod | Custom HIP kernel |
| Sign | Custom HIP kernel |
| Where | Custom HIP kernel |
| Less | Custom HIP kernel through rank 4; static rank > 4 when broadcast-safe packing to rank ≤ 4 is possible |
| Greater | Decomposed to `Less(B, A)`; static rank > 4 when broadcast-safe packing to rank ≤ 4 is possible |
| GreaterOrEqual | Decomposed to `Not(Less(A, B))` |
| LessOrEqual | Decomposed to `Not(Less(B, A))` |
| Min | MIOpen |
| Max | MIOpen |
| ReduceSum | Custom HIP kernel |
| ReduceMax | Custom HIP kernel |
| ReduceMin | Custom HIP kernel |
| ReduceProd | Custom HIP kernel |
| ReduceMean | Custom HIP kernel |
| CumSum | Custom HIP kernel |
| Pad | Custom HIP kernel |
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
| Compress | Custom HIP kernel; a dynamic selected extent is scanned and read back before allocation |
| OneHot | Custom HIP kernel |
| LayerNormalization | Custom HIP kernel |
| SkipLayerNormalization (`com.microsoft`) | Decomposed to Add + LayerNormalization |
| RMSNormalization | MIOpen |
| SimplifiedLayerNormalization | MIOpen |
| SkipSimplifiedLayerNormalization (`com.microsoft`) | MIOpen |
| LpNormalization | Decomposed to Mul / ReduceSum / Sqrt / Div |
| RotaryEmbedding (`com.microsoft`) | Custom HIP kernel |
| RotaryEmbedding (`ai.onnx`) | Custom HIP kernel |
| GroupQueryAttention (`com.microsoft`) | Custom HIP kernels and hipBLASLt |
| MultiHeadAttention (`com.microsoft`) | Lowered to GroupQueryAttention or decomposed hipBLASLt/custom-kernel paths |
| Attention (`com.microsoft`) | Fused QKV split and GroupQueryAttention path for supported forms |
| Attention (`ai.onnx`, opset 23/24) | Lowered to GroupQueryAttention for supported rank-3/rank-4, causal/masked, output, and KV-cache forms |
| MatMulNBits (`com.microsoft`) | Custom HIP kernel |
| QMoE (`com.microsoft`) | Custom HIP kernel |
| GatherBlockQuantized (`com.microsoft`) | Custom HIP kernel |
| LinearAttention (`com.microsoft`) | Custom HIP kernel |
| CausalConvWithState (`com.microsoft`) | Custom HIP kernel fast paths with MIOpen fallback |
| Relu | Decomposed to Max |
| LeakyRelu | Custom HIP kernel |
| Clip | Decomposed to Max + Min |
| MaxPool | Custom HIP kernel |
| AveragePool | Custom HIP kernel |
| LpPool | Custom HIP kernel |
| Resize | Custom HIP kernel |
| GlobalAveragePool | Custom HIP kernel |
| GlobalMaxPool | Custom HIP kernel |
| GlobalLpPool | Custom HIP kernel |

For high-rank Add, Sub, Mul, Div, Less, and Greater, convert-onnx-to-hip first
packs static ONNX shapes to rank ≤ 4 when grouping is broadcast-safe (the same
pre-lowering pattern-set wiring as the Gather/Reshape shape folds), then maps
the packed operation to hip.*. Shapes that cannot be packed are mapped at their
original rank; HIP-to-LLVM currently rejects rank > 4.

## Control flow

| Operation | Implementation |
|---|---|
| If | Outlined then/else functions with runtime branch dispatch |
| Loop | Outlined loop body with runtime loop dispatch |

## Compiler-optimized operations

These operations are handled through standard MLIR transformations and generally do not require operation-specific runtime backend support.

| Operation | Implementation | Notes |
|---|---|---|
| Reshape | `tensor.expand_shape` / `tensor.collapse_shape` | Metadata-only where representable; dynamic dimensions are supported |
| Unsqueeze | `tensor.expand_shape` | Inserts size-one axes |
| Squeeze | `tensor.collapse_shape` | Removes size-one axes |
| Split | `tensor.extract_slice` | Produces tensor slices that bufferize to views |
| Slice | `tensor.extract_slice` or `hip.slice` | Constant positive-stride forms decompose to tensor slices; runtime indices or negative steps use the runtime path |
| Concat | `tensor.empty` + `tensor.insert_slice` | Bufferizes to destination subviews and copies |
| Shape | `tensor.dim` + `tensor.from_elements` | Static shapes fold to constants; dynamic dimensions remain runtime SSA |
| Constant | `arith.constant` or external constants file | Large values are externalized |
| ConstantOfShape | `arith.constant` when foldable | Produces a splat constant for constant shape inputs |
| Identity | SSA value forwarding | No runtime operation |
| Flatten | `tensor.collapse_shape` and, where needed, `tensor.expand_shape` | Metadata-only where representable |

## Fusion and preprocessing

The frontend also recognizes or simplifies selected patterns before operation conversion, including approximate/FastGelu chains, Erf-based Gelu chains, supported Pow forms, LpNormalization decompositions, CastLike simplification, and model-exporter cleanup patterns.
