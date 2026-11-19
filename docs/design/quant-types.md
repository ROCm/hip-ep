<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Quantization as element types

A hip op is quantized when its tensors have `!quant.uniform` element types
(upstream MLIR `quant` dialect). The same op with float tensors is the float
op. So there is one `hip.add` for QDQ and float models, and a fusion written
against `hip.add` applies to both. The types, not the op name, select the
kernel and decide whether the kernel dequantizes/quantizes.

Status: `hip.add`, `hip.matmul` and `hip.gemm` (`HipQuantCapableOpInterface`). `hip.qmul`, `hip.qconv`, ...
still use the Q-specific ops and their PDLL fusions, as do the cases below that
types cannot carry.

## IR

```mlir
// Float model
%s = hip.add(%ctx) ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
                   outs(%e : tensor<8xf32>) -> tensor<8xf32>

// QDQ model, after hip-fusion-transform
%qa = hip.storage_cast %a : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.25:32768>>
%qb = hip.storage_cast %b : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.5:32768>>
%s  = hip.add(%ctx) ins(%qa, %qb : ...) outs(%e : tensor<8x!quant.uniform<u16:f32, 0.125:32768>>)
%r  = hip.storage_cast %s : tensor<8x!quant.uniform<u16:f32, 0.125:32768>> to tensor<8xui16>
```

- `hip.storage_cast` is a bit-preserving view between a quantized tensor and
  its hip storage tensor (`uiN` for unsigned, signless `iN` for signed). It
  exists because `quant.scast` rejects `uiN`. A cast pair that restores the
  original type folds, so consecutive quantized ops connect directly.
- `hip.add` verifies that its tensors are all quantized or all not.
- After legalization, every `hip.add` carries `quant_types`: the quantized
  types for a QDQ add, its own float or integer types for any other
  (`[f32, f32, f32]`). Lowering asks only whether those entries are quantized,
  so a `ui16` quantized add is never mistaken for an integer add.

## Flow

| Step | Where | What |
|---|---|---|
| Fold Q/DQ into types | `hip-fold-qdq-to-quant-types` (`FoldQdqToQuantTypes.cpp`), run just before `hip-fusion-transform` | DQ → op → Q becomes the quant-typed op when every Q/DQ has a per-tensor type (splat constant scale, readable zero point, integer storage, not packed). Kernels are not checked here. |
| Quant-type-agnostic fusions | `hip-fusion-transform`, PDLL in `fusion_pattern/` (`MatmulBiasFusion.pdll`) and native `gemm_to_matmul.cpp` | Read element types only, so float and quant-typed chains share them. See below. |
| Legalize | `hip-legalize-quant-types`, before bufferization | Memrefs cannot hold quant types: the [lhs, rhs, output] element types of every add move into its `quant_types` attribute, quantized tensors become storage integers, casts disappear. |
| Lower | `convert-hip-to-llvm` | quantized `quant_types` → `wrap_qelementwise` (scales/zero points from the types); otherwise `wrap_elementwise`. |

## Matmul and gemm

Same fold, same attribute: `hip.matmul` and `hip.gemm` are quantized by their
element types, and `quant_types` holds [A, B, (C), output] after legalization.
`hip.gemm` keeps `alpha`, `beta`, `transA` and `transB` as attributes. Lowering
calls the existing `wrap_qmatmul` / `wrap_qgemm` with the same arguments the
attribute forms (`hip.qmatmul`, `hip.qgemm`) produce, so the kernels are
unchanged.

What stays on `hip.qmatmul` / `hip.qgemm` (QMatMulFusion.pdll,
QGemmFusion.pdll), because it is not expressible as a per-tensor type:

- per-column weights: their scales and zero points are file-backed constants,
  not type parameters;
- packed 4-bit weights: two values per byte, not one per element.

Matmul + bias Add is not fused into a gemm yet.

## Quant-capable ops

An op opts in through `HipQuantCapableOpInterface`
(`include/hip/Dialect/IR/HipQuantCapableOpInterface.td`):

- `getQuantOperands()` lists the tensors that carry the quantization, in
  `quant_types` order (data inputs, optional ones only when present, then the
  output). The interface verifier checks all-or-none quantization and, after
  legalization, that `quant_types` matches the storage types.
- `supportsQuantTypes(inputs, result)` is the op's kernel capability.

`hip-legalize-quant-types` and lowering only go through the interface, so
moving another op over means implementing these two methods.

## Kernel capability

Each op's `supportsQuantTypes` (`lib/Dialect/IR/HipQuantTypes.cpp`) says which
quant types it has a kernel for. Only `hip-legalize-quant-types` asks it; the
folds and fusions do not (see "Known gaps").

- `hip.add`: per-tensor, one storage type for lhs, rhs and output
  (i8/ui8/i16/ui16/i32).
- `hip.matmul`: per-tensor; A and result 8/16-bit, B 8-bit (the kernel splits
  only A into byte planes), optional bias 8/16/32-bit.
- `hip.gemm`: as matmul, with C as the bias. `alpha`, `beta`, `transA` and
  `transB` stay attributes.

Lowering reads the scales and zero points from `quant_types` and calls the
same `wrap_qelementwise` / `wrap_qmatmul` / `wrap_qgemm` as the Q-specific ops,
so the kernel contract is unchanged.

## Gemm (`gemm_to_matmul.cpp`)

A `hip.gemm` that is a plain matmul plus a row bias (alpha = 1; no C, or
beta = 1 with a `[N]` / `[1, N]` C) becomes `hip.matmul` (with its bias), so
fusions and quant types are written against one op. Other gemms stay
`hip.gemm` and keep their own Q/DQ folds. A weight with per-channel or packed
4-bit Q/DQ also keeps `hip.gemm`, since only `QGemmFusion.pdll` handles those.
The rewrite is type-agnostic, so its order against the Q/DQ folds does not
matter (the generic fold covers the converted form).

## Matmul + bias (`MatmulBiasFusion.pdll`)

`hip.matmul` followed by a row-bias `hip.add` becomes one `hip.matmul` with an
optional `bias` operand (rank-1 `[N]`; either add operand order; a `[1, N]`
bias is collapsed to `[N]`):

```mlir
%y = hip.matmul(%ctx) ins(%a, %w, %bias : tensor<1x196x512xT>, tensor<512x1536xT>, tensor<1536xT>)
                      outs(%init : tensor<1x196x1536xT>) : tensor<1x196x1536xT>
```

The op keeps A's shape; its leading dims fold into the rows at lowering (a
view, as for a batched matmul), so there is no reshape in the IR. That needs a
rank-2 weight and `transA = 0` for rank > 2 A, which the op verifier enforces.
Dynamic extents are fine.

The pattern checks structure only, not kernels: float and quant-typed chains
fuse alike. A float chain must use one element type throughout. Lowering: a
quantized matmul with a bias calls `wrap_qgemm`, a float one `wrap_gemm`
(hipBLASLt bias epilogue, with a `GemmState` op state).

- **Intermediate quantization is dropped.** A QDQ model has a Q/DQ between the
  matmul and the bias add; the fused kernel accumulates, adds the bias and
  quantizes once. The result is not bit-identical to the unfused chain.
- The Q/DQ must first fold into types, and the matmul and add must agree on the
  intermediate quant type, i.e. no requantization between them.
- Not covered: `transA` with rank > 2, and an add whose Q/DQ do not fold (e.g.
  an int32 bias against a 16-bit matmul result).

## Known gaps

**The Q/DQ folds do not check kernels (temporary).** A fold fires whenever
every Q/DQ has a per-tensor quant type, so that later fusions are not blocked
by a kernel that only the unfused op would need. Example: an int32-bias
MatMul + Add, whose add (u16 + i32 -> u16) has no elementwise kernel but is
fused into `hip.matmul(a, w, bias)`, which has one.

The kernel question is asked once, in `hip-legalize-quant-types`, and a typed op
without a kernel is a compile error there (no fallback to DQ -> op -> Q). Known
cases that hit it today:

- **PSI:** its 48 attention matmuls are A16W16 (`u16 x u16 -> u16`); the
  quantized matmul kernel takes 8-bit B only. PSI does not compile until an
  A16W16 quantized matmul kernel exists.
- A mixed-width add that is not fused away (e.g. u16 + i8 -> u16).
- A matmul with a 16-bit weight.

Revisit once the missing kernels exist or the fallback policy is decided.
