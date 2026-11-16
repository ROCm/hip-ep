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

Status: `hip.add` only. `hip.qmul`, `hip.qmatmul`, ... still use the
Q-specific ops and their PDLL fusions.

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
| Fold Q/DQ into types | `hip-fusion-transform`, `fusion_pattern/QdqToQuantTypes.pdll` | DQ, DQ → `hip.add` → Q becomes a quant-typed `hip.add` when every Q/DQ has a per-tensor type (splat constant scale, readable zero point) and a kernel takes the combination. |
| Quant-type-agnostic fusions | `hip-fusion-transform` (future) | e.g. Add + ReLU, written once. |
| Legalize | `hip-legalize-quant-types`, before bufferization | Memrefs cannot hold quant types: the [lhs, rhs, output] element types of every add move into its `quant_types` attribute, quantized tensors become storage integers, casts disappear. |
| Lower | `convert-hip-to-llvm` | quantized `quant_types` → `wrap_qelementwise` (scales/zero points from the types); otherwise `wrap_elementwise`. |

## Kernel capability

`hip::supportsQuantTypes(op, inputTypes, resultType)`
(`lib/Dialect/IR/HipQuantTypes.cpp`) is the one place that says which quant
types an op has a kernel for. The fusion and the legalize pass both use it.
For `hip.add`: per-tensor, one storage type for lhs, rhs and output
(i8/ui8/i16/ui16/i32).
