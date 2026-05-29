// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for the `--hip-infer-shapes` propagation pass.
//
// The pass walks ops with `ReifyRankedShapedTypeOpInterface`, collects the
// reified result shapes, and refines:
//   1. The DPS outs operand's `tensor.empty` producer (the only refinable
//      producer today),
//   2. The op's own result type, in place,
//   3. Each non-DPS use, by inserting a `tensor.cast` so consumers that
//      expect the old type stay well-formed.
//
// DPS-outs use edges are NOT cast — chained DPS ops refine in turn on the
// same module walk.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-infer-shapes %s | FileCheck %s

// --- Single matmul with a fully-static result that the IR types still spell
//     as `?` -> the empty producer + the matmul's result + the function
//     return get refined to the static shape. ---
// CHECK-LABEL: func.func @refine_single_matmul
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x8xf16>
// CHECK:         %[[Y:.*]] = hip.matmul
// CHECK-SAME:                  outs(%[[E]] : tensor<2x8xf16>) : tensor<2x8xf16>
// CHECK:         %[[CAST:.*]] = tensor.cast %[[Y]] : tensor<2x8xf16> to tensor<?x?xf16>
// CHECK:         return %[[CAST]] : tensor<?x?xf16>
func.func @refine_single_matmul(%ctx: !hip.context,
                                %a: tensor<2x4xf16>,
                                %b: tensor<4x8xf16>,
                                %dM: index, %dN: index) -> tensor<?x?xf16> {
  %e = tensor.empty(%dM, %dN) : tensor<?x?xf16>
  %y = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e : tensor<?x?xf16>) : tensor<?x?xf16>
  return %y : tensor<?x?xf16>
}

// -----

// --- Chained matmul: `A @ B @ C`. After the pass, both empties should be
//     refined to their static shapes. The first matmul's result is now
//     `tensor<2x8xf16>`, but the second matmul's `ins` operand was originally
//     typed `tensor<?x?xf16>`; we insert a `tensor.cast` on that non-DPS edge
//     to keep the second matmul's signature well-formed.
//
//     Despite the cast on the ins edge, the second matmul still refines its
//     M dim to 2 because `tensor.dim %cast, 0` folds through the cast (via
//     upstream `tensor::DimOp::fold` / `foldTensorCast`) to the source's
//     static dim — and our reify uses `tensor::getMixedSize`'s `createOrFold`
//     form. Verifies that the per-op refinement in this pass cooperates
//     correctly with upstream cast-folding instead of fighting it.
// CHECK-LABEL: func.func @refine_chained_matmul
// CHECK:         %[[E1:.*]] = tensor.empty() : tensor<2x8xf16>
// CHECK:         %[[Y1:.*]] = hip.matmul
// CHECK-SAME:      outs(%[[E1]] : tensor<2x8xf16>) : tensor<2x8xf16>
// CHECK:         %[[CAST:.*]] = tensor.cast %[[Y1]] : tensor<2x8xf16> to tensor<?x?xf16>
// CHECK:         %[[E2:.*]] = tensor.empty() : tensor<2x16xf16>
// CHECK:         %[[Y2:.*]] = hip.matmul
// CHECK-SAME:      ins(%[[CAST]], %{{.*}} : tensor<?x?xf16>, tensor<8x16xf16>)
// CHECK-SAME:      outs(%[[E2]] : tensor<2x16xf16>) : tensor<2x16xf16>
func.func @refine_chained_matmul(%ctx: !hip.context,
                                 %a: tensor<2x4xf16>,
                                 %b: tensor<4x8xf16>,
                                 %c: tensor<8x16xf16>,
                                 %d1: index, %d2: index,
                                 %d3: index, %d4: index)
    -> tensor<?x?xf16> {
  %e1 = tensor.empty(%d1, %d2) : tensor<?x?xf16>
  %y1 = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e1 : tensor<?x?xf16>) : tensor<?x?xf16>
  %e2 = tensor.empty(%d3, %d4) : tensor<?x?xf16>
  %y2 = hip.matmul(%ctx)
    ins(%y1, %c : tensor<?x?xf16>, tensor<8x16xf16>)
    outs(%e2 : tensor<?x?xf16>) : tensor<?x?xf16>
  return %y2 : tensor<?x?xf16>
}

// -----

// --- Outs producer is NOT a tensor.empty (here it's a function arg) -> the
//     pass refuses to refine that result and leaves the IR untouched. ---
// CHECK-LABEL: func.func @skip_non_empty_producer
// CHECK:         hip.matmul
// CHECK-SAME:    outs(%{{.*}} : tensor<?x?xf16>) : tensor<?x?xf16>
func.func @skip_non_empty_producer(%ctx: !hip.context,
                                   %a: tensor<2x4xf16>,
                                   %b: tensor<4x8xf16>,
                                   %c: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %y = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%c : tensor<?x?xf16>) : tensor<?x?xf16>
  return %y : tensor<?x?xf16>
}

// -----

// --- Idempotence: if the result type is already maximally static, the pass
//     leaves the IR unchanged (no spurious tensor.cast). ---
// CHECK-LABEL: func.func @noop_on_static
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x8xf16>
// CHECK:         %[[Y:.*]] = hip.matmul
// CHECK-SAME:      outs(%[[E]] : tensor<2x8xf16>) : tensor<2x8xf16>
// CHECK-NOT:    tensor.cast
// CHECK:         return %[[Y]] : tensor<2x8xf16>
func.func @noop_on_static(%ctx: !hip.context,
                          %a: tensor<2x4xf16>,
                          %b: tensor<4x8xf16>) -> tensor<2x8xf16> {
  %e = tensor.empty() : tensor<2x8xf16>
  %y = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e : tensor<2x8xf16>) : tensor<2x8xf16>
  return %y : tensor<2x8xf16>
}

// -----

// --- Non-DPS reify-implementing op: tensor.pad's result has dynamic dims at
//     the type level (its `low`/`high` operands here are SSA values), but
//     when those operands are constants in IR the upstream pad reify
//     resolves them to integer attrs. We want to confirm: (a) the pass does
//     not regress on non-DPS ops (no producer rewrite applies, but the
//     result type is refined and a cast is inserted for non-DPS uses), and
//     (b) the pass coexists with upstream reify-supporting ops.
// CHECK-LABEL: func.func @refine_tensor_pad
// CHECK:         %[[P:.*]] = tensor.pad
// CHECK:         tensor<4x8xf16> to tensor<6x10xf16>
// CHECK:         %[[CAST:.*]] = tensor.cast %[[P]]{{.*}}: tensor<6x10xf16> to tensor<?x?xf16>
// CHECK:         return %[[CAST]] : tensor<?x?xf16>
func.func @refine_tensor_pad(%a: tensor<4x8xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f16
  %padded = tensor.pad %a low[%c1, %c1] high[%c1, %c1] {
    ^bb0(%i: index, %j: index):
      tensor.yield %cst : f16
  } : tensor<4x8xf16> to tensor<?x?xf16>
  return %padded : tensor<?x?xf16>
}
