// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --refine-loop-body-types re-infers stale onnx.* result types in an
// outlined hip.loop body function from the (already-refined) operand types
// and propagates the refined return type into the function signature.
//
// This is the canonical pattern produced by --onnx-loop-outline on a model
// whose ONNX exporter annotated the iter_var with v_init's degenerate
// rank-0 sentinel type while the actual v_init operand value is rank-3.
// LoopOutline corrects the function arg type to match the v_init operand,
// then the cloned body's onnx.Concat still claims rank-0 result. Without
// this pass, --convert-onnx-to-hip rejects the Concat (rank-0 result not
// allowed).
// ============================================================================

// RUN: hip-mlir-opt %s --refine-loop-body-types | FileCheck %s

module {
  // Outer function with a hip.loop call. Iter_arg type is rank-3 (correct,
  // already set by loop-outline from the v_init operand).
  func.func @main_graph(%ctx: !hip.context, %m: index, %cond: i1,
                        %v: tensor<?x?x?xf16>)
      -> (tensor<?x?x?xf16>) {
    %r = hip.loop(%ctx, %m, %cond)
             iter_args(%v : tensor<?x?x?xf16>)
             -> (tensor<?x?x?xf16>)
             body @body
             {num_loop_carried = 1 : i32}
    return %r : tensor<?x?x?xf16>
  }

  // Outlined body with the canonical bug pattern: cloned Concat has rank-3
  // operands (correct, post-LoopOutline) but a stale rank-0 result type
  // (annotated by the exporter). Return type matches Concat → also rank-0.
  func.func private @body(%ctx: !hip.context, %iter: tensor<i64>,
                          %cond: tensor<ui8>, %acc: tensor<?x?x?xf16>,
                          %new: tensor<?x?x?xf16>)
      -> tensor<f16> {
    %concat = "onnx.Concat"(%acc, %new) {axis = 1 : si64}
        : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<f16>
    return %concat : tensor<f16>
  }
}
// CHECK-LABEL: func.func private @body
// Refined function return type updated to match the refined yield operand.
// CHECK-SAME: -> tensor<?x?x?xf16>
// Refined Concat result type matches operand rank/element-type with axis=1
// summed (dynamic), other axes tightened from the first ranked operand.
// CHECK: "onnx.Concat"
// CHECK-SAME: -> tensor<?x?x?xf16>
// Return op operand matches the refined sig.
// CHECK: return {{.*}} : tensor<?x?x?xf16>
