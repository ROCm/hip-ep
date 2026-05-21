// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the `--onnx-loop-outline` pass rewrites each `onnx.Loop` op into a
// `hip.loop` op plus a sibling outlined `func.func` carrying the body, so the
// body's onnx.* ops can be visited by the standard `--convert-onnx-to-hip`
// pass like ops in main_graph.
//
// Source IR is a simplified counted accumulator loop -- the trip count is a
// constant, the body is a single elementwise onnx.Add on a captured outer
// tensor (no onnx.Gather), and the cond is a passthrough `onnx.Identity`.
// This is the loop_add_v1 model from `SimpleTestModels/` with the gather
// slice stripped out, so only the outlining surface area is exercised.
//
// This test validates:
// - onnx.Loop -> hip.loop + outlined `func.func @<parent>_loop_body_n0`
// - 0-D `onnx.Constant<i64>` trip count is folded to `arith.constant ... : index`
//   (avoids a tensor.extract that would later bufferize to a host load from a
//    GPU-resident constant -- same family of bug as PR #212's host-scalar fix)
// - 0-D `onnx.Constant<i1>` and `onnx.Constant<ui8>` cond_init both fold to
//   `arith.constant true` (covers the canonical MLIR and morphizen importer
//   spellings of ONNX BOOL)
// - SSA-equal `cond_out` / `cond_in` (with an intermediate `onnx.Identity`)
//   sets the `cond_is_passthrough` UnitAttr and strips the cond return slot
//   from the outlined body
// - Outer-graph SSA values used inside the body become explicit `captures(...)`
//   operands on hip.loop and re-bound entry args on the outlined body
// - Non-passthrough cond keeps cond_out in the body's return tuple and omits
//   the `cond_is_passthrough` attribute, so the LLVM lowering picks the slow
//   path `hipdnn_ep_run_loop` instead of `hipdnn_ep_run_counted_loop`
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-loop-outline --split-input-file %s | FileCheck %s

// -----

// Test 1: passthrough cond, i1 cond_init.  Loop-carried = %A (accumulator),
// capture = %B.  Body is a single onnx.Add(%acc_in, %B).
module {
  func.func @main_graph(%A: tensor<16xf32>, %B: tensor<16xf32>) -> tensor<16xf32> {
    %M = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %v_final = "onnx.Loop"(%M, %cond_init, %A) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>, %acc_in: tensor<16xf32>):
      %acc_out = "onnx.Add"(%acc_in, %B) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<i1>) -> tensor<i1>
      "onnx.Yield"(%cond_out, %acc_out) : (tensor<i1>, tensor<16xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
    return %v_final : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<16xf32>, %[[B:.*]]: tensor<16xf32>)
  //
  // Trip count folded directly to `index` (no tensor.extract from a GPU constant).
  // CHECK-DAG: %[[M_IDX:.*]] = arith.constant 4 : index
  // Cond folded to i1 (no tensor.extract from a GPU constant).
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  //
  // hip.loop op: passthrough cond, 1 loop-carried (%A), 1 capture (%B).
  // CHECK: %[[RESULT:.*]] = hip.loop(%[[CTX]], %[[M_IDX]], %[[TRUE]])
  // CHECK-SAME: iter_args(%[[A]] : tensor<16xf32>)
  // CHECK-SAME: captures(%[[B]] : tensor<16xf32>)
  // CHECK-SAME: -> (tensor<16xf32>)
  // CHECK-SAME: body @main_graph_loop_body_n0
  // CHECK-SAME: cond_is_passthrough
  // CHECK-SAME: num_loop_carried = 1 : i32
  //
  // CHECK: return %[[RESULT]] : tensor<16xf32>
  //
  // Outlined body func sits at module scope.  Signature:
  //   (!hip.context, iter, cond_in, v_in, capture) -> v_out
  // Cond return slot is stripped under passthrough.
  // CHECK-LABEL: func.func private @main_graph_loop_body_n0
  // CHECK-SAME: (%{{.*}}: !hip.context,
  // CHECK-SAME:  %{{.*}}: tensor<i64>,
  // CHECK-SAME:  %{{.*}}: tensor<i1>,
  // CHECK-SAME:  %[[V_IN:.*]]: tensor<16xf32>,
  // CHECK-SAME:  %[[CAP:.*]]: tensor<16xf32>) -> tensor<16xf32>
  //
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%[[V_IN]], %[[CAP]])
  // CHECK: return %[[ADD]] : tensor<16xf32>
}

// -----

// Test 2: passthrough cond, ui8 cond_init (morphizen importer spelling).
// Same loop topology; verifies unboxCondInit folds dense<1> : tensor<ui8> -> i1
// and that the cloned `onnx.Identity` ops in the body are removed (would
// otherwise fail bufferization since onnx.Identity has no ConvertOnnxToHip
// pattern when its sole use was the yield).
module {
  func.func @main_graph_ui8(%A: tensor<16xf32>, %B: tensor<16xf32>) -> tensor<16xf32> {
    %M = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<ui8>} : () -> tensor<ui8>
    %v_final = "onnx.Loop"(%M, %cond_init, %A) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<ui8>, %acc_in: tensor<16xf32>):
      %acc_out = "onnx.Add"(%acc_in, %B) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<ui8>) -> tensor<ui8>
      "onnx.Yield"(%cond_out, %acc_out) : (tensor<ui8>, tensor<16xf32>) -> ()
    }) : (tensor<i64>, tensor<ui8>, tensor<16xf32>) -> tensor<16xf32>
    return %v_final : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @main_graph_ui8
  // ui8 cond folds to the same `arith.constant true` as the i1 variant.
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK: hip.loop({{.*}}, {{.*}}, %[[TRUE]])
  // CHECK-SAME: cond_is_passthrough
  //
  // Body's cond_in arg type is preserved as tensor<ui8> -- the outliner narrows
  // the OUTER cond_init to i1 (above) but leaves the body's internal cond
  // plumbing in its original type since the only use was the (now-erased)
  // passthrough Identity chain.
  // CHECK-LABEL: func.func private @main_graph_ui8_loop_body_n0
  // CHECK-SAME: %{{.*}}: tensor<ui8>,
  //
  // CHECK-NOT: onnx.Identity
}

// -----

// Test 3: non-passthrough cond (cond_out = Not(cond_in)).  Verifies cond_out
// stays in the returned tuple and `cond_is_passthrough` is NOT set on the
// hip.loop op -- the LLVM lowering reads this to pick `hipdnn_ep_run_loop`
// (dynamic / per-iter cond readback) over `hipdnn_ep_run_counted_loop`.
module {
  func.func @main_graph_dynamic_cond(%A: tensor<16xf32>, %B: tensor<16xf32>) -> tensor<16xf32> {
    %M = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %v_final = "onnx.Loop"(%M, %cond_init, %A) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>, %acc_in: tensor<16xf32>):
      %acc_out = "onnx.Add"(%acc_in, %B) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
      %cond_out = "onnx.Not"(%cond_in) : (tensor<i1>) -> tensor<i1>
      "onnx.Yield"(%cond_out, %acc_out) : (tensor<i1>, tensor<16xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
    return %v_final : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @main_graph_dynamic_cond
  // Note the substring match: cond_is_passthrough would change the printed
  // attr-dict from `{num_loop_carried = ...}` to
  // `{cond_is_passthrough, num_loop_carried = ...}`, so the literal `{num_`
  // requires its absence.
  // CHECK: hip.loop({{.*}}) iter_args({{.*}}) captures({{.*}}) -> (tensor<16xf32>) body @main_graph_dynamic_cond_loop_body_n0 {num_loop_carried = 1 : i32}
  //
  // Body returns (cond_out, v_out).  Op order in the body matches source order
  // (onnx.Add first because acc_out is computed before cond_out in the loop
  // body block), but the return tuple is (cond_out, v_out) per the ONNX
  // Yield order that the outliner mirrors.
  // CHECK-LABEL: func.func private @main_graph_dynamic_cond_loop_body_n0
  // CHECK-SAME: -> (tensor<i1>, tensor<16xf32>)
  // CHECK: %[[ADD:.*]] = "onnx.Add"
  // CHECK: %[[NOT:.*]] = "onnx.Not"
  // CHECK: return %[[NOT]], %[[ADD]] : tensor<i1>, tensor<16xf32>
}
