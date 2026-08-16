// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the `--onnx-loop-outline` pass rewrites each `onnx.Loop` op into a
// `hip.loop` op plus a sibling outlined `func.func` carrying the body, so the
// body's onnx.* ops can be visited by the standard `--convert-onnx-to-hip`
// pass like ops in main_graph.
//
// Source IR is a minimal counted accumulator loop -- constant trip count,
// a single elementwise onnx.Add on a captured outer tensor, and a
// passthrough `onnx.Identity` cond chain.  This is intentionally simpler
// than the `loop_add_v1` model under `SimpleTestModels/` (which has a
// dynamic trip count via `Squeeze(Dim(input))`, two adds, and a gather);
// the goal here is to exercise only the outlining surface area.
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
// - Missing `cond_init` (operand is `onnx.NoValue` -> `none`) synthesizes an
//   i1-true and forces `cond_is_passthrough` so the runtime takes the fast
//   counted-loop path -- matches ONNX spec where missing cond means M is the
//   only termination condition
// - When v_init's operand type is more refined than the source onnx.Loop body
//   block v_in arg (canonical case: importer emits `tensor<*xT>` for body
//   block args inside Loop subgraphs), v_carry types on the outlined body
//   func's entry block AND the `hip.loop` result types both come from v_init
//   -- the outliner does not propagate the under-refined block arg type
//   through. Refinement of the cloned body ops is left for `--hip-infer-shapes`
//   post-conversion (see `docs/design/unranked-tensor-handling.md`)
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
  //   (!hip.context, iter, cond_in, v_in, capture, !hip.loop_frame)
  //       -> (i32 status, v_out)
  // Cond return slot is stripped under passthrough.
  // CHECK-LABEL: func.func private @main_graph_loop_body_n0
  // CHECK-SAME: (%{{.*}}: !hip.context,
  // CHECK-SAME:  %{{.*}}: tensor<i64>,
  // CHECK-SAME:  %{{.*}}: tensor<i1>,
  // CHECK-SAME: tensor<16xf32>, %{{.*}}: tensor<16xf32>,
  // CHECK-SAME: !hip.loop_frame) -> (i32, tensor<16xf32>)
  //
  // CHECK: %[[ADD:.*]] = "onnx.Add"
  // CHECK: return %{{.*}}, %[[ADD]] : i32, tensor<16xf32>
}

// -----

// Test 2: passthrough cond, ui8 cond_init (morphizen importer spelling).
// Same loop topology; verifies that unboxCondInit folds dense<1> :
// tensor<ui8> -> i1, and that the now-dead `onnx.Identity` op cloned into
// the body is erased.  (Standard DCE leaves it behind because onnx.Identity
// is unregistered and MLIR conservatively assumes unregistered ops have
// side effects, so the outliner has to erase those Identity chains itself
// -- see LoopOutline.cpp:352-372.  Without that erase, the dead Identity
// reaches `--convert-onnx-to-hip` with no matching pattern and later
// fails bufferization with "op was not bufferized".)
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
  // Body's cond_in arg type is preserved as tensor<ui8>.  The outliner
  // narrows only the OUTER cond_init (the operand fed into hip.loop) to i1
  // via unboxCondInit; the body's block-arg types are copied verbatim from
  // the source onnx.Loop block, regardless of how cond_in is used inside
  // (see LoopOutline.cpp:285-286).
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
  // The attr-dict is matched literally as `{num_loop_carried = 1 : i32}`.
  // If `cond_is_passthrough` were set, the printer would emit
  // `{cond_is_passthrough, num_loop_carried = 1 : i32}` instead, and this
  // CHECK line would fail to match -- so the literal `{num_` after the
  // body symbol is what enforces the attribute's absence.
  // CHECK: hip.loop({{.*}}) iter_args({{.*}}) captures({{.*}}) -> (tensor<16xf32>) body @main_graph_dynamic_cond_loop_body_n0 {num_loop_carried = 1 : i32}
  //
  // Body returns (cond_out, v_out).  Op order in the body matches source order
  // (onnx.Add first because acc_out is computed before cond_out in the loop
  // body block), but the return tuple is (cond_out, v_out) per the ONNX
  // Yield order that the outliner mirrors.
  // CHECK-LABEL: func.func private @main_graph_dynamic_cond_loop_body_n0
  // CHECK-SAME: -> (i32, tensor<i1>, tensor<16xf32>)
  // CHECK: %[[ADD:.*]] = "onnx.Add"
  // CHECK: %[[NOT:.*]] = "onnx.Not"
  // CHECK: return %{{.*}}, %[[NOT]], %[[ADD]] : i32, tensor<i1>, tensor<16xf32>
}

// -----

// Test 4: missing cond_init (counted loop).  ONNX Loop spec marks operand 1
// as optional; importers spell its absence as `onnx.NoValue` with `none`
// type.  ONNX semantics is "cond stays true forever; only max_trip_count
// terminates" so the body's yielded cond_out is ignored.  The outliner
// synthesizes `arith.constant true : i1` for hip.loop's cond_init and
// forces `cond_is_passthrough` -- the LLVM lowering then picks the fast
// path `hipdnn_ep_run_counted_loop`, identical to a dynamically-detected
// passthrough case.  Matches the canonical HF vision-encoder counted-
// attention-loop topology (Loop with no cond_init, fixed trip count
// from the encoder depth, body is one iter of attention).
module {
  func.func @main_graph_no_cond(%A: tensor<16xf32>, %B: tensor<16xf32>) -> tensor<16xf32> {
    %M = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.NoValue"() {value} : () -> none
    %v_final = "onnx.Loop"(%M, %cond_init, %A) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>, %acc_in: tensor<16xf32>):
      %acc_out = "onnx.Add"(%acc_in, %B) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
      %cond_out = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
      "onnx.Yield"(%cond_out, %acc_out) : (tensor<i1>, tensor<16xf32>) -> ()
    }) : (tensor<i64>, none, tensor<16xf32>) -> tensor<16xf32>
    return %v_final : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @main_graph_no_cond
  //
  // Trip count folds as before; cond_init is synthesized (no source
  // operand to fold).
  // CHECK-DAG: %[[M_IDX:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  //
  // hip.loop op: passthrough forced, 1 loop-carried (%A), 1 capture (%B).
  // CHECK: hip.loop({{.*}}, %[[M_IDX]], %[[TRUE]])
  // CHECK-SAME: cond_is_passthrough
  // CHECK-SAME: num_loop_carried = 1 : i32
  //
  // Body func skips the cond return slot (passthrough mode).  cond_in arg
  // is preserved as tensor<i1> (block-arg type copied verbatim).
  // CHECK-LABEL: func.func private @main_graph_no_cond_loop_body_n0
  // CHECK-SAME: %{{.*}}: tensor<i1>,
  // CHECK-SAME: tensor<16xf32>, %{{.*}}: tensor<16xf32>,
  // CHECK-SAME: !hip.loop_frame) -> (i32, tensor<16xf32>)
  //
  // CHECK: %[[ADD:.*]] = "onnx.Add"
  // CHECK: return %{{.*}}, %[[ADD]] : i32, tensor<16xf32>
}

// -----

// Test 5: outline-time v_init plumbing.  When the v_init operand carries
// a more-refined type than the source onnx.Loop body block v_in arg
// (canonical case: importer emits `tensor<*xT>` for body block args
// inside a Loop subgraph because ONNX shape inference does not recurse
// into Loop bodies, while the v_init feeding the loop from the outer
// graph carries the importer-derived rank), the outliner must source
// the v_carry entry arg from v_init -- NOT from the body block arg.
// Both invariants below are required for the loop verifier
// (`hip.loop result_type[i] == v_init[i].type`) and for the LLVM
// lowering's trampoline construction, which reads the body func
// argument types directly to build per-arg memref descriptor structs
// (see LoopLowering.cpp:155 and `inference_compute`'s call boundary).
// Refinement of the cloned body ops' types happens later in the
// pipeline via `--hip-infer-shapes`; this test pins only what
// LoopOutline itself produces.
module {
  func.func @main_graph_v_init_refined(%A: tensor<16xf32>) -> tensor<16xf32> {
    %M = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %v_final = "onnx.Loop"(%M, %cond_init, %A) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>, %acc_in: tensor<?xf32>):
      %acc_out = "onnx.Identity"(%acc_in) : (tensor<?xf32>) -> tensor<?xf32>
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<i1>) -> tensor<i1>
      "onnx.Yield"(%cond_out, %acc_out) : (tensor<i1>, tensor<?xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
    return %v_final : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @main_graph_v_init_refined
  // The joined carrier contract widens because the body current/yield are
  // dynamic even though v_init and the source Loop result are static.
  // CHECK: %[[SEED_CAST:.*]] = tensor.cast %{{.*}} : tensor<16xf32> to tensor<?xf32>
  // CHECK: %[[R:.*]] = hip.loop
  // CHECK-SAME: iter_args(%[[SEED_CAST]] : tensor<?xf32>)
  // CHECK-SAME: -> (tensor<?xf32>)
  // CHECK-SAME: body @main_graph_v_init_refined_loop_body_n0
  // CHECK: %[[RESULT_CAST:.*]] = tensor.cast %[[R]] : tensor<?xf32> to tensor<16xf32>
  // CHECK: return %[[RESULT_CAST]] : tensor<16xf32>
  //
  // Body current/result slots use the same joined dynamic contract.
  // CHECK-LABEL: func.func private @main_graph_v_init_refined_loop_body_n0
  // CHECK-SAME: (%{{.*}}: !hip.context,
  // CHECK-SAME:  %{{.*}}: tensor<i64>,
  // CHECK-SAME:  %{{.*}}: tensor<i1>,
  // CHECK-SAME:  %[[V_IN:.*]]: tensor<?xf32>,
  // CHECK-SAME:  %{{.*}}: !hip.loop_frame) -> (i32, tensor<?xf32>)
}

// -----

// Test 6: regression closure for the canonical Loop-body shape gap.
//
// Source pattern lifted from a real HF vision-encoder ONNX export with
// an `onnx.Loop` whose body terminates in a Concat of the v_carry
// block arg and a captured rank-3 tensor. The importer's shape
// inference has no per-iter rank for the body block arg (no caller
// has visited the loop yet at import time), so it emits
// `tensor<*xf16>` (unranked) for that arg and for every body-internal
// value derived from it. The `onnx.Loop`'s declared result type, by
// contrast, mirrors the v_init operand fed in from the outer graph
// (rank-3 dynamic).
//
// Pre-fix, OnnxLoopOutlinePass took `loopOp->getResultTypes()`
// verbatim for hip.loop's result types when the source onnx.Loop's
// declared result was rank-0 / unranked, and the loop verifier
// (correctly) rejected:
//
//   error: 'hip.loop' op result type #0 ('tensor<*xf16>') must match
//          v_init type #0 ('tensor<?x?x?xf16>')
//
// Post-fix: hip.loop result types come from `LoopOp::inferReturnTypes`
// (reads v_init); the outlined body func's v_carry arg slot comes
// from v_init too. The cloned `onnx.Concat` still carries its source
// unranked result type at outline-pass exit -- that's refined to
// ranked post-conversion by `--hip-infer-shapes` via
// `ReifyRankedShapedTypeOpInterface`. This test pins only the
// outline-time invariants.
module {
  func.func @vision_encoder_loop(%attn_in: tensor<?x?x?xf16>,
                                 %M: tensor<i64>,
                                 %newshape: tensor<4xi64>)
      -> tensor<1x?x16x72xf16> {
    %cond_init = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Loop"(%M, %cond_init, %attn_in) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<ui8>, %acc: tensor<*xf16>):
      %step = "onnx.Concat"(%acc, %attn_in) {axis = 1 : si64}
              : (tensor<*xf16>, tensor<?x?x?xf16>) -> tensor<*xf16>
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<ui8>) -> tensor<ui8>
      "onnx.Yield"(%cond_out, %step) : (tensor<ui8>, tensor<*xf16>) -> ()
    }) : (tensor<i64>, none, tensor<?x?x?xf16>) -> tensor<*xf16>
    %y = "onnx.Reshape"(%r, %newshape) {allowzero = 0 : si64}
         : (tensor<*xf16>, tensor<4xi64>) -> tensor<1x?x16x72xf16>
    return %y : tensor<1x?x16x72xf16>
  }

  // CHECK-LABEL: func.func @vision_encoder_loop
  //
  // hip.loop's result type is sourced from v_init via
  // `LoopOp::inferReturnTypes`. Without that, the unranked source
  // result type would flow through and the loop verifier would reject.
  // CHECK: %[[R:.*]] = hip.loop
  // CHECK-SAME: iter_args(%{{.*}} : tensor<?x?x?xf16>)
  // CHECK-SAME: -> (tensor<?x?x?xf16>)
  // CHECK-SAME: cond_is_passthrough
  //
  // Body func arg slot 3 (v_carry) is the rank-3 v_init type, NOT the
  // unranked type the original onnx.Loop body block declared for
  // %acc. The cloned onnx.Concat keeps its source unranked result
  // type; the body func's declared return type matches. Refinement
  // happens later via `--hip-infer-shapes`.
  // CHECK-LABEL: func.func private @vision_encoder_loop_loop_body_n0
  // CHECK-SAME: (%{{.*}}: !hip.context,
  // CHECK-SAME:  %{{.*}}: tensor<i64>,
  // CHECK-SAME:  %{{.*}}: tensor<ui8>,
  // CHECK-SAME: tensor<?x?x?xf16>, %{{.*}}: tensor<?x?x?xf16>,
  // CHECK-SAME: !hip.loop_frame) -> (i32, tensor<?x?x?xf16>)
  //
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"
  // CHECK-SAME: -> tensor<*xf16>
  // CHECK: tensor.cast %[[CONCAT]] : tensor<*xf16> to tensor<?x?x?xf16>
  // CHECK: return %{{.*}}, %{{.*}} : i32, tensor<?x?x?xf16>
}
