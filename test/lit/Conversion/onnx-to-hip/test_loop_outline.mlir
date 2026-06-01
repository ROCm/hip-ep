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
  // CHECK-SAME: -> (tensor<i1>, tensor<16xf32>)
  // CHECK: %[[ADD:.*]] = "onnx.Add"
  // CHECK: %[[NOT:.*]] = "onnx.Not"
  // CHECK: return %[[NOT]], %[[ADD]] : tensor<i1>, tensor<16xf32>
}

// -----

// Test 4: missing cond_init (counted loop).  ONNX Loop spec marks operand 1
// as optional; importers spell its absence as `onnx.NoValue` with `none`
// type.  ONNX semantics is "cond stays true forever; only max_trip_count
// terminates" so the body's yielded cond_out is ignored.  The outliner
// synthesizes `arith.constant true : i1` for hip.loop's cond_init and
// forces `cond_is_passthrough` -- the LLVM lowering then picks the fast
// path `hipdnn_ep_run_counted_loop`, identical to a dynamically-detected
// passthrough case.  Matches Qwen3.5 vision encoder loop topology.
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
  // CHECK-SAME: %[[V_IN:.*]]: tensor<16xf32>,
  // CHECK-SAME: %[[CAP:.*]]: tensor<16xf32>) -> tensor<16xf32>
  //
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%[[V_IN]], %[[CAP]])
  // CHECK: return %[[ADD]] : tensor<16xf32>
}

// -----

// PR #265 commit 2 — body func v_carry arg type comes from the v_init
// operand type, NOT from the original onnx.Loop body block v_in arg
// type. The two diverge whenever upstream shape inference has refined
// the SSA value feeding v_init but the body block region was not
// re-shape-inferred (the canonical HF ONNX export shape — see Qwen
// regression test below).
//
// Here we drift them artificially: %A is `tensor<16xf32>` so v_init is
// rank-1 static, while the body block v_in arg is declared
// `tensor<?xf32>` (rank-1 dynamic). After outlining:
//
//   * hip.loop.result_type[0]  = `tensor<16xf32>` (from
//     `LoopOp::inferReturnTypes`, which reads v_init).
//   * outlined-func arg slot 3 = `tensor<16xf32>` (sourced from v_init,
//     not from the under-refined body block arg).
//
// Both invariants are required for the loop verifier
// (`result_type[i] == v_init[i].type`) and for the LLVM lowering's
// trampoline construction, which reads the body func argument types
// directly to build per-arg memref descriptor structs (see
// LoopLowering.cpp:155 and `inference_compute`'s call boundary).
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
  // hip.loop.result_type derived from v_init (tensor<16xf32>), not from
  // the under-refined onnx.Loop body output (tensor<?xf32>).
  // CHECK: %[[R:.*]] = hip.loop
  // CHECK-SAME: iter_args(%{{.*}} : tensor<16xf32>)
  // CHECK-SAME: -> (tensor<16xf32>)
  // CHECK-SAME: body @main_graph_v_init_refined_loop_body_n0
  // CHECK: return %[[R]] : tensor<16xf32>
  //
  // Body func arg slot 3 (v_carry) has the REFINED v_init type
  // (tensor<16xf32>), not the under-refined body block v_in arg type
  // (tensor<?xf32>).
  // CHECK-LABEL: func.func private @main_graph_v_init_refined_loop_body_n0
  // CHECK-SAME: (%{{.*}}: !hip.context,
  // CHECK-SAME:  %{{.*}}: tensor<i64>,
  // CHECK-SAME:  %{{.*}}: tensor<i1>,
  // CHECK-SAME:  %[[V_IN:.*]]: tensor<16xf32>) -> tensor<?xf32>
}

// -----

// PR #265 commits 2 + 5 — Qwen3.5-9B vision encoder regression closure.
//
// Lifted shape from a real HF Qwen3.5-9B vision-encoder ONNX export
// (search the production IR dump for `onnx.Loop`). The export ships an
// `onnx.Loop` whose body terminal `Concat(rank-0-block-arg,
// rank-3-mha-output) -> rank-0` declares a rank-0 v_carry-out type
// because shape inference does not recurse into Loop body regions;
// the `onnx.Loop`'s declared result follows that rank-0 yield. The v_init
// operand fed in from the outer graph, by contrast, was shape-inferred
// to a rank-3 dynamic type (`tensor<?x?x?xf16>`).
//
// Pre-PR-265, OnnxLoopOutlinePass took `loopOp->getResultTypes()`
// verbatim for hip.loop's result types, so the bad rank-0 type flowed
// straight through and the loop verifier (correctly) rejected:
//
//   error: 'hip.loop' op result type #0 ('tensor<f16>') must match
//          v_init type #0 ('tensor<?x?x?xf16>')
//
// 10+ instances per compile, 3 compiles per session — the EP silently
// fell back to CPU and the encoder did not actually offload to the
// MorphiZen EP.
//
// Post-PR-265 commit 2: hip.loop's result types come from
// `LoopOp::inferReturnTypes` (reads v_init); the outlined body func's
// arg slot 3 comes from v_init too. Both are `tensor<?x?x?xf16>` here.
// But the cloned `onnx.Concat` still declared rank-0 -- ConvertOnnxToHip's
// rank-aware pattern bailed -> one-shot-bufferize aborted -> still silent
// CPU fallback (just relocated the failure point).
//
// Post-PR-265 commit 5 (this CHECK shape): `refineClonedBodyOpTypes`
// dispatches `OnnxResultTypeInferenceInterface` on each cloned body op.
// The Concat rule sees operand[0] is rank-0 (rank-mismatched relative
// to operand[1]'s rank-3) and returns `tensor<?x?x?xf16>` -- rank
// correct, axis dim conservatively dynamic. The body Concat result
// gets rewritten in place; the declared body func return type catches
// up via `newFn.setType`. The pipeline now actually converts and
// bufferizes the body.
module {
  func.func @qwen_vision_loop(%attn_in: tensor<?x?x?xf16>,
                              %M: tensor<i64>,
                              %newshape: tensor<4xi64>)
      -> tensor<1x?x16x72xf16> {
    %cond_init = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Loop"(%M, %cond_init, %attn_in) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<ui8>, %acc: tensor<f16>):
      %step = "onnx.Concat"(%acc, %attn_in) {axis = 1 : si64}
              : (tensor<f16>, tensor<?x?x?xf16>) -> tensor<f16>
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<ui8>) -> tensor<ui8>
      "onnx.Yield"(%cond_out, %step) : (tensor<ui8>, tensor<f16>) -> ()
    }) : (tensor<i64>, none, tensor<?x?x?xf16>) -> tensor<f16>
    %y = "onnx.Reshape"(%r, %newshape) {allowzero = 0 : si64}
         : (tensor<f16>, tensor<4xi64>) -> tensor<1x?x16x72xf16>
    return %y : tensor<1x?x16x72xf16>
  }

  // CHECK-LABEL: func.func @qwen_vision_loop
  //
  // hip.loop's result type is sourced from v_init via
  // `LoopOp::inferReturnTypes`. Pre-PR-265 it was `tensor<f16>` (the
  // rank-0 onnx.Loop declared result), which the verifier rejected.
  // CHECK: %[[R:.*]] = hip.loop
  // CHECK-SAME: iter_args(%{{.*}} : tensor<?x?x?xf16>)
  // CHECK-SAME: -> (tensor<?x?x?xf16>)
  // CHECK-SAME: cond_is_passthrough
  //
  // Body func arg slot 3 (v_carry) is the REFINED rank-3 v_init type,
  // NOT the rank-0 type the original onnx.Loop body block declared
  // for %arg4. Declared return type is also rank-3 dynamic (commit 5:
  // `refineClonedBodyOpTypes` propagated the rank from operands via
  // the Concat rule; the `newFn.setType` post-refine call synced the
  // function signature).
  // CHECK-LABEL: func.func private @qwen_vision_loop_loop_body_n0
  // CHECK-SAME: (%{{.*}}: !hip.context,
  // CHECK-SAME:  %{{.*}}: tensor<i64>,
  // CHECK-SAME:  %{{.*}}: tensor<ui8>,
  // CHECK-SAME:  %{{.*}}: tensor<?x?x?xf16>,
  // CHECK-SAME:  %{{.*}}: tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  //
  // Cloned Concat result type was tensor<f16> (rank-0 placeholder) in
  // the source onnx.Loop body; commit 5 promotes it to all-dynamic at
  // rank-3 via the interface's Concat rule. Without this the next
  // pipeline pass `--convert-onnx-to-hip` bails on the rank mismatch
  // and the pipeline aborts at one-shot-bufferize -> silent CPU
  // fallback.
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"
  // CHECK-SAME: -> tensor<?x?x?xf16>
  // CHECK: return %[[CONCAT]] : tensor<?x?x?xf16>
}

// -----

// PR #265 commit 5 — `refineClonedBodyOpTypes` interface dispatch.
//
// Minimal repro of the canonical Qwen pattern: a single
// `onnx.Add(rank-0-block-arg, rank-3-capture) -> rank-0` in the body.
// The block-arg is rank-0 in the source onnx.Loop because shape
// inference did not recurse; commit 2 flips it to rank-3 (sourcing
// from the v_init operand type). Commit 5's interface dispatch then
// promotes the Add result via the broadcast rule: max(rank-0,
// rank-3) = rank-3, all-dim broadcast yields the rank-3 shape from
// the higher-rank operand verbatim.
//
// What this case PINS beyond `qwen_vision_loop` above is the body op
// refinement HELPER's contract in isolation: a body op whose result
// has a stale rank-0 placeholder + at least one higher-rank operand
// gets its result type rewritten by the interface, and the outlined
// func's declared return type catches up via `newFn.setType`. The
// `onnx.Add` choice (instead of Concat) exercises the broadcast rule
// rather than the concat-axis-sum path.
//
// `func.func @... -> ()` (void): the source `onnx.Loop`'s rank-0
// declared result is unused at parse time, sidestepping the verifier
// constraint that `return %r : T` must agree with the function's
// declared return type at parse time -- which would clash with this
// test's whole point of asserting the loop result gets refined to
// rank-3 mid-pass. Letting `%r` go dead post-outlining is fine for a
// LIT-only fixture.
module {
  func.func @refine_add_rank0_to_rank3(
      %attn: tensor<?x?x?xf16>, %M: tensor<i64>) {
    %cond = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Loop"(%M, %cond, %attn) ({
    ^bb0(%it: tensor<i64>, %ci: tensor<ui8>, %acc: tensor<f16>):
      %step = "onnx.Add"(%acc, %attn)
              : (tensor<f16>, tensor<?x?x?xf16>) -> tensor<f16>
      %co = "onnx.Identity"(%ci) : (tensor<ui8>) -> tensor<ui8>
      "onnx.Yield"(%co, %step) : (tensor<ui8>, tensor<f16>) -> ()
    }) : (tensor<i64>, none, tensor<?x?x?xf16>) -> tensor<f16>
    return
  }

  // CHECK-LABEL: func.func @refine_add_rank0_to_rank3
  //
  // Outlined body func: declared return rank-3 (post-`newFn.setType`).
  // CHECK-LABEL: func.func private @refine_add_rank0_to_rank3_loop_body_n0
  // CHECK-SAME: -> tensor<?x?x?xf16>
  //
  // Cloned Add result type promoted from rank-0 to rank-3 by the
  // interface's broadcast rule.
  // CHECK: %[[ADD:.*]] = "onnx.Add"
  // CHECK-SAME: -> tensor<?x?x?xf16>
  // CHECK: return %[[ADD]] : tensor<?x?x?xf16>
}

// -----

// PR #265 commit 5 — interface safety belt.
//
// Verifies that the `OnnxResultTypeInferenceInterface` rules library
// correctly returns null Type (caller leaves the op alone) for ops
// outside the library's coverage. `onnx.ReduceSum` with `keepdims=0`
// over all axes IS a real rank-3 -> rank-0 op (the source IR is
// correct). The interface MUST NOT promote its result back to rank-3,
// or downstream passes would receive an op claiming rank-3 output
// where the actual computation produces a scalar.
//
// The mechanism: the rules library has no rule for `onnx.ReduceSum`,
// so `iface.computeResultType(0)` returns null. The helper's
// `if (!infRanked) continue;` then skips the op entirely. This is the
// safety belt against new export shapes adding rank-changing ops we
// have not yet reasoned about: silently-correct, never miscompiled.
module {
  func.func @reduce_sum_stays_rank0(
      %x: tensor<?x?x?xf32>, %M: tensor<i64>) -> tensor<f32> {
    %cond = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Loop"(%M, %cond, %x) ({
    ^bb0(%it: tensor<i64>, %ci: tensor<ui8>, %acc: tensor<?x?x?xf32>):
      // ReduceSum with keepdims=0 over all axes: rank-3 -> rank-0.
      // This is a TRUE rank-changing op -- the helper must leave it
      // alone or the IR would lie about the result rank.
      %sum = "onnx.ReduceSum"(%acc) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
             : (tensor<?x?x?xf32>) -> tensor<f32>
      // Identity of the rank-3 acc -- result is also rank-3, but its
      // operand is rank-3 too, so the helper's rank-promotion guard
      // (infRanked.getRank() > curRanked.getRank()) doesn't fire.
      %step = "onnx.Identity"(%acc) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %co = "onnx.Identity"(%ci) : (tensor<ui8>) -> tensor<ui8>
      "onnx.Yield"(%co, %step) : (tensor<ui8>, tensor<?x?x?xf32>) -> ()
    }) : (tensor<i64>, none, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %sum = "onnx.ReduceSum"(%r) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
           : (tensor<?x?x?xf32>) -> tensor<f32>
    return %sum : tensor<f32>
  }

  // CHECK-LABEL: func.func @reduce_sum_stays_rank0
  //
  // Outlined body func: declared return is rank-3 (the `onnx.Identity`
  // step value, which the helper leaves at its existing rank-3 type).
  // The `onnx.ReduceSum` inside the body retains its rank-0 result.
  // CHECK-LABEL: func.func private @reduce_sum_stays_rank0_loop_body_n0
  // CHECK-SAME: -> tensor<?x?x?xf32>
  //
  // ReduceSum result stays rank-0 -- the rules library has no rule
  // for it, so the helper skips it (safety belt).
  // CHECK: "onnx.ReduceSum"
  // CHECK-SAME: -> tensor<f32>
}
