// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-add-context-arg (insert !hip.context as arg 0).
//
// Two-phase behavior verified by these tests:
//   Phase 1: Every func.func gains !hip.context as its first argument.
//   Phase 2: Every func.call targeting an updated function is rewritten to
//            forward the caller's !hip.context.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-add-context-arg %s | FileCheck %s

// ===== Basic: single function gains !hip.context =====
//
// CHECK-LABEL: func.func @single_func
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<8x8xf32>)
// CHECK:         return
func.func @single_func(%a: memref<8x8xf32>) -> memref<8x8xf32> {
  return %a : memref<8x8xf32>
}

// ===== Skip: function already has !hip.context =====
//
// CHECK-LABEL: func.func @already_has_context
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<8x8xf32>)
// CHECK-NOT:     !hip.context, !hip.context
// CHECK:         return
func.func @already_has_context(%ctx: !hip.context, %a: memref<8x8xf32>) -> memref<8x8xf32> {
  return %a : memref<8x8xf32>
}

// ===== Call site update: caller passes context to callee =====
//
// CHECK-LABEL: func.func @callee_for_call_test
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<8x8xf32>)

// CHECK-LABEL: func.func @caller_updates_call
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[A:.*]]: memref<8x8xf32>)
// CHECK:         call @callee_for_call_test(%[[CTX]], %[[A]])
// CHECK:         return
func.func @callee_for_call_test(%a: memref<8x8xf32>) -> memref<8x8xf32> {
  return %a : memref<8x8xf32>
}

func.func @caller_updates_call(%a: memref<8x8xf32>) -> memref<8x8xf32> {
  %res = func.call @callee_for_call_test(%a) : (memref<8x8xf32>) -> memref<8x8xf32>
  return %res : memref<8x8xf32>
}

// ===== Multi-level calls: A calls B calls C =====
//
// All three gain !hip.context and both call sites forward it.
//
// CHECK-LABEL: func.func @leaf
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<8x8xf32>)
// CHECK:         return

// CHECK-LABEL: func.func @middle
// CHECK-SAME:    (%[[CTX2:.*]]: !hip.context, %[[A2:.*]]: memref<8x8xf32>)
// CHECK:         call @leaf(%[[CTX2]], %[[A2]])
// CHECK:         return

// CHECK-LABEL: func.func @top
// CHECK-SAME:    (%[[CTX3:.*]]: !hip.context, %[[A3:.*]]: memref<8x8xf32>)
// CHECK:         call @middle(%[[CTX3]], %[[A3]])
// CHECK:         return
func.func @leaf(%a: memref<8x8xf32>) -> memref<8x8xf32> {
  return %a : memref<8x8xf32>
}

func.func @middle(%a: memref<8x8xf32>) -> memref<8x8xf32> {
  %res = func.call @leaf(%a) : (memref<8x8xf32>) -> memref<8x8xf32>
  return %res : memref<8x8xf32>
}

func.func @top(%a: memref<8x8xf32>) -> memref<8x8xf32> {
  %res = func.call @middle(%a) : (memref<8x8xf32>) -> memref<8x8xf32>
  return %res : memref<8x8xf32>
}

// ===== Empty module: no functions is a no-op =====
// (This is implicitly tested -- the pass runs on the entire module above
//  which includes all test functions. If there were zero functions, there
//  would be nothing to check.)

// ===== Multiple arguments: context is always first =====
//
// CHECK-LABEL: func.func @many_args
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>, %arg3: index)
func.func @many_args(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %n: index) -> memref<4x4xf32> {
  return %a : memref<4x4xf32>
}

// ===== Call with results: result types are preserved =====
//
// CHECK-LABEL: func.func @callee_with_result
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<4x4xf32>)

// CHECK-LABEL: func.func @caller_preserves_result
// CHECK-SAME:    (%[[CTX4:.*]]: !hip.context, %[[IN:.*]]: memref<4x4xf32>)
// CHECK:         %[[RES:.*]] = call @callee_with_result(%[[CTX4]], %[[IN]])
// CHECK:         return %[[RES]]
func.func @callee_with_result(%a: memref<4x4xf32>) -> memref<4x4xf32> {
  return %a : memref<4x4xf32>
}

func.func @caller_preserves_result(%a: memref<4x4xf32>) -> memref<4x4xf32> {
  %res = func.call @callee_with_result(%a) : (memref<4x4xf32>) -> memref<4x4xf32>
  return %res : memref<4x4xf32>
}

// ===== Mixed: pre-existing-context callee + caller that both already have context =====
//
// When a callee already has !hip.context, the pass skips it in Phase 1.
// Callers that already have !hip.context are also skipped.  The call site
// is already valid (correct arity and types) and must not be rewritten.
//
// Note: a caller *without* !hip.context cannot call a callee that expects
// !hip.context (no way to produce the value), so the "mixed pre-existing
// callee + non-context caller" scenario is impossible with valid input IR.
//
// CHECK-LABEL: func.func @preexisting_callee
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<8x8xf32>)
// CHECK-NOT:     !hip.context, !hip.context
// CHECK:         return

// CHECK-LABEL: func.func @preexisting_caller
// CHECK-SAME:    (%[[CTX5:.*]]: !hip.context, %[[A5:.*]]: memref<8x8xf32>)
// CHECK:         call @preexisting_callee(%[[CTX5]], %[[A5]])
// CHECK:         return
func.func @preexisting_callee(%ctx: !hip.context, %a: memref<8x8xf32>) -> memref<8x8xf32> {
  return %a : memref<8x8xf32>
}

func.func @preexisting_caller(%ctx: !hip.context, %a: memref<8x8xf32>) -> memref<8x8xf32> {
  %res = func.call @preexisting_callee(%ctx, %a) : (!hip.context, memref<8x8xf32>) -> memref<8x8xf32>
  return %res : memref<8x8xf32>
}

// ===== Coexistence: pre-existing-context func alongside updated func =====
//
// @standalone_preexisting already has !hip.context and is not called by anyone.
// @standalone_needs_update does not.  After the pass, both have context and
// the pre-existing function is completely untouched.
//
// CHECK-LABEL: func.func @standalone_preexisting
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<4x4xf32>)
// CHECK-NOT:     !hip.context, !hip.context
// CHECK:         return

// CHECK-LABEL: func.func @standalone_needs_update
// CHECK-SAME:    (%arg0: !hip.context, %arg1: memref<4x4xf32>)
// CHECK:         return
func.func @standalone_preexisting(%ctx: !hip.context, %a: memref<4x4xf32>) -> memref<4x4xf32> {
  return %a : memref<4x4xf32>
}

func.func @standalone_needs_update(%a: memref<4x4xf32>) -> memref<4x4xf32> {
  return %a : memref<4x4xf32>
}
