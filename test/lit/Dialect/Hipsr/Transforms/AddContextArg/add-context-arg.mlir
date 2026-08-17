// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hipsr-add-context-arg (insert !hipsr.context as arg 0).
//
// Two-phase behavior verified by these tests:
//   Phase 1: Every func.func gains !hipsr.context as its first argument.
//   Phase 2: Every func.call targeting an updated function is rewritten to
//            forward the caller's !hipsr.context.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hipsr-add-context-arg %s | FileCheck %s

// ===== Basic: single function gains !hipsr.context =====
//
// CHECK-LABEL: func.func @single_func
// CHECK-SAME:    (%arg0: !hipsr.context, %arg1: tensor<8x8xf32>)
// CHECK:         return
func.func @single_func(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  return %a : tensor<8x8xf32>
}

// ===== Skip: function already has !hipsr.context =====
//
// CHECK-LABEL: func.func @already_has_context
// CHECK-SAME:    (%arg0: !hipsr.context, %arg1: tensor<8x8xf32>)
// CHECK-NOT:     !hipsr.context, !hipsr.context
// CHECK:         return
func.func @already_has_context(%ctx: !hipsr.context, %a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  return %a : tensor<8x8xf32>
}

// ===== Call site update: caller passes context to callee =====
//
// CHECK-LABEL: func.func @callee_for_call_test
// CHECK-SAME:    (%arg0: !hipsr.context, %arg1: tensor<8x8xf32>)

// CHECK-LABEL: func.func @caller_updates_call
// CHECK-SAME:    (%[[CTX:.*]]: !hipsr.context, %[[A:.*]]: tensor<8x8xf32>)
// CHECK:         call @callee_for_call_test(%[[CTX]], %[[A]])
// CHECK:         return
func.func @callee_for_call_test(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  return %a : tensor<8x8xf32>
}

func.func @caller_updates_call(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %res = func.call @callee_for_call_test(%a) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %res : tensor<8x8xf32>
}
