// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Exercise the simplify-onnx pass in isolation (no other passes, no HIP
// dialect involved). Covers each invariant documented on SimplifyOnnxPass:
//
//   * onnx.CastLike(input, target) -> onnx.Cast(input); type-donor function
//     argument that becomes use-empty is dropped.
//   * Identity CastLike (same input/result element type) is forwarded; both
//     the op and the now-dead type-donor arg are dropped.
//   * Function arguments that were ALREADY dead in the input IR are preserved
//     -- so the "metadata captures the original signature" contract from
//     convert-onnx-to-hip stays intact.
//   * No-CastLike functions are left fully unchanged.
//
// The pass is pure ONNX-dialect, so the RUN line invokes it standalone (no
// --hip-add-context-arg). The end-to-end chain through convert-onnx-to-hip
// is covered separately by test_castlike.mlir.
// ============================================================================

// RUN: hip-mlir-opt --simplify-onnx %s | FileCheck %s

module {
  // -------------------------------------------------------------------------
  // Case 1: basic narrowing. %target is consumed only by onnx.CastLike, so
  // it should be dropped from the function signature.
  // -------------------------------------------------------------------------
  func.func @castlike_drops_type_donor(%input: tensor<3x4xf32>, %target: tensor<0xf16>) -> tensor<3x4xf16> {
    %result = "onnx.CastLike"(%input, %target) : (tensor<3x4xf32>, tensor<0xf16>) -> tensor<3x4xf16>
    return %result : tensor<3x4xf16>
  }

  // CHECK-LABEL: func.func @castlike_drops_type_donor
  // CHECK-SAME: (%[[IN:.*]]: tensor<3x4xf32>) -> tensor<3x4xf16>
  // CHECK-NOT: onnx.CastLike
  // CHECK: %[[CAST:.*]] = "onnx.Cast"(%[[IN]]) : (tensor<3x4xf32>) -> tensor<3x4xf16>
  // CHECK: return %[[CAST]] : tensor<3x4xf16>

  // -------------------------------------------------------------------------
  // Case 2: identity CastLike (input element type already matches result).
  // The op is replaced by direct forwarding, and the type-donor arg goes away.
  // -------------------------------------------------------------------------
  func.func @castlike_identity_short_circuit(%input: tensor<4xf32>, %target: tensor<0xf32>) -> tensor<4xf32> {
    %result = "onnx.CastLike"(%input, %target) : (tensor<4xf32>, tensor<0xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @castlike_identity_short_circuit
  // CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NOT: onnx.CastLike
  // CHECK-NOT: onnx.Cast
  // CHECK: return %[[IN]] : tensor<4xf32>

  // -------------------------------------------------------------------------
  // Case 3: a function argument that was ALREADY dead in the input IR is
  // NOT dropped -- this preserves the "metadata captures the original
  // signature" contract used by convert-onnx-to-hip.
  // -------------------------------------------------------------------------
  func.func @preserve_already_dead_arg(%a: tensor<4xf32>, %unused: tensor<8xi32>) -> tensor<4xf32> {
    return %a : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @preserve_already_dead_arg
  // CHECK-SAME: (%[[A:.*]]: tensor<4xf32>, %[[UNUSED:.*]]: tensor<8xi32>) -> tensor<4xf32>
  // CHECK: return %[[A]] : tensor<4xf32>

  // -------------------------------------------------------------------------
  // Case 4: no CastLike in the body. The pass must be a complete no-op on the
  // function.
  // -------------------------------------------------------------------------
  func.func @no_castlike(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    return %x : tensor<2x2xf32>
  }

  // CHECK-LABEL: func.func @no_castlike
  // CHECK-SAME: (%[[X:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %[[X]] : tensor<2x2xf32>
}
