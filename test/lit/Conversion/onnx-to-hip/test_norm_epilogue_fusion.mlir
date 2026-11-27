// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that a Relu, LeakyRelu or residual Add consuming a single-use
// InstanceNormalization or GroupNorm folds into hip.group_norm's epilogue
// (activation, alpha, residual), and that the fusion is refused when it would
// change semantics.
//
// Positive cases:
// 1. Relu(InstanceNormalization)            -> activation 2, groups = C
// 2. Add(InstanceNormalization, r)          -> residual r
// 3. Add(r, InstanceNormalization)          -> residual r (operand order)
// 4. LeakyRelu(GroupNormalization)          -> activation 3, alpha
// 5. Relu(Add(InstanceNormalization, r))    -> residual r and activation 2
// 6. Relu(com.microsoft GroupNorm, NHWC)    -> activation 2, channels_last 1
//    (the default, so not printed)
//
// Negative cases:
// 7. Norm output with several users         -> no fusion
// 8. Broadcasting residual                  -> no fusion
// 9. Residual of a different type           -> no fusion
// 10. GroupNorm that already applies SiLU   -> no fusion
// 11. Dynamic channel count                 -> no fusion
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: Relu(InstanceNormalization) ---
  func.func @relu_instance_norm(%X: tensor<1x8x16x16xf32>, %S: tensor<8xf32>, %B: tensor<8xf32>) -> tensor<1x8x16x16xf32> {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-03 : f32} : (tensor<1x8x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
    %y = "onnx.Relu"(%n) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    return %y : tensor<1x8x16x16xf32>
  }

  // CHECK-LABEL: func.func @relu_instance_norm
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x8x16x16xf32>, %[[S:.*]]: tensor<8xf32>, %[[B:.*]]: tensor<8xf32>)
  // CHECK-NOT: onnx.
  // CHECK-NOT: hip.max
  // CHECK-NOT: hip.instance_norm
  // CHECK: %[[Y:.*]] = hip.group_norm(%[[CTX]]) ins(%[[X]], %[[S]], %[[B]] :
  // CHECK-NOT: residual(
  // CHECK-SAME: activation = 2 : i64
  // CHECK-SAME: channels_last = 0 : i64
  // CHECK-SAME: epsilon = 1.000000e-03 : f32
  // CHECK-SAME: groups = 8 : i64
  // CHECK-NOT: hip.max
  // CHECK: return %[[Y]]

  // --- Case 2: Add(InstanceNormalization, r) ---
  func.func @add_instance_norm(%X: tensor<1x8x16x16xf32>, %S: tensor<8xf32>, %B: tensor<8xf32>, %R: tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x8x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
    %y = "onnx.Add"(%n, %R) : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    return %y : tensor<1x8x16x16xf32>
  }

  // CHECK-LABEL: func.func @add_instance_norm
  // CHECK-SAME: %[[R:[^:]*]]: tensor<1x8x16x16xf32>)
  // CHECK-NOT: hip.add
  // CHECK: %[[Y:.*]] = hip.group_norm(%{{[^)]*}}) ins(
  // CHECK-SAME: residual(%[[R]] : tensor<1x8x16x16xf32>)
  // CHECK-NOT: activation
  // CHECK-SAME: groups = 8 : i64
  // CHECK-NOT: hip.add
  // CHECK: return %[[Y]]

  // --- Case 3: Add(r, InstanceNormalization) ---
  func.func @add_instance_norm_swapped(%X: tensor<1x8x16x16xf32>, %S: tensor<8xf32>, %B: tensor<8xf32>, %R: tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x8x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
    %y = "onnx.Add"(%R, %n) : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    return %y : tensor<1x8x16x16xf32>
  }

  // CHECK-LABEL: func.func @add_instance_norm_swapped
  // CHECK-SAME: %[[R:[^:]*]]: tensor<1x8x16x16xf32>)
  // CHECK: hip.group_norm(%{{[^)]*}}) ins(
  // CHECK-SAME: residual(%[[R]] : tensor<1x8x16x16xf32>)
  // CHECK-NOT: hip.add

  // --- Case 4: LeakyRelu(GroupNormalization) ---
  func.func @leaky_relu_group_normalization(%X: tensor<2x8x4x4xf16>, %S: tensor<8xf16>, %B: tensor<8xf16>) -> tensor<2x8x4x4xf16> {
    %n = "onnx.GroupNormalization"(%X, %S, %B) {num_groups = 4 : si64, epsilon = 1.000000e-05 : f32} : (tensor<2x8x4x4xf16>, tensor<8xf16>, tensor<8xf16>) -> tensor<2x8x4x4xf16>
    %y = "onnx.LeakyRelu"(%n) {alpha = 2.000000e-01 : f32} : (tensor<2x8x4x4xf16>) -> tensor<2x8x4x4xf16>
    return %y : tensor<2x8x4x4xf16>
  }

  // CHECK-LABEL: func.func @leaky_relu_group_normalization
  // CHECK-NOT: hip.leaky_relu
  // CHECK: %[[Y:.*]] = hip.group_norm(%{{[^)]*}}) ins(
  // CHECK-SAME: activation = 3 : i64
  // CHECK-SAME: alpha = 2.000000e-01 : f32
  // CHECK-SAME: channels_last = 0 : i64
  // CHECK-SAME: groups = 4 : i64
  // CHECK-NOT: hip.leaky_relu
  // CHECK: return %[[Y]]

  // --- Case 5: Relu(Add(InstanceNormalization, r)) ---
  func.func @relu_add_instance_norm(%X: tensor<1x8x16x16xf32>, %S: tensor<8xf32>, %B: tensor<8xf32>, %R: tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x8x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
    %a = "onnx.Add"(%n, %R) : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %y = "onnx.Relu"(%a) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    return %y : tensor<1x8x16x16xf32>
  }

  // CHECK-LABEL: func.func @relu_add_instance_norm
  // CHECK-SAME: %[[R:[^:]*]]: tensor<1x8x16x16xf32>)
  // CHECK-NOT: hip.add
  // CHECK-NOT: hip.max
  // CHECK: %[[Y:.*]] = hip.group_norm(%{{[^)]*}}) ins(
  // CHECK-SAME: residual(%[[R]] : tensor<1x8x16x16xf32>)
  // CHECK-SAME: activation = 2 : i64
  // CHECK-NOT: hip.add
  // CHECK-NOT: hip.max
  // CHECK: return %[[Y]]

  // --- Case 6: Relu(com.microsoft GroupNorm, NHWC) ---
  func.func @relu_ms_group_norm(%X: tensor<1x4x4x8xf16>, %S: tensor<8xf16>, %B: tensor<8xf16>) -> tensor<1x4x4x8xf16> {
    %n = "onnx.Custom"(%X, %S, %B) {domain_name = "com.microsoft", function_name = "GroupNorm", groups = 2 : si64, activation = 0 : si64, channels_last = 1 : si64, epsilon = 1.000000e-05 : f32} : (tensor<1x4x4x8xf16>, tensor<8xf16>, tensor<8xf16>) -> tensor<1x4x4x8xf16>
    %y = "onnx.Relu"(%n) : (tensor<1x4x4x8xf16>) -> tensor<1x4x4x8xf16>
    return %y : tensor<1x4x4x8xf16>
  }

  // CHECK-LABEL: func.func @relu_ms_group_norm
  // CHECK-NOT: hip.max
  // CHECK: hip.group_norm(%{{[^)]*}}) ins(
  // CHECK-SAME: activation = 2 : i64
  // CHECK-NOT: channels_last
  // CHECK-SAME: groups = 2 : i64
  // CHECK-NOT: hip.max

  // --- Case 7: norm output with several users ---
  func.func @multi_use_norm(%X: tensor<1x8x16x16xf32>, %S: tensor<8xf32>, %B: tensor<8xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x8x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
    %y = "onnx.Relu"(%n) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    return %y, %n : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
  }

  // CHECK-LABEL: func.func @multi_use_norm
  // CHECK-NOT: hip.group_norm
  // CHECK: hip.instance_norm
  // CHECK: hip.max
  // CHECK-NOT: hip.group_norm
  // CHECK: return

  // --- Case 8: broadcasting residual ---
  func.func @broadcast_residual(%X: tensor<1x8x16x16xf32>, %S: tensor<8xf32>, %B: tensor<8xf32>, %R: tensor<1x8x1x1xf32>) -> tensor<1x8x16x16xf32> {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x8x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
    %y = "onnx.Add"(%n, %R) : (tensor<1x8x16x16xf32>, tensor<1x8x1x1xf32>) -> tensor<1x8x16x16xf32>
    return %y : tensor<1x8x16x16xf32>
  }

  // CHECK-LABEL: func.func @broadcast_residual
  // CHECK-NOT: hip.group_norm
  // CHECK: hip.instance_norm
  // CHECK-NOT: hip.group_norm
  // CHECK: return

  // --- Case 9: residual of a different type ---
  func.func @residual_type_mismatch(%X: tensor<1x8x16x16xf32>, %S: tensor<8xf32>, %B: tensor<8xf32>, %R: tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32> {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x8x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
    %y = "onnx.Add"(%n, %R) : (tensor<1x8x16x16xf32>, tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32>
    return %y : tensor<?x8x16x16xf32>
  }

  // CHECK-LABEL: func.func @residual_type_mismatch
  // CHECK-NOT: hip.group_norm
  // CHECK: hip.instance_norm
  // CHECK-NOT: hip.group_norm
  // CHECK: return

  // --- Case 10: GroupNorm that already applies SiLU ---
  func.func @silu_group_norm(%X: tensor<1x4x4x8xf16>, %S: tensor<8xf16>, %B: tensor<8xf16>) -> tensor<1x4x4x8xf16> {
    %n = "onnx.Custom"(%X, %S, %B) {domain_name = "com.microsoft", function_name = "GroupNorm", groups = 2 : si64, activation = 1 : si64, channels_last = 1 : si64, epsilon = 1.000000e-05 : f32} : (tensor<1x4x4x8xf16>, tensor<8xf16>, tensor<8xf16>) -> tensor<1x4x4x8xf16>
    %y = "onnx.Relu"(%n) : (tensor<1x4x4x8xf16>) -> tensor<1x4x4x8xf16>
    return %y : tensor<1x4x4x8xf16>
  }

  // CHECK-LABEL: func.func @silu_group_norm
  // CHECK: hip.group_norm
  // CHECK-SAME: activation = 1 : i64
  // CHECK: hip.max

  // --- Case 11: dynamic channel count ---
  func.func @dynamic_channels(%X: tensor<1x?x16x16xf32>, %S: tensor<?xf32>, %B: tensor<?xf32>) -> tensor<1x?x16x16xf32> {
    %n = "onnx.InstanceNormalization"(%X, %S, %B) {epsilon = 1.000000e-05 : f32} : (tensor<1x?x16x16xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<1x?x16x16xf32>
    %y = "onnx.Relu"(%n) : (tensor<1x?x16x16xf32>) -> tensor<1x?x16x16xf32>
    return %y : tensor<1x?x16x16xf32>
  }

  // CHECK-LABEL: func.func @dynamic_channels
  // CHECK-NOT: hip.group_norm
  // CHECK: hip.instance_norm
  // CHECK: hip.max
}
