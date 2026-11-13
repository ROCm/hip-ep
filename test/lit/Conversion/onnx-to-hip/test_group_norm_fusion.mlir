// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that the InstanceNormalization spelling of an NCHW GroupNorm
// (Reshape / InstanceNormalization / Reshape / per-channel Mul and Add) is
// rewritten to one hip.group_norm with channels_last = 0, that a following
// SiLU (Mul(y, Sigmoid(y))) is folded in as activation = 1 only when y feeds
// nothing else, and that a non-identity instance-norm affine is left alone.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    return %arg0 : tensor<1x1xf16>
  }

  // GroupNorm + SiLU, as diffusers' ResNet blocks export it.
  // CHECK-LABEL: func.func @group_norm_silu
  // CHECK-NOT: hip.instance_norm
  // CHECK: hip.group_norm(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x64x8x8xf16>, tensor<64xf16>, tensor<64xf16>)
  // CHECK-SAME: activation = 1 : i64
  // CHECK-SAME: channels_last = 0 : i64
  // CHECK-SAME: groups = 32 : i64
  // CHECK-NOT: hip.sigmoid
  // CHECK: return
  func.func @group_norm_silu(%x: tensor<2x64x8x8xf16>) -> tensor<2x64x8x8xf16> {
    %view = "onnx.Constant"() {value = dense<[0, 32, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %back = "onnx.Constant"() {value = dense<[2, 64, 8, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ones = "onnx.Constant"() {value = dense<1.0> : tensor<32xf16>} : () -> tensor<32xf16>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<32xf16>} : () -> tensor<32xf16>
    %gamma = "onnx.Constant"() {value = dense<1.5> : tensor<64x1x1xf16>} : () -> tensor<64x1x1xf16>
    %beta = "onnx.Constant"() {value = dense<0.25> : tensor<64x1x1xf16>} : () -> tensor<64x1x1xf16>
    %r = "onnx.Reshape"(%x, %view) {allowzero = 0 : si64} : (tensor<2x64x8x8xf16>, tensor<3xi64>) -> tensor<2x32x128xf16>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) {epsilon = 9.99999974E-6 : f32} : (tensor<2x32x128xf16>, tensor<32xf16>, tensor<32xf16>) -> tensor<2x32x128xf16>
    %u = "onnx.Reshape"(%n, %back) {allowzero = 0 : si64} : (tensor<2x32x128xf16>, tensor<4xi64>) -> tensor<2x64x8x8xf16>
    %m = "onnx.Mul"(%u, %gamma) : (tensor<2x64x8x8xf16>, tensor<64x1x1xf16>) -> tensor<2x64x8x8xf16>
    %y = "onnx.Add"(%m, %beta) : (tensor<2x64x8x8xf16>, tensor<64x1x1xf16>) -> tensor<2x64x8x8xf16>
    %s = "onnx.Sigmoid"(%y) : (tensor<2x64x8x8xf16>) -> tensor<2x64x8x8xf16>
    %z = "onnx.Mul"(%y, %s) : (tensor<2x64x8x8xf16>, tensor<2x64x8x8xf16>) -> tensor<2x64x8x8xf16>
    return %z : tensor<2x64x8x8xf16>
  }

  // y also feeds the result directly, so the SiLU stays outside.
  // CHECK-LABEL: func.func @group_norm_shared_output
  // CHECK: hip.group_norm
  // CHECK-SAME: channels_last = 0 : i64
  // CHECK-NOT: activation = 1
  // CHECK: hip.sigmoid
  // CHECK: return
  func.func @group_norm_shared_output(%x: tensor<1x32x4x4xf16>) -> (tensor<1x32x4x4xf16>, tensor<1x32x4x4xf16>) {
    %view = "onnx.Constant"() {value = dense<[1, 8, 64]> : tensor<3xi64>} : () -> tensor<3xi64>
    %back = "onnx.Constant"() {value = dense<[1, 32, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %ones = "onnx.Constant"() {value = dense<1.0> : tensor<8xf16>} : () -> tensor<8xf16>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<8xf16>} : () -> tensor<8xf16>
    %gamma = "onnx.Constant"() {value = dense<2.0> : tensor<1x32x1x1xf16>} : () -> tensor<1x32x1x1xf16>
    %beta = "onnx.Constant"() {value = dense<0.5> : tensor<1x32x1x1xf16>} : () -> tensor<1x32x1x1xf16>
    %r = "onnx.Reshape"(%x, %view) : (tensor<1x32x4x4xf16>, tensor<3xi64>) -> tensor<1x8x64xf16>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) {epsilon = 9.99999974E-6 : f32} : (tensor<1x8x64xf16>, tensor<8xf16>, tensor<8xf16>) -> tensor<1x8x64xf16>
    %u = "onnx.Reshape"(%n, %back) : (tensor<1x8x64xf16>, tensor<4xi64>) -> tensor<1x32x4x4xf16>
    %m = "onnx.Mul"(%gamma, %u) : (tensor<1x32x1x1xf16>, tensor<1x32x4x4xf16>) -> tensor<1x32x4x4xf16>
    %y = "onnx.Add"(%m, %beta) : (tensor<1x32x4x4xf16>, tensor<1x32x1x1xf16>) -> tensor<1x32x4x4xf16>
    %s = "onnx.Sigmoid"(%y) : (tensor<1x32x4x4xf16>) -> tensor<1x32x4x4xf16>
    %z = "onnx.Mul"(%y, %s) : (tensor<1x32x4x4xf16>, tensor<1x32x4x4xf16>) -> tensor<1x32x4x4xf16>
    return %y, %z : tensor<1x32x4x4xf16>, tensor<1x32x4x4xf16>
  }

  // The instance norm carries its own per-group scale: not a plain GroupNorm.
  // CHECK-LABEL: func.func @instance_norm_with_affine
  // CHECK-NOT: hip.group_norm
  // CHECK: hip.instance_norm
  // CHECK: return
  func.func @instance_norm_with_affine(%x: tensor<1x32x4x4xf16>) -> tensor<1x32x4x4xf16> {
    %view = "onnx.Constant"() {value = dense<[1, 8, 64]> : tensor<3xi64>} : () -> tensor<3xi64>
    %back = "onnx.Constant"() {value = dense<[1, 32, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %scale = "onnx.Constant"() {value = dense<3.0> : tensor<8xf16>} : () -> tensor<8xf16>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<8xf16>} : () -> tensor<8xf16>
    %gamma = "onnx.Constant"() {value = dense<2.0> : tensor<32x1x1xf16>} : () -> tensor<32x1x1xf16>
    %beta = "onnx.Constant"() {value = dense<0.5> : tensor<32x1x1xf16>} : () -> tensor<32x1x1xf16>
    %r = "onnx.Reshape"(%x, %view) : (tensor<1x32x4x4xf16>, tensor<3xi64>) -> tensor<1x8x64xf16>
    %n = "onnx.InstanceNormalization"(%r, %scale, %zeros) {epsilon = 9.99999974E-6 : f32} : (tensor<1x8x64xf16>, tensor<8xf16>, tensor<8xf16>) -> tensor<1x8x64xf16>
    %u = "onnx.Reshape"(%n, %back) : (tensor<1x8x64xf16>, tensor<4xi64>) -> tensor<1x32x4x4xf16>
    %m = "onnx.Mul"(%u, %gamma) : (tensor<1x32x4x4xf16>, tensor<32x1x1xf16>) -> tensor<1x32x4x4xf16>
    %y = "onnx.Add"(%m, %beta) : (tensor<1x32x4x4xf16>, tensor<32x1x1xf16>) -> tensor<1x32x4x4xf16>
    return %y : tensor<1x32x4x4xf16>
  }
}
