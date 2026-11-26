// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Fold the exported RMSNorm primitive chain into hip.rms_norm via
// onnx.RMSNormalization.
//
// 1. SD3.5-style stash form: fp16->fp32 Casts around Pow/Mul/Div, then Cast
//    back before the scale Mul.
// 2. Plain fp16 Mul(x,x) + Reciprocal form (no Casts).
// 3. LayerNorm-style Pow(Sub(...)) must stay unfused.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // SD3.5 stash form (fp16 activations, fp32 math, fp16 scale).
  func.func @rms_norm_sd35_stash(%x: tensor<2x4x8xf16>, %scale: tensor<8xf16>)
      -> tensor<2x4x8xf16> {
    %two = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
    %eps = "onnx.Constant"() {value = dense<1.0e-06> : tensor<f32>} : () -> tensor<f32>
    %one = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %x0 = "onnx.Cast"(%x) {to = f32} : (tensor<2x4x8xf16>) -> tensor<2x4x8xf32>
    %sq = "onnx.Pow"(%x0, %two) : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4x8xf32>
    %mean = "onnx.ReduceMean"(%sq, %axes) {keepdims = 1 : si64} : (tensor<2x4x8xf32>, tensor<i64>) -> tensor<2x4x1xf32>
    %added = "onnx.Add"(%mean, %eps) : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4x1xf32>
    %rms = "onnx.Sqrt"(%added) : (tensor<2x4x1xf32>) -> tensor<2x4x1xf32>
    %inv = "onnx.Div"(%one, %rms) : (tensor<f32>, tensor<2x4x1xf32>) -> tensor<2x4x1xf32>
    %x1 = "onnx.Cast"(%x) {to = f32} : (tensor<2x4x8xf16>) -> tensor<2x4x8xf32>
    %norm = "onnx.Mul"(%x1, %inv) : (tensor<2x4x8xf32>, tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
    %norm_f16 = "onnx.Cast"(%norm) {to = f16} : (tensor<2x4x8xf32>) -> tensor<2x4x8xf16>
    %y = "onnx.Mul"(%norm_f16, %scale) : (tensor<2x4x8xf16>, tensor<8xf16>) -> tensor<2x4x8xf16>
    return %y : tensor<2x4x8xf16>
  }

  // CHECK-LABEL: func.func @rms_norm_sd35_stash
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x4x8xf16>, %[[SCALE:.*]]: tensor<8xf16>)
  // CHECK-NOT: hip.pow
  // CHECK-NOT: hip.reduce_mean
  // CHECK-NOT: hip.sqrt
  // CHECK: %[[Y:.*]] = hip.rms_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<2x4x8xf16>, tensor<8xf16>)
  // CHECK-SAME: axis = -1
  // CHECK-SAME: epsilon = 1.000000e-06
  // CHECK: return %[[Y]]

  // Plain fp16: Mul(x,x) + Reciprocal.
  func.func @rms_norm_mul_reciprocal(%x: tensor<2x8xf16>, %scale: tensor<8xf16>)
      -> tensor<2x8xf16> {
    %eps = "onnx.Constant"() {value = dense<1.0e-05> : tensor<f16>} : () -> tensor<f16>
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %sq = "onnx.Mul"(%x, %x) : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
    %mean = "onnx.ReduceMean"(%sq, %axes) {keepdims = 1 : si64} : (tensor<2x8xf16>, tensor<i64>) -> tensor<2x1xf16>
    %added = "onnx.Add"(%mean, %eps) : (tensor<2x1xf16>, tensor<f16>) -> tensor<2x1xf16>
    %rms = "onnx.Sqrt"(%added) : (tensor<2x1xf16>) -> tensor<2x1xf16>
    %inv = "onnx.Reciprocal"(%rms) : (tensor<2x1xf16>) -> tensor<2x1xf16>
    %norm = "onnx.Mul"(%x, %inv) : (tensor<2x8xf16>, tensor<2x1xf16>) -> tensor<2x8xf16>
    %y = "onnx.Mul"(%norm, %scale) : (tensor<2x8xf16>, tensor<8xf16>) -> tensor<2x8xf16>
    return %y : tensor<2x8xf16>
  }

  // CHECK-LABEL: func.func @rms_norm_mul_reciprocal
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x8xf16>, %[[SCALE:.*]]: tensor<8xf16>)
  // CHECK: %[[Y:.*]] = hip.rms_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<2x8xf16>, tensor<8xf16>)
  // CHECK-SAME: axis = -1
  // CHECK: return %[[Y]]

  // Centered LayerNorm variance fragment: must NOT fuse to rms_norm.
  func.func @layernorm_centered_rejected(%x: tensor<2x8xf32>, %scale: tensor<8xf32>)
      -> tensor<2x8xf32> {
    %two = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
    %eps = "onnx.Constant"() {value = dense<1.0e-05> : tensor<f32>} : () -> tensor<f32>
    %one = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %mean_x = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64} : (tensor<2x8xf32>, tensor<i64>) -> tensor<2x1xf32>
    %delta = "onnx.Sub"(%x, %mean_x) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Pow"(%delta, %two) : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq, %axes) {keepdims = 1 : si64} : (tensor<2x8xf32>, tensor<i64>) -> tensor<2x1xf32>
    %added = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %std = "onnx.Sqrt"(%added) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %inv = "onnx.Div"(%one, %std) : (tensor<f32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    %norm = "onnx.Mul"(%delta, %inv) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%norm, %scale) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @layernorm_centered_rejected
  // CHECK-NOT: hip.rms_norm
}
