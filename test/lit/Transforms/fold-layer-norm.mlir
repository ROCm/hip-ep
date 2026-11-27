// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-fold-layer-norm folds a primitive LayerNorm chain over the last axis,
// as convert-onnx-to-hip lowers it to HIP ops, into one hip.layer_norm (fp32
// accumulation): Pow(d, 2) or Mul(d, d) for the square; Div by Sqrt /
// Pow(., 0.5) or Mul by Reciprocal(Sqrt) / Pow(., -0.5) / Div(1, Sqrt) for the
// normalization; with or without the affine tail; ranks above 4 through their
// collapse/expand pack. A residual Add feeding the norm folds into
// hip.skip_layer_norm when its operands' runtime extents are proven equal.
// Chains over a non-last axis, with a non-constant epsilon, that regroup
// rows, or in bf16 are left alone.
//
// The second RUN line follows with the rocMLIR fusion: the pass runs in the
// shared head before it, so the chain and its residual are folded before the
// fusion can outline them.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --hip-fold-layer-norm %s | FileCheck %s
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --hip-fold-layer-norm --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s | FileCheck %s --check-prefix=RMT

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @ln_pow(%x: tensor<1x16x768xf16>, %g: tensor<768xf16>, %b: tensor<768xf16>) -> tensor<1x16x768xf16> {
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %two = "onnx.Constant"() {value = dense<2.0> : tensor<f16>} : () -> tensor<f16>
    %eps = "onnx.Constant"() {value = dense<9.99999997E-7> : tensor<f16>} : () -> tensor<f16>
    %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64} : (tensor<1x16x768xf16>, tensor<1xi64>) -> tensor<1x16x1xf16>
    %d = "onnx.Sub"(%x, %mean) : (tensor<1x16x768xf16>, tensor<1x16x1xf16>) -> tensor<1x16x768xf16>
    %sq = "onnx.Pow"(%d, %two) : (tensor<1x16x768xf16>, tensor<f16>) -> tensor<1x16x768xf16>
    %var = "onnx.ReduceMean"(%sq, %axes) {keepdims = 1 : si64} : (tensor<1x16x768xf16>, tensor<1xi64>) -> tensor<1x16x1xf16>
    %ve = "onnx.Add"(%var, %eps) : (tensor<1x16x1xf16>, tensor<f16>) -> tensor<1x16x1xf16>
    %s = "onnx.Sqrt"(%ve) : (tensor<1x16x1xf16>) -> tensor<1x16x1xf16>
    %n = "onnx.Div"(%d, %s) : (tensor<1x16x768xf16>, tensor<1x16x1xf16>) -> tensor<1x16x768xf16>
    %m = "onnx.Mul"(%n, %g) : (tensor<1x16x768xf16>, tensor<768xf16>) -> tensor<1x16x768xf16>
    %y = "onnx.Add"(%m, %b) : (tensor<1x16x768xf16>, tensor<768xf16>) -> tensor<1x16x768xf16>
    return %y : tensor<1x16x768xf16>
  }

  // CHECK-LABEL: func.func @ln_pow
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x16x768xf16>, %[[G:.*]]: tensor<768xf16>, %[[B:.*]]: tensor<768xf16>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[G]], %[[B]] :
  // CHECK-NOT: hip.sqrt
  // CHECK: return %[[Y]]

  // RMT-LABEL: func.func @ln_pow
  // RMT-NOT: hip.reduce_mean
  // RMT: hip.layer_norm
  // RMT-NOT: hip.rocmlir
  // RMT: return



  func.func @ln_mul_no_bias(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%g, %n) : (tensor<8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ln_mul_no_bias
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x8xf32>, %[[G:.*]]: tensor<8xf32>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[G]] :
  // CHECK: return %[[Y]]

  func.func @not_last_axis(%x: tensor<8x2xf32>, %g: tensor<2xf32>) -> tensor<8x2xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [0], keepdims = 1 : si64} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [0], keepdims = 1 : si64} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<1x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<8x2xf32>, tensor<2xf32>) -> tensor<8x2xf32>
    return %y : tensor<8x2xf32>
  }

  // CHECK-LABEL: func.func @not_last_axis
  // CHECK-NOT: hip.layer_norm
  // CHECK: hip.reduce_mean
  // CHECK: return

  // An f32 epsilon of 1e-8 cast to f16 is 0 there: the fused op takes the
  // value the cast produces, not the f32 source.
  func.func @cast_epsilon_rounds(%x: tensor<2x8xf16>, %g: tensor<8xf16>) -> tensor<2x8xf16> {
    %eps32 = "onnx.Constant"() {value = dense<1.0E-8> : tensor<f32>} : () -> tensor<f32>
    %eps = "onnx.Cast"(%eps32) {saturate = 1 : si64, to = f16} : (tensor<f32>) -> tensor<f16>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf16>) -> tensor<2x1xf16>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf16>, tensor<2x1xf16>) -> tensor<2x8xf16>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf16>) -> tensor<2x1xf16>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf16>, tensor<f16>) -> tensor<2x1xf16>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf16>) -> tensor<2x1xf16>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf16>, tensor<2x1xf16>) -> tensor<2x8xf16>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf16>, tensor<8xf16>) -> tensor<2x8xf16>
    return %y : tensor<2x8xf16>
  }

  // CHECK-LABEL: func.func @cast_epsilon_rounds
  // CHECK: hip.layer_norm
  // CHECK-SAME: epsilon = 0.000000e+00 : f32

  // A cast through bool changes the value (1e-5 -> true -> 1.0), so the
  // chain is not treated as LayerNorm with that epsilon.
  func.func @cast_through_bool_rejected(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %eps32 = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %flag = "onnx.Cast"(%eps32) {saturate = 1 : si64, to = i1} : (tensor<f32>) -> tensor<i1>
    %eps = "onnx.Cast"(%flag) {saturate = 1 : si64, to = f32} : (tensor<i1>) -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @cast_through_bool_rejected
  // CHECK-NOT: hip.layer_norm
  // CHECK: hip.reduce_mean
  // CHECK: return

  // A ui8 epsilon constant zero-extends when cast to f32.
  func.func @ui8_epsilon_zero_extends(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %c = "onnx.Constant"() {value = dense<255> : tensor<ui8>} : () -> tensor<ui8>
    %eps = "onnx.Cast"(%c) {saturate = 1 : si64, to = f32} : (tensor<ui8>) -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ui8_epsilon_zero_extends
  // CHECK: hip.layer_norm
  // CHECK-SAME: epsilon = 2.550000e+02 : f32

  // A i1 epsilon constant zero-extends when cast to f32.
  func.func @i1_epsilon_zero_extends(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %eps = "onnx.Cast"(%c) {saturate = 1 : si64, to = f32} : (tensor<i1>) -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @i1_epsilon_zero_extends
  // CHECK: hip.layer_norm
  // CHECK-SAME: epsilon = 1.000000e+00 : f32

  // d * Reciprocal(Sqrt(var + eps)), then the affine tail.
  func.func @ln_reciprocal_sqrt(%x: tensor<2x8xf16>, %g: tensor<8xf16>, %b: tensor<8xf16>) -> tensor<2x8xf16> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f16>} : () -> tensor<f16>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf16>) -> tensor<2x1xf16>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf16>, tensor<2x1xf16>) -> tensor<2x8xf16>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf16>) -> tensor<2x1xf16>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf16>, tensor<f16>) -> tensor<2x1xf16>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf16>) -> tensor<2x1xf16>
    %r = "onnx.Reciprocal"(%s) : (tensor<2x1xf16>) -> tensor<2x1xf16>
    %n = "onnx.Mul"(%d, %r) : (tensor<2x8xf16>, tensor<2x1xf16>) -> tensor<2x8xf16>
    %m = "onnx.Mul"(%n, %g) : (tensor<2x8xf16>, tensor<8xf16>) -> tensor<2x8xf16>
    %y = "onnx.Add"(%m, %b) : (tensor<2x8xf16>, tensor<8xf16>) -> tensor<2x8xf16>
    return %y : tensor<2x8xf16>
  }

  // CHECK-LABEL: func.func @ln_reciprocal_sqrt
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x8xf16>, %[[G:.*]]: tensor<8xf16>, %[[B:.*]]: tensor<8xf16>)
  // CHECK-NOT: hip.reciprocal
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[G]], %[[B]] :
  // CHECK-NOT: hip.reciprocal
  // CHECK: return %[[Y]]

  // Pow(var + eps, -0.5) * d, inverse std on the left.
  func.func @ln_pow_neg_half(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %p = "onnx.Constant"() {value = dense<-5.0E-1> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %r = "onnx.Pow"(%ve, %p) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %n = "onnx.Mul"(%r, %d) : (tensor<2x1xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ln_pow_neg_half
  // CHECK-NOT: hip.reduce_mean
  // CHECK: hip.layer_norm
  // CHECK-NOT: hip.sqrt
  // CHECK: return

  // d * Div(1, Sqrt(var + eps)).
  func.func @ln_div_one_sqrt(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %one = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %r = "onnx.Div"(%one, %s) : (tensor<f32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Mul"(%d, %r) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ln_div_one_sqrt
  // CHECK-NOT: hip.reduce_mean
  // CHECK: hip.layer_norm
  // CHECK-NOT: hip.div
  // CHECK: return

  // d / Pow(var + eps, 0.5).
  func.func @ln_div_pow_half(%x: tensor<2x8xf32>, %g: tensor<8xf32>) -> tensor<2x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %h = "onnx.Constant"() {value = dense<5.0E-1> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Pow"(%ve, %h) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ln_div_pow_half
  // CHECK-NOT: hip.reduce_mean
  // CHECK: hip.layer_norm
  // CHECK-NOT: hip.sqrt
  // CHECK: return

  // No affine tail: the norm gets a constant scale of ones.
  func.func @ln_no_affine(%x: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    return %n : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ln_no_affine
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x8xf32>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %{{.*}} : tensor<2x8xf32>, tensor<8xf32>)
  // CHECK: return %[[Y]]

  // A scalar Mul after the norm is not a per-channel scale: the norm takes
  // ones and the Mul stays.
  func.func @ln_scalar_mul_kept(%x: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %k = "onnx.Constant"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %k) : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @ln_scalar_mul_kept
  // CHECK: %[[N:.*]] = hip.layer_norm
  // CHECK: hip.mul(%{{.*}}) ins(%[[N]],
  // CHECK: return

  // A non-constant epsilon is not folded.
  func.func @dynamic_epsilon_rejected(%x: tensor<2x8xf32>, %g: tensor<8xf32>, %eps: tensor<f32>) -> tensor<2x8xf32> {
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x8xf32>) -> tensor<2x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %r = "onnx.Reciprocal"(%s) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %n = "onnx.Mul"(%d, %r) : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @dynamic_epsilon_rejected
  // CHECK-NOT: hip.layer_norm
  // CHECK: hip.reduce_mean
  // CHECK: return

  // Mul(d, Reciprocal(Sqrt)) over a non-last axis is not folded.
  func.func @mul_form_not_last_axis(%x: tensor<8x2xf32>) -> tensor<8x2xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [0], keepdims = 1 : si64} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [0], keepdims = 1 : si64} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<1x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %r = "onnx.Reciprocal"(%s) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %n = "onnx.Mul"(%d, %r) : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
    return %n : tensor<8x2xf32>
  }

  // CHECK-LABEL: func.func @mul_form_not_last_axis
  // CHECK-NOT: hip.layer_norm
  // CHECK: hip.reduce_mean
  // CHECK: return

  // A residual Add feeding a decomposed norm folds into hip.skip_layer_norm;
  // the later residual consumer reads the sum the fused op writes.
  func.func @residual_add_folds(%a: tensor<1x4x8xf16>, %b: tensor<1x4x8xf16>, %c: tensor<1x4x8xf16>, %g: tensor<8xf16>, %be: tensor<8xf16>) -> (tensor<1x4x8xf16>, tensor<1x4x8xf16>) {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f16>} : () -> tensor<f16>
    %x = "onnx.Add"(%a, %b) : (tensor<1x4x8xf16>, tensor<1x4x8xf16>) -> tensor<1x4x8xf16>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x4x8xf16>) -> tensor<1x4x1xf16>
    %d = "onnx.Sub"(%x, %mean) : (tensor<1x4x8xf16>, tensor<1x4x1xf16>) -> tensor<1x4x8xf16>
    %sq = "onnx.Mul"(%d, %d) : (tensor<1x4x8xf16>, tensor<1x4x8xf16>) -> tensor<1x4x8xf16>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<1x4x8xf16>) -> tensor<1x4x1xf16>
    %ve = "onnx.Add"(%var, %eps) : (tensor<1x4x1xf16>, tensor<f16>) -> tensor<1x4x1xf16>
    %s = "onnx.Sqrt"(%ve) : (tensor<1x4x1xf16>) -> tensor<1x4x1xf16>
    %n = "onnx.Div"(%d, %s) : (tensor<1x4x8xf16>, tensor<1x4x1xf16>) -> tensor<1x4x8xf16>
    %m = "onnx.Mul"(%n, %g) : (tensor<1x4x8xf16>, tensor<8xf16>) -> tensor<1x4x8xf16>
    %y = "onnx.Add"(%m, %be) : (tensor<1x4x8xf16>, tensor<8xf16>) -> tensor<1x4x8xf16>
    %next = "onnx.Add"(%x, %c) : (tensor<1x4x8xf16>, tensor<1x4x8xf16>) -> tensor<1x4x8xf16>
    return %y, %next : tensor<1x4x8xf16>, tensor<1x4x8xf16>
  }

  // CHECK-LABEL: func.func @residual_add_folds
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<1x4x8xf16>, %[[B:.*]]: tensor<1x4x8xf16>, %[[C:.*]]: tensor<1x4x8xf16>, %[[G:.*]]: tensor<8xf16>, %[[BE:.*]]: tensor<8xf16>)
  // CHECK-NOT: hip.layer_norm
  // CHECK: %[[R:.*]]:2 = hip.skip_layer_norm(%[[CTX]]) ins(%[[A]], %[[B]], %[[G]] : tensor<1x4x8xf16>, tensor<1x4x8xf16>, tensor<8xf16>, beta %[[BE]] : tensor<8xf16>)
  // CHECK-SAME: round_sum
  // CHECK: %[[NEXT:.*]] = hip.add(%[[CTX]]) ins(%[[R]]#1, %[[C]] :
  // CHECK: return %[[R]]#0, %[[NEXT]]

  // RMT-LABEL: func.func @residual_add_folds
  // RMT: hip.skip_layer_norm
  // RMT: return



  // onnx.LayerNormalization of a residual Add used only by the norm: no sum
  // buffer.
  func.func @residual_layer_normalization(%a: tensor<2x16xf32>, %b: tensor<2x16xf32>, %g: tensor<16xf32>) -> tensor<2x16xf32> {
    %x = "onnx.Add"(%a, %b) : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
    %y = "onnx.LayerNormalization"(%x, %g) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32} : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
    return %y : tensor<2x16xf32>
  }

  // CHECK-LABEL: func.func @residual_layer_normalization
  // CHECK-NOT: hip.add
  // CHECK: %[[Y:.*]] = hip.skip_layer_norm(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x16xf32>, tensor<2x16xf32>, tensor<16xf32>) outs(%{{[^,]*}} : tensor<2x16xf32>)
  // CHECK: return %[[Y]]

  // A broadcasting Add (bias add) is not a residual: no fold.
  func.func @broadcast_add_not_folded(%a: tensor<2x16xf32>, %bias: tensor<16xf32>, %g: tensor<16xf32>) -> tensor<2x16xf32> {
    %x = "onnx.Add"(%a, %bias) : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
    %y = "onnx.LayerNormalization"(%x, %g) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32} : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
    return %y : tensor<2x16xf32>
  }

  // CHECK-LABEL: func.func @broadcast_add_not_folded
  // CHECK-NOT: hip.skip_layer_norm
  // CHECK: hip.add
  // CHECK: hip.layer_norm
  // CHECK: return

  // The sum is also used before the norm, where the fused op cannot reach.
  func.func @sum_used_before_norm(%a: tensor<2x16xf32>, %b: tensor<2x16xf32>, %g: tensor<16xf32>) -> (tensor<2x16xf32>, tensor<2x16xf32>) {
    %x = "onnx.Add"(%a, %b) : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
    %u = "onnx.Mul"(%x, %x) : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
    %y = "onnx.LayerNormalization"(%x, %g) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32} : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
    return %y, %u : tensor<2x16xf32>, tensor<2x16xf32>
  }

  // CHECK-LABEL: func.func @sum_used_before_norm
  // CHECK-NOT: hip.skip_layer_norm
  // CHECK: hip.layer_norm
  // CHECK: return

  // Residual fold needs the norm over the last axis.
  func.func @residual_not_last_axis(%a: tensor<2x16xf32>, %b: tensor<2x16xf32>, %g: tensor<2x16xf32>) -> tensor<2x16xf32> {
    %x = "onnx.Add"(%a, %b) : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
    %y = "onnx.LayerNormalization"(%x, %g) {axis = 0 : si64, epsilon = 9.99999974E-6 : f32} : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
    return %y : tensor<2x16xf32>
  }

  // CHECK-LABEL: func.func @residual_not_last_axis
  // CHECK-NOT: hip.skip_layer_norm
  // CHECK: hip.layer_norm
  // CHECK: return

  // Rank 5 is packed to rank 4 before fusion. The match looks through one
  // collapse/expand pair and still produces one norm.
  func.func @ln_rank5(%x: tensor<2x3x4x5x8xf32>, %g: tensor<8xf32>) -> tensor<2x3x4x5x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<2x3x4x5x8xf32>) -> tensor<2x3x4x5x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x3x4x5x8xf32>, tensor<2x3x4x5x1xf32>) -> tensor<2x3x4x5x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x3x4x5x8xf32>, tensor<2x3x4x5x8xf32>) -> tensor<2x3x4x5x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<2x3x4x5x8xf32>) -> tensor<2x3x4x5x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x3x4x5x1xf32>, tensor<f32>) -> tensor<2x3x4x5x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x3x4x5x1xf32>) -> tensor<2x3x4x5x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<2x3x4x5x8xf32>, tensor<2x3x4x5x1xf32>) -> tensor<2x3x4x5x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x3x4x5x8xf32>, tensor<8xf32>) -> tensor<2x3x4x5x8xf32>
    return %y : tensor<2x3x4x5x8xf32>
  }

  // CHECK-LABEL: func.func @ln_rank5
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3x4x5x8xf32>, %[[G:.*]]: tensor<8xf32>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[G]] :
  // CHECK: return %[[Y]]


  // A packed chain ends in an expand, so the norm gets a new destination;
  // it carries x's runtime batch.
  func.func @ln_rank5_dynamic(%x: tensor<?x3x4x5x8xf32>, %g: tensor<8xf32>) -> tensor<?x3x4x5x8xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<?x3x4x5x8xf32>) -> tensor<?x3x4x5x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<?x3x4x5x8xf32>, tensor<?x3x4x5x1xf32>) -> tensor<?x3x4x5x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<?x3x4x5x8xf32>, tensor<?x3x4x5x8xf32>) -> tensor<?x3x4x5x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<?x3x4x5x8xf32>) -> tensor<?x3x4x5x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<?x3x4x5x1xf32>, tensor<f32>) -> tensor<?x3x4x5x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<?x3x4x5x1xf32>) -> tensor<?x3x4x5x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<?x3x4x5x8xf32>, tensor<?x3x4x5x1xf32>) -> tensor<?x3x4x5x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<?x3x4x5x8xf32>, tensor<8xf32>) -> tensor<?x3x4x5x8xf32>
    return %y : tensor<?x3x4x5x8xf32>
  }

  // CHECK-LABEL: func.func @ln_rank5_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x3x4x5x8xf32>, %[[G:.*]]: tensor<8xf32>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[D:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x3x4x5x8xf32>
  // CHECK: %[[E:.*]] = tensor.empty(%[[D]]) : tensor<?x3x4x5x8xf32>
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]]) ins(%[[X]], %[[G]] : {{.*}}) outs(%[[E]] :
  // CHECK: return %[[Y]]

  // The squares are reshaped to [1, 1, 4] before the variance, so both rows
  // share one variance. That is not a LayerNorm over x's rows: not fused.
  func.func @reshape_regroups_rows(%x: tensor<1x2x2xf32>) -> tensor<1x2x2xf32> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %flat = "onnx.Constant"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
    %row = "onnx.Constant"() {value = dense<[1, 1, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
    %mean = "onnx.ReduceMean"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x2x2xf32>) -> tensor<1x2x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<1x2x2xf32>, tensor<1x2x1xf32>) -> tensor<1x2x2xf32>
    %df = "onnx.Reshape"(%d, %flat) : (tensor<1x2x2xf32>, tensor<2xi64>) -> tensor<1x4xf32>
    %sqf = "onnx.Mul"(%df, %df) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %sq = "onnx.Reshape"(%sqf, %row) : (tensor<1x4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [-1], keepdims = 1 : si64} : (tensor<1x1x4xf32>) -> tensor<1x1x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<1x1x1xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<1x2x2xf32>, tensor<1x1x1xf32>) -> tensor<1x2x2xf32>
    return %n : tensor<1x2x2xf32>
  }

  // CHECK-LABEL: func.func @reshape_regroups_rows
  // CHECK-NOT: hip.layer_norm
  // CHECK: hip.reduce_mean
  // CHECK: tensor.expand_shape {{.*}} into tensor<1x1x4xf32>
  // CHECK: hip.reciprocal
  // CHECK: return

  // The runtime LayerNorm reads any 2-byte element as fp16: bf16 is not fused.
  func.func @bf16_chain_kept(%x: tensor<2x8xbf16>, %g: tensor<8xbf16>) -> tensor<2x8xbf16> {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<bf16>} : () -> tensor<bf16>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xbf16>) -> tensor<2x1xbf16>
    %d = "onnx.Sub"(%x, %mean) : (tensor<2x8xbf16>, tensor<2x1xbf16>) -> tensor<2x8xbf16>
    %sq = "onnx.Mul"(%d, %d) : (tensor<2x8xbf16>, tensor<2x8xbf16>) -> tensor<2x8xbf16>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<2x8xbf16>) -> tensor<2x1xbf16>
    %ve = "onnx.Add"(%var, %eps) : (tensor<2x1xbf16>, tensor<bf16>) -> tensor<2x1xbf16>
    %s = "onnx.Sqrt"(%ve) : (tensor<2x1xbf16>) -> tensor<2x1xbf16>
    %n = "onnx.Div"(%d, %s) : (tensor<2x8xbf16>, tensor<2x1xbf16>) -> tensor<2x8xbf16>
    %y = "onnx.Mul"(%n, %g) : (tensor<2x8xbf16>, tensor<8xbf16>) -> tensor<2x8xbf16>
    return %y : tensor<2x8xbf16>
  }

  // CHECK-LABEL: func.func @bf16_chain_kept
  // CHECK-NOT: hip.layer_norm
  // CHECK: hip.reduce_mean
  // CHECK: hip.sqrt
  // CHECK: return


  // Two independent ?x16 arguments may hold [2, 16] and [1, 16]; hip.add
  // broadcasts, the fused kernel would not, so the add stays.
  func.func @dynamic_residual_unproven(%a: tensor<?x16xf32>, %b: tensor<?x16xf32>, %g: tensor<16xf32>) -> tensor<?x16xf32> {
    %x = "onnx.Add"(%a, %b) : (tensor<?x16xf32>, tensor<?x16xf32>) -> tensor<?x16xf32>
    %y = "onnx.LayerNormalization"(%x, %g) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32} : (tensor<?x16xf32>, tensor<16xf32>) -> tensor<?x16xf32>
    return %y : tensor<?x16xf32>
  }

  // CHECK-LABEL: func.func @dynamic_residual_unproven
  // CHECK-NOT: hip.skip_layer_norm
  // CHECK: hip.add
  // CHECK: hip.layer_norm
  // CHECK: return

  // The second operand is sized from the first, so their extents are equal.
  func.func @dynamic_residual_proven(%a: tensor<?x16xf32>, %g: tensor<16xf32>) -> tensor<?x16xf32> {
    %b = "onnx.Mul"(%a, %a) : (tensor<?x16xf32>, tensor<?x16xf32>) -> tensor<?x16xf32>
    %x = "onnx.Add"(%a, %b) : (tensor<?x16xf32>, tensor<?x16xf32>) -> tensor<?x16xf32>
    %y = "onnx.LayerNormalization"(%x, %g) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32} : (tensor<?x16xf32>, tensor<16xf32>) -> tensor<?x16xf32>
    return %y : tensor<?x16xf32>
  }

  // CHECK-LABEL: func.func @dynamic_residual_proven
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<?x16xf32>, %[[G:.*]]: tensor<16xf32>)
  // CHECK: %[[B:.*]] = hip.mul(%[[CTX]]) ins(%[[A]], %[[A]] :
  // CHECK-NOT: hip.add
  // CHECK: %[[Y:.*]] = hip.skip_layer_norm(%[[CTX]]) ins(%[[A]], %[[B]], %[[G]] :
  // CHECK: return %[[Y]]

  // A shape query on the mean survives the fusion and reads x's batch.
  func.func @mean_shape_used(%x: tensor<?x8xf32>, %g: tensor<8xf32>) -> (tensor<?x8xf32>, tensor<2xi64>) {
    %eps = "onnx.Constant"() {value = dense<1.0E-5> : tensor<f32>} : () -> tensor<f32>
    %mean = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<?x8xf32>) -> tensor<?x1xf32>
    %d = "onnx.Sub"(%x, %mean) : (tensor<?x8xf32>, tensor<?x1xf32>) -> tensor<?x8xf32>
    %sq = "onnx.Mul"(%d, %d) : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
    %var = "onnx.ReduceMean"(%sq) {axes = [1], keepdims = 1 : si64} : (tensor<?x8xf32>) -> tensor<?x1xf32>
    %ve = "onnx.Add"(%var, %eps) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
    %s = "onnx.Sqrt"(%ve) : (tensor<?x1xf32>) -> tensor<?x1xf32>
    %n = "onnx.Div"(%d, %s) : (tensor<?x8xf32>, tensor<?x1xf32>) -> tensor<?x8xf32>
    %y = "onnx.Mul"(%n, %g) : (tensor<?x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
    %shape = "onnx.Shape"(%mean) : (tensor<?x1xf32>) -> tensor<2xi64>
    return %y, %shape : tensor<?x8xf32>, tensor<2xi64>
  }

  // CHECK-LABEL: func.func @mean_shape_used
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x8xf32>, %[[G:.*]]: tensor<8xf32>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK: %[[Y:.*]] = hip.layer_norm(%[[CTX]])
  // CHECK: %[[D:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x8xf32>
  // CHECK: %[[D64:.*]] = arith.index_cast %[[D]]
  // CHECK: %[[S:.*]] = tensor.from_elements %[[D64]], %{{.*}} : tensor<2xi64>
  // CHECK: return %[[Y]], %[[S]]

}
