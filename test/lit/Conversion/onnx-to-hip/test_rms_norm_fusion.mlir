// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// An unweighted RMS normalization
//
//   y = x / sqrt(mean(x^2) + eps)
//
// spelled out as primitives (no learned gamma, so exporters have no fused op
// to emit) folds back into one hip.rms_norm with an all-ones scale. Attention
// q/k normalization is the usual source: the reduction runs over the head
// dimension, so without this every layer pays six to eight launches to
// normalize a few kilobytes.
//
// Negative cases below pin the conditions the rewrite depends on for
// correctness: the reduction must be a keepdims mean over the trailing axis,
// the squared value must be x itself, and the normalized extent must be
// static.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Canonical decode-shaped chain: 8 kv heads x 256 head_dim, bracketed by the
  // fp32 cast pair exporters add to keep the sum of squares off fp16. The
  // kernel accumulates in fp32 regardless, so both casts are absorbed and the
  // whole chain becomes a single f16 hip.rms_norm.
  func.func @rms_norm_qk_f16_cast_bracket(%x: tensor<1x1x8x256xf16>)
      -> tensor<1x1x8x256xf16> {
    %xf = "onnx.Cast"(%x) {to = f32}
        : (tensor<1x1x8x256xf16>) -> tensor<1x1x8x256xf32>
    %sq = "onnx.Mul"(%xf, %xf)
        : (tensor<1x1x8x256xf32>, tensor<1x1x8x256xf32>) -> tensor<1x1x8x256xf32>
    %mean = "onnx.ReduceMean"(%sq) {axes = [-1 : si64], keepdims = 1 : si64}
        : (tensor<1x1x8x256xf32>) -> tensor<1x1x8x1xf32>
    %eps = "onnx.Constant"() {value = dense<9.99999997E-7> : tensor<1xf32>}
        : () -> tensor<1xf32>
    %add = "onnx.Add"(%mean, %eps)
        : (tensor<1x1x8x1xf32>, tensor<1xf32>) -> tensor<1x1x8x1xf32>
    %rms = "onnx.Sqrt"(%add) : (tensor<1x1x8x1xf32>) -> tensor<1x1x8x1xf32>
    %inv = "onnx.Reciprocal"(%rms)
        : (tensor<1x1x8x1xf32>) -> tensor<1x1x8x1xf32>
    %nf = "onnx.Mul"(%xf, %inv)
        : (tensor<1x1x8x256xf32>, tensor<1x1x8x1xf32>) -> tensor<1x1x8x256xf32>
    %y = "onnx.Cast"(%nf) {to = f16}
        : (tensor<1x1x8x256xf32>) -> tensor<1x1x8x256xf16>
    return %y : tensor<1x1x8x256xf16>
  }
  // CHECK-LABEL: func.func @rms_norm_qk_f16_cast_bracket
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x1x8x256xf16>)
  // CHECK-NOT: hip.reduce_mean
  // CHECK-NOT: hip.reciprocal
  // CHECK-NOT: hip.sqrt
  // CHECK-NOT: hip.cast
  // CHECK: hip.rms_norm
  // CHECK-NOT: hip.reduce_mean

  // Same chain with no cast bracket and a bare broadcasting Div instead of
  // Reciprocal + Mul. BroadcastDivToMulReciprocal canonicalizes the Div in the
  // same pre-lowering loop, and either spelling reaches the fused op.
  func.func @rms_norm_div_no_cast_f32(%x: tensor<2x512xf32>) -> tensor<2x512xf32> {
    %sq = "onnx.Mul"(%x, %x)
        : (tensor<2x512xf32>, tensor<2x512xf32>) -> tensor<2x512xf32>
    %mean = "onnx.ReduceMean"(%sq) {axes = [-1 : si64], keepdims = 1 : si64}
        : (tensor<2x512xf32>) -> tensor<2x1xf32>
    %eps = "onnx.Constant"() {value = dense<1.000000e-06> : tensor<1xf32>}
        : () -> tensor<1xf32>
    %add = "onnx.Add"(%mean, %eps)
        : (tensor<2x1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
    %rms = "onnx.Sqrt"(%add) : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %y = "onnx.Div"(%x, %rms) : (tensor<2x512xf32>, tensor<2x1xf32>) -> tensor<2x512xf32>
    return %y : tensor<2x512xf32>
  }
  // CHECK-LABEL: func.func @rms_norm_div_no_cast_f32
  // CHECK-NOT: hip.reduce_mean
  // CHECK: hip.rms_norm

  // Epsilon is optional; a chain that goes straight from the mean to the sqrt
  // still fuses, with epsilon = 0.
  func.func @rms_norm_no_epsilon_f16(%x: tensor<1x16x256xf16>) -> tensor<1x16x256xf16> {
    %sq = "onnx.Mul"(%x, %x)
        : (tensor<1x16x256xf16>, tensor<1x16x256xf16>) -> tensor<1x16x256xf16>
    %mean = "onnx.ReduceMean"(%sq) {axes = [-1 : si64], keepdims = 1 : si64}
        : (tensor<1x16x256xf16>) -> tensor<1x16x1xf16>
    %rms = "onnx.Sqrt"(%mean) : (tensor<1x16x1xf16>) -> tensor<1x16x1xf16>
    %inv = "onnx.Reciprocal"(%rms) : (tensor<1x16x1xf16>) -> tensor<1x16x1xf16>
    %y = "onnx.Mul"(%x, %inv)
        : (tensor<1x16x256xf16>, tensor<1x16x1xf16>) -> tensor<1x16x256xf16>
    return %y : tensor<1x16x256xf16>
  }
  // CHECK-LABEL: func.func @rms_norm_no_epsilon_f16
  // CHECK-NOT: hip.reduce_mean
  // CHECK: hip.rms_norm

  // Axes given as the opset-18 constant operand rather than the attribute.
  func.func @rms_norm_axes_operand_f16(%x: tensor<1x8x256xf16>) -> tensor<1x8x256xf16> {
    %ax = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sq = "onnx.Mul"(%x, %x)
        : (tensor<1x8x256xf16>, tensor<1x8x256xf16>) -> tensor<1x8x256xf16>
    %mean = "onnx.ReduceMean"(%sq, %ax) {keepdims = 1 : si64}
        : (tensor<1x8x256xf16>, tensor<1xi64>) -> tensor<1x8x1xf16>
    %rms = "onnx.Sqrt"(%mean) : (tensor<1x8x1xf16>) -> tensor<1x8x1xf16>
    %inv = "onnx.Reciprocal"(%rms) : (tensor<1x8x1xf16>) -> tensor<1x8x1xf16>
    %y = "onnx.Mul"(%x, %inv)
        : (tensor<1x8x256xf16>, tensor<1x8x1xf16>) -> tensor<1x8x256xf16>
    return %y : tensor<1x8x256xf16>
  }
  // CHECK-LABEL: func.func @rms_norm_axes_operand_f16
  // CHECK-NOT: hip.reduce_mean
  // CHECK: hip.rms_norm

  // NEGATIVE: reduction over a non-trailing axis is not an RMS norm over the
  // normalized extent rms_norm assumes, so the chain must stay decomposed.
  func.func @rms_norm_axis_not_trailing(%x: tensor<2x4x8xf16>) -> tensor<2x4x8xf16> {
    %sq = "onnx.Mul"(%x, %x)
        : (tensor<2x4x8xf16>, tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
    %mean = "onnx.ReduceMean"(%sq) {axes = [1 : si64], keepdims = 1 : si64}
        : (tensor<2x4x8xf16>) -> tensor<2x1x8xf16>
    %rms = "onnx.Sqrt"(%mean) : (tensor<2x1x8xf16>) -> tensor<2x1x8xf16>
    %inv = "onnx.Reciprocal"(%rms) : (tensor<2x1x8xf16>) -> tensor<2x1x8xf16>
    %y = "onnx.Mul"(%x, %inv)
        : (tensor<2x4x8xf16>, tensor<2x1x8xf16>) -> tensor<2x4x8xf16>
    return %y : tensor<2x4x8xf16>
  }
  // CHECK-LABEL: func.func @rms_norm_axis_not_trailing
  // CHECK-NOT: hip.rms_norm
  // CHECK: hip.reduce_mean

  // NEGATIVE: the reduced value is a product of two different tensors, so the
  // mean is not a mean of squares.
  func.func @rms_norm_not_a_square(%x: tensor<2x8xf16>, %z: tensor<2x8xf16>)
      -> tensor<2x8xf16> {
    %sq = "onnx.Mul"(%x, %z)
        : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
    %mean = "onnx.ReduceMean"(%sq) {axes = [-1 : si64], keepdims = 1 : si64}
        : (tensor<2x8xf16>) -> tensor<2x1xf16>
    %rms = "onnx.Sqrt"(%mean) : (tensor<2x1xf16>) -> tensor<2x1xf16>
    %inv = "onnx.Reciprocal"(%rms) : (tensor<2x1xf16>) -> tensor<2x1xf16>
    %y = "onnx.Mul"(%x, %inv)
        : (tensor<2x8xf16>, tensor<2x1xf16>) -> tensor<2x8xf16>
    return %y : tensor<2x8xf16>
  }
  // CHECK-LABEL: func.func @rms_norm_not_a_square
  // CHECK-NOT: hip.rms_norm
  // CHECK: hip.reduce_mean

  // NEGATIVE: keepdims=0 drops the broadcast axis the chain relies on.
  func.func @rms_norm_no_keepdims(%x: tensor<2x8xf16>) -> tensor<2x8xf16> {
    %sq = "onnx.Mul"(%x, %x)
        : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
    %mean = "onnx.ReduceMean"(%sq) {axes = [-1 : si64], keepdims = 0 : si64}
        : (tensor<2x8xf16>) -> tensor<2xf16>
    %rms = "onnx.Sqrt"(%mean) : (tensor<2xf16>) -> tensor<2xf16>
    %inv = "onnx.Reciprocal"(%rms) : (tensor<2xf16>) -> tensor<2xf16>
    %y = "onnx.Mul"(%x, %inv)
        : (tensor<2x8xf16>, tensor<2xf16>) -> tensor<2x8xf16>
    return %y : tensor<2x8xf16>
  }
  // CHECK-LABEL: func.func @rms_norm_no_keepdims
  // CHECK-NOT: hip.rms_norm
  // CHECK: hip.reduce_mean
}
