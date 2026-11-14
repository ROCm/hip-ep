// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the pre-lowering fold of primitive-op self-attention (the SigLIP /
// ViT export form) into com.microsoft.MultiHeadAttention, and that a masked
// variant is left alone.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    return %arg0 : tensor<1x1xf16>
  }

  // Q and K are each pre-scaled by H^-0.25 and the output reshape target is
  // built from the input's shape, as in the Gemma3 vision tower export.
  func.func @siglip_attention(%x: tensor<1x8x8xf16>, %q: tensor<1x8x8xf16>,
                              %k: tensor<1x8x8xf16>, %v: tensor<1x8x8xf16>)
      -> tensor<?x?x?xf16> {
    %split = "onnx.Constant"() {value = dense<[1, 8, 2, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %s = "onnx.Constant"() {value = dense<0.707106769> : tensor<f16>} : () -> tensor<f16>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<1x8x8xf16>, tensor<4xi64>) -> tensor<1x8x2x4xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x8x4xf16>
    %qs = "onnx.Mul"(%qt, %s) : (tensor<1x2x8x4xf16>, tensor<f16>) -> tensor<1x2x8x4xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<1x8x8xf16>, tensor<4xi64>) -> tensor<1x8x2x4xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x4x8xf16>
    %ks = "onnx.Mul"(%kt, %s) : (tensor<1x2x4x8xf16>, tensor<f16>) -> tensor<1x2x4x8xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<1x8x8xf16>, tensor<4xi64>) -> tensor<1x8x2x4xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x8x4xf16>
    %scores = "onnx.MatMul"(%qs, %ks) : (tensor<1x2x8x4xf16>, tensor<1x2x4x8xf16>) -> tensor<1x2x8x8xf16>
    %probs = "onnx.Softmax"(%scores) {axis = -1 : si64} : (tensor<1x2x8x8xf16>) -> tensor<1x2x8x8xf16>
    %ctx = "onnx.MatMul"(%probs, %vt) : (tensor<1x2x8x8xf16>, tensor<1x2x8x4xf16>) -> tensor<1x2x8x4xf16>
    %ot = "onnx.Transpose"(%ctx) {perm = [0, 2, 1, 3]} : (tensor<1x2x8x4xf16>) -> tensor<1x8x2x4xf16>
    %shape = "onnx.Shape"(%x) : (tensor<1x8x8xf16>) -> tensor<3xi64>
    %i0 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %i1 = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %i2 = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %ax = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %d0 = "onnx.Gather"(%shape, %i0) {axis = 0 : si64} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %d1 = "onnx.Gather"(%shape, %i1) {axis = 0 : si64} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %d2 = "onnx.Gather"(%shape, %i2) {axis = 0 : si64} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %u0 = "onnx.Unsqueeze"(%d0, %ax) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %u1 = "onnx.Unsqueeze"(%d1, %ax) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %u2 = "onnx.Unsqueeze"(%d2, %ax) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %merge = "onnx.Concat"(%u0, %u1, %u2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %out = "onnx.Reshape"(%ot, %merge) : (tensor<1x8x2x4xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %out : tensor<?x?x?xf16>
  }

  // The two f16 pre-scales (0.70703125 each) multiply into the MHA scale.
  // CHECK-LABEL: func.func @siglip_attention
  // CHECK-NOT: softmax
  // CHECK: hip.multi_head_attention
  // CHECK-SAME: num_heads = 2
  // CHECK-SAME: scale = 0.4998{{[0-9]+}}
  // CHECK-NOT: softmax
  // CHECK: return

  // An additive mask between Q x K^T and the softmax: the runtime MHA has no
  // attention bias, so the subgraph is left as is.
  func.func @masked_attention(%q: tensor<1x8x8xf16>, %k: tensor<1x8x8xf16>,
                              %v: tensor<1x8x8xf16>, %mask: tensor<1x1x8x8xf16>)
      -> tensor<1x8x8xf16> {
    %split = "onnx.Constant"() {value = dense<[1, 8, 2, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %merge = "onnx.Constant"() {value = dense<[1, 8, 8]> : tensor<3xi64>} : () -> tensor<3xi64>
    %q4 = "onnx.Reshape"(%q, %split) : (tensor<1x8x8xf16>, tensor<4xi64>) -> tensor<1x8x2x4xf16>
    %qt = "onnx.Transpose"(%q4) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x8x4xf16>
    %k4 = "onnx.Reshape"(%k, %split) : (tensor<1x8x8xf16>, tensor<4xi64>) -> tensor<1x8x2x4xf16>
    %kt = "onnx.Transpose"(%k4) {perm = [0, 2, 3, 1]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x4x8xf16>
    %v4 = "onnx.Reshape"(%v, %split) : (tensor<1x8x8xf16>, tensor<4xi64>) -> tensor<1x8x2x4xf16>
    %vt = "onnx.Transpose"(%v4) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x8x4xf16>
    %scores = "onnx.MatMul"(%qt, %kt) : (tensor<1x2x8x4xf16>, tensor<1x2x4x8xf16>) -> tensor<1x2x8x8xf16>
    %biased = "onnx.Add"(%scores, %mask) : (tensor<1x2x8x8xf16>, tensor<1x1x8x8xf16>) -> tensor<1x2x8x8xf16>
    %probs = "onnx.Softmax"(%biased) {axis = -1 : si64} : (tensor<1x2x8x8xf16>) -> tensor<1x2x8x8xf16>
    %ctx = "onnx.MatMul"(%probs, %vt) : (tensor<1x2x8x8xf16>, tensor<1x2x8x4xf16>) -> tensor<1x2x8x4xf16>
    %ot = "onnx.Transpose"(%ctx) {perm = [0, 2, 1, 3]} : (tensor<1x2x8x4xf16>) -> tensor<1x8x2x4xf16>
    %out = "onnx.Reshape"(%ot, %merge) : (tensor<1x8x2x4xf16>, tensor<3xi64>) -> tensor<1x8x8xf16>
    return %out : tensor<1x8x8xf16>
  }

  // CHECK-LABEL: func.func @masked_attention
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: softmax
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: return
}
