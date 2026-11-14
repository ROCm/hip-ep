// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the pre-lowering fold of primitive-op self-attention (the forms
// vision encoders export) into com.microsoft.MultiHeadAttention: the
// unmasked SigLIP form, an additive bias, the unbatched ViT form with a
// Where key mask, and a Where that is not a mask and must stay unfolded.
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

  // Gemma-4 vision form: Q and V come from [B,S,N,H] tensors that are not
  // reshapes, K is transposed twice, and a key mask is added to the scores.
  func.func @biased_attention(%q: tensor<1x8x2x4xf16>, %k: tensor<1x8x2x4xf16>,
                              %v: tensor<1x8x2x4xf16>, %mask: tensor<1x1x1x8xf16>)
      -> tensor<1x8x8xf16> {
    %merge = "onnx.Constant"() {value = dense<[0, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %qt = "onnx.Transpose"(%q) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x8x4xf16>
    %k1 = "onnx.Transpose"(%k) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x8x4xf16>
    %kt = "onnx.Transpose"(%k1) {perm = [0, 1, 3, 2]} : (tensor<1x2x8x4xf16>) -> tensor<1x2x4x8xf16>
    %vt = "onnx.Transpose"(%v) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x4xf16>) -> tensor<1x2x8x4xf16>
    %scores = "onnx.MatMul"(%qt, %kt) : (tensor<1x2x8x4xf16>, tensor<1x2x4x8xf16>) -> tensor<1x2x8x8xf16>
    %biased = "onnx.Add"(%scores, %mask) : (tensor<1x2x8x8xf16>, tensor<1x1x1x8xf16>) -> tensor<1x2x8x8xf16>
    %probs = "onnx.Softmax"(%biased) {axis = -1 : si64} : (tensor<1x2x8x8xf16>) -> tensor<1x2x8x8xf16>
    %ctx = "onnx.MatMul"(%probs, %vt) : (tensor<1x2x8x8xf16>, tensor<1x2x8x4xf16>) -> tensor<1x2x8x4xf16>
    %ot = "onnx.Transpose"(%ctx) {perm = [0, 2, 1, 3]} : (tensor<1x2x8x4xf16>) -> tensor<1x8x2x4xf16>
    %out = "onnx.Reshape"(%ot, %merge) : (tensor<1x8x2x4xf16>, tensor<3xi64>) -> tensor<1x8x8xf16>
    return %out : tensor<1x8x8xf16>
  }

  // CHECK-LABEL: func.func @biased_attention
  // CHECK-NOT: softmax
  // CHECK: hip.multi_head_attention
  // CHECK-SAME: tensor<1x1x1x8xf16>
  // CHECK-SAME: num_heads = 2
  // CHECK-SAME: scale = 1.0
  // CHECK-NOT: softmax
  // CHECK: return

  // Unbatched ViT form: [S,N,H] heads, scaled scores and a Where key mask.
  func.func @unbatched_masked_attention(%q: tensor<8x2x4xf16>, %k: tensor<8x2x4xf16>,
                                        %v: tensor<8x2x4xf16>, %keep: tensor<8x8xi1>)
      -> tensor<8x8xf16> {
    %merge = "onnx.Constant"() {value = dense<[-1, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %s = "onnx.Constant"() {value = dense<0.5> : tensor<f16>} : () -> tensor<f16>
    %ninf = "onnx.Constant"() {value = dense<0xFC00> : tensor<f16>} : () -> tensor<f16>
    %qt = "onnx.Transpose"(%q) {perm = [1, 0, 2]} : (tensor<8x2x4xf16>) -> tensor<2x8x4xf16>
    %kt = "onnx.Transpose"(%k) {perm = [1, 2, 0]} : (tensor<8x2x4xf16>) -> tensor<2x4x8xf16>
    %vt = "onnx.Transpose"(%v) {perm = [1, 0, 2]} : (tensor<8x2x4xf16>) -> tensor<2x8x4xf16>
    %scores = "onnx.MatMul"(%qt, %kt) : (tensor<2x8x4xf16>, tensor<2x4x8xf16>) -> tensor<2x8x8xf16>
    %scaled = "onnx.Mul"(%scores, %s) : (tensor<2x8x8xf16>, tensor<f16>) -> tensor<2x8x8xf16>
    %masked = "onnx.Where"(%keep, %scaled, %ninf) : (tensor<8x8xi1>, tensor<2x8x8xf16>, tensor<f16>) -> tensor<2x8x8xf16>
    %probs = "onnx.Softmax"(%masked) {axis = -1 : si64} : (tensor<2x8x8xf16>) -> tensor<2x8x8xf16>
    %ctx = "onnx.MatMul"(%probs, %vt) : (tensor<2x8x8xf16>, tensor<2x8x4xf16>) -> tensor<2x8x4xf16>
    %ot = "onnx.Transpose"(%ctx) {perm = [1, 0, 2]} : (tensor<2x8x4xf16>) -> tensor<8x2x4xf16>
    %out = "onnx.Reshape"(%ot, %merge) : (tensor<8x2x4xf16>, tensor<2xi64>) -> tensor<8x8xf16>
    return %out : tensor<8x8xf16>
  }

  // CHECK-LABEL: func.func @unbatched_masked_attention
  // CHECK-NOT: softmax
  // CHECK: hip.where
  // CHECK: hip.multi_head_attention
  // CHECK-SAME: tensor<8x8xf16>
  // CHECK-SAME: num_heads = 2
  // CHECK-SAME: scale = 5.0
  // CHECK-NOT: softmax
  // CHECK: return

  // A Where that fills with 0 instead of a large negative value is not a key
  // mask, so the subgraph stays as it is.
  func.func @where_not_a_mask(%q: tensor<8x2x4xf16>, %k: tensor<8x2x4xf16>,
                              %v: tensor<8x2x4xf16>, %keep: tensor<8x8xi1>)
      -> tensor<8x8xf16> {
    %merge = "onnx.Constant"() {value = dense<[-1, 8]> : tensor<2xi64>} : () -> tensor<2xi64>
    %zero = "onnx.Constant"() {value = dense<0.0> : tensor<f16>} : () -> tensor<f16>
    %qt = "onnx.Transpose"(%q) {perm = [1, 0, 2]} : (tensor<8x2x4xf16>) -> tensor<2x8x4xf16>
    %kt = "onnx.Transpose"(%k) {perm = [1, 2, 0]} : (tensor<8x2x4xf16>) -> tensor<2x4x8xf16>
    %vt = "onnx.Transpose"(%v) {perm = [1, 0, 2]} : (tensor<8x2x4xf16>) -> tensor<2x8x4xf16>
    %scores = "onnx.MatMul"(%qt, %kt) : (tensor<2x8x4xf16>, tensor<2x4x8xf16>) -> tensor<2x8x8xf16>
    %masked = "onnx.Where"(%keep, %scores, %zero) : (tensor<8x8xi1>, tensor<2x8x8xf16>, tensor<f16>) -> tensor<2x8x8xf16>
    %probs = "onnx.Softmax"(%masked) {axis = -1 : si64} : (tensor<2x8x8xf16>) -> tensor<2x8x8xf16>
    %ctx = "onnx.MatMul"(%probs, %vt) : (tensor<2x8x8xf16>, tensor<2x8x4xf16>) -> tensor<2x8x4xf16>
    %ot = "onnx.Transpose"(%ctx) {perm = [1, 0, 2]} : (tensor<2x8x4xf16>) -> tensor<8x2x4xf16>
    %out = "onnx.Reshape"(%ot, %merge) : (tensor<8x2x4xf16>, tensor<2xi64>) -> tensor<8x8xf16>
    return %out : tensor<8x8xf16>
  }

  // CHECK-LABEL: func.func @where_not_a_mask
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: softmax
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: return
}
