// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that primitive scaled-dot-product attention (Transpose / scale /
// MatMul / Softmax / MatMul / Transpose over BSNH projections) is rewritten to
// com.microsoft.MultiHeadAttention and so lowered to hip.multi_head_attention,
// with the scale folded from wherever the export put it, and that shapes the
// flash path does not take keep their primitive lowering.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    return %arg0 : tensor<1x1xf16>
  }

  // diffusers-style self-attention: H^-1/4 on both Q and K^T, K^T as one
  // [0, 2, 3, 1] Transpose. B=2, S=64, N=2, H=40.
  // CHECK-LABEL: func.func @self_attention_split_scale
  // CHECK-NOT: hip.miopen.softmax
  // CHECK-NOT: hip.transpose
  // CHECK: hip.multi_head_attention
  // CHECK-SAME: tensor<2x64x80xf16>, tensor<2x64x80xf16>, tensor<2x64x80xf16>
  // CHECK-SAME: num_heads = 2
  // CHECK-SAME: scale = 0.158169
  // CHECK-NOT: hip.miopen.softmax
  // CHECK: return
  func.func @self_attention_split_scale(%q: tensor<2x64x80xf16>,
                                        %k: tensor<2x64x80xf16>,
                                        %v: tensor<2x64x80xf16>)
      -> tensor<2x64x2x40xf16> {
    %shape = "onnx.Constant"() {value = dense<[2, -1, 2, 40]> : tensor<4xi64>} : () -> tensor<4xi64>
    %c = "onnx.Constant"() {value = dense<0.397705078125> : tensor<1xf16>} : () -> tensor<1xf16>
    %qr = "onnx.Reshape"(%q, %shape) {allowzero = 0 : si64} : (tensor<2x64x80xf16>, tensor<4xi64>) -> tensor<2x64x2x40xf16>
    %qt = "onnx.Transpose"(%qr) {perm = [0, 2, 1, 3]} : (tensor<2x64x2x40xf16>) -> tensor<2x2x64x40xf16>
    %qs = "onnx.Mul"(%qt, %c) : (tensor<2x2x64x40xf16>, tensor<1xf16>) -> tensor<2x2x64x40xf16>
    %kr = "onnx.Reshape"(%k, %shape) {allowzero = 0 : si64} : (tensor<2x64x80xf16>, tensor<4xi64>) -> tensor<2x64x2x40xf16>
    %kt = "onnx.Transpose"(%kr) {perm = [0, 2, 3, 1]} : (tensor<2x64x2x40xf16>) -> tensor<2x2x40x64xf16>
    %ks = "onnx.Mul"(%kt, %c) : (tensor<2x2x40x64xf16>, tensor<1xf16>) -> tensor<2x2x40x64xf16>
    %vr = "onnx.Reshape"(%v, %shape) {allowzero = 0 : si64} : (tensor<2x64x80xf16>, tensor<4xi64>) -> tensor<2x64x2x40xf16>
    %vt = "onnx.Transpose"(%vr) {perm = [0, 2, 1, 3]} : (tensor<2x64x2x40xf16>) -> tensor<2x2x64x40xf16>
    %s = "onnx.MatMul"(%qs, %ks) : (tensor<2x2x64x40xf16>, tensor<2x2x40x64xf16>) -> tensor<2x2x64x64xf16>
    %p = "onnx.Softmax"(%s) {axis = -1 : si64} : (tensor<2x2x64x64xf16>) -> tensor<2x2x64x64xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<2x2x64x64xf16>, tensor<2x2x64x40xf16>) -> tensor<2x2x64x40xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<2x2x64x40xf16>) -> tensor<2x64x2x40xf16>
    return %ot : tensor<2x64x2x40xf16>
  }

  // Cross-attention (Skv != Sq) with the scale as a Div on the scores and K^T
  // spelled Transpose(BNSH, [0, 1, 3, 2]); the output stays BNSH, so a
  // Transpose is emitted for it.
  // CHECK-LABEL: func.func @cross_attention_score_div
  // CHECK-NOT: hip.miopen.softmax
  // CHECK: hip.multi_head_attention
  // CHECK-SAME: tensor<1x32x64xf16>, tensor<1x7x64xf16>, tensor<1x7x64xf16>
  // CHECK-SAME: num_heads = 4
  // CHECK-SAME: scale = 2.500000e-01
  // CHECK: hip.transpose
  // CHECK-NOT: hip.miopen.softmax
  // CHECK: return
  func.func @cross_attention_score_div(%q: tensor<1x32x64xf16>,
                                       %k: tensor<1x7x64xf16>,
                                       %v: tensor<1x7x64xf16>)
      -> tensor<1x4x32x16xf16> {
    %qshape = "onnx.Constant"() {value = dense<[1, 32, 4, 16]> : tensor<4xi64>} : () -> tensor<4xi64>
    %kshape = "onnx.Constant"() {value = dense<[1, 7, 4, 16]> : tensor<4xi64>} : () -> tensor<4xi64>
    %d = "onnx.Constant"() {value = dense<4.0> : tensor<f16>} : () -> tensor<f16>
    %qr = "onnx.Reshape"(%q, %qshape) : (tensor<1x32x64xf16>, tensor<4xi64>) -> tensor<1x32x4x16xf16>
    %qt = "onnx.Transpose"(%qr) {perm = [0, 2, 1, 3]} : (tensor<1x32x4x16xf16>) -> tensor<1x4x32x16xf16>
    %kr = "onnx.Reshape"(%k, %kshape) : (tensor<1x7x64xf16>, tensor<4xi64>) -> tensor<1x7x4x16xf16>
    %kb = "onnx.Transpose"(%kr) {perm = [0, 2, 1, 3]} : (tensor<1x7x4x16xf16>) -> tensor<1x4x7x16xf16>
    %kt = "onnx.Transpose"(%kb) {perm = [0, 1, 3, 2]} : (tensor<1x4x7x16xf16>) -> tensor<1x4x16x7xf16>
    %vr = "onnx.Reshape"(%v, %kshape) : (tensor<1x7x64xf16>, tensor<4xi64>) -> tensor<1x7x4x16xf16>
    %vt = "onnx.Transpose"(%vr) {perm = [0, 2, 1, 3]} : (tensor<1x7x4x16xf16>) -> tensor<1x4x7x16xf16>
    %s = "onnx.MatMul"(%qt, %kt) : (tensor<1x4x32x16xf16>, tensor<1x4x16x7xf16>) -> tensor<1x4x32x7xf16>
    %sd = "onnx.Div"(%s, %d) : (tensor<1x4x32x7xf16>, tensor<f16>) -> tensor<1x4x32x7xf16>
    %p = "onnx.Softmax"(%sd) {axis = -1 : si64} : (tensor<1x4x32x7xf16>) -> tensor<1x4x32x7xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<1x4x32x7xf16>, tensor<1x4x7x16xf16>) -> tensor<1x4x32x16xf16>
    return %o : tensor<1x4x32x16xf16>
  }

  // A softmax over another axis is not attention.
  // CHECK-LABEL: func.func @softmax_not_last_axis
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: hip.miopen.softmax
  func.func @softmax_not_last_axis(%q: tensor<1x8x2x16xf16>,
                                   %k: tensor<1x8x2x16xf16>,
                                   %v: tensor<1x8x2x16xf16>)
      -> tensor<1x2x8x16xf16> {
    %qt = "onnx.Transpose"(%q) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %kt = "onnx.Transpose"(%k) {perm = [0, 2, 3, 1]} : (tensor<1x8x2x16xf16>) -> tensor<1x2x16x8xf16>
    %vt = "onnx.Transpose"(%v) {perm = [0, 2, 1, 3]} : (tensor<1x8x2x16xf16>) -> tensor<1x2x8x16xf16>
    %s = "onnx.MatMul"(%qt, %kt) : (tensor<1x2x8x16xf16>, tensor<1x2x16x8xf16>) -> tensor<1x2x8x8xf16>
    %p = "onnx.Softmax"(%s) {axis = 2 : si64} : (tensor<1x2x8x8xf16>) -> tensor<1x2x8x8xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<1x2x8x8xf16>, tensor<1x2x8x16xf16>) -> tensor<1x2x8x16xf16>
    return %o : tensor<1x2x8x16xf16>
  }

  // A head dimension past the flash kernel's 256 keeps the primitive path.
  // CHECK-LABEL: func.func @head_dim_too_wide
  // CHECK-NOT: hip.multi_head_attention
  // CHECK: hip.miopen.softmax
  func.func @head_dim_too_wide(%q: tensor<1x16x1x512xf16>,
                               %k: tensor<1x16x1x512xf16>,
                               %v: tensor<1x16x1x512xf16>)
      -> tensor<1x16x1x512xf16> {
    %qt = "onnx.Transpose"(%q) {perm = [0, 2, 1, 3]} : (tensor<1x16x1x512xf16>) -> tensor<1x1x16x512xf16>
    %kt = "onnx.Transpose"(%k) {perm = [0, 2, 3, 1]} : (tensor<1x16x1x512xf16>) -> tensor<1x1x512x16xf16>
    %vt = "onnx.Transpose"(%v) {perm = [0, 2, 1, 3]} : (tensor<1x16x1x512xf16>) -> tensor<1x1x16x512xf16>
    %s = "onnx.MatMul"(%qt, %kt) : (tensor<1x1x16x512xf16>, tensor<1x1x512x16xf16>) -> tensor<1x1x16x16xf16>
    %p = "onnx.Softmax"(%s) {axis = -1 : si64} : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
    %o = "onnx.MatMul"(%p, %vt) : (tensor<1x1x16x16xf16>, tensor<1x1x16x512xf16>) -> tensor<1x1x16x512xf16>
    %ot = "onnx.Transpose"(%o) {perm = [0, 2, 1, 3]} : (tensor<1x1x16x512xf16>) -> tensor<1x16x1x512xf16>
    return %ot : tensor<1x16x1x512xf16>
  }
}
