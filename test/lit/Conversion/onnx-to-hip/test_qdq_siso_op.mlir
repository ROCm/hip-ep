// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// DQ -> SISO -> Q with identical per-tensor scale/zero-point is just the SISO
// op over the quantized input. Storage type is not restricted. Layout-only
// SISO ops whose existing lowering already handles integer storage are
// covered; mismatched or non-scalar parameters must retain the original QDQ
// operations.

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x2x3x4xui16>)
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty() : tensor<1x3x4x2xui16>
  // CHECK-NEXT: %[[T:.*]] = hip.transpose(%[[CTX]]) ins(%[[X]] : tensor<1x2x3x4xui16>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<1x3x4x2xui16>)
  // CHECK-SAME: perm = [0, 2, 3, 1]
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return %[[T]]
  func.func @main_graph(%x: tensor<1x2x3x4xui16>)
      -> tensor<1x3x4x2xui16> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<32768> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<1x2x3x4xui16>, tensor<f32>, tensor<ui16>)
        -> tensor<1x2x3x4xf32>
    %t = "onnx.Transpose"(%dq) {perm = [0, 2, 3, 1]}
        : (tensor<1x2x3x4xf32>) -> tensor<1x3x4x2xf32>
    %q = "onnx.QuantizeLinear"(%t, %scale, %zp)
        : (tensor<1x3x4x2xf32>, tensor<f32>, tensor<ui16>)
        -> tensor<1x3x4x2xui16>
    return %q : tensor<1x3x4x2xui16>
  }

  // Exercise the com.microsoft form used by ORC and LoRA. Canonicalization
  // runs before PDLL, so it must eliminate the same sandwich.
  // CHECK-LABEL: func.func @custom_qdq_transpose
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NEXT: tensor.empty
  // CHECK-NEXT: hip.transpose
  // CHECK-SAME: ins(%{{.*}} : tensor<1x2x3x4xui16>)
  // CHECK-SAME: perm = [0, 3, 1, 2]
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return
  func.func @custom_qdq_transpose(%x: tensor<1x2x3x4xui16>)
      -> tensor<1x4x2x3xui16> {
    %scale = "onnx.Constant"() {
      value = dense<2.500000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<123> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.Custom"(%x, %scale, %zp) {
      domain_name = "com.microsoft",
      function_name = "DequantizeLinear"
    } : (tensor<1x2x3x4xui16>, tensor<f32>, tensor<ui16>)
        -> tensor<1x2x3x4xf32>
    %t = "onnx.Transpose"(%dq) {perm = [0, 3, 1, 2]}
        : (tensor<1x2x3x4xf32>) -> tensor<1x4x2x3xf32>
    %q = "onnx.Custom"(%t, %scale, %zp) {
      domain_name = "com.microsoft",
      function_name = "QuantizeLinear"
    } : (tensor<1x4x2x3xf32>, tensor<f32>, tensor<ui16>)
        -> tensor<1x4x2x3xui16>
    return %q : tensor<1x4x2x3xui16>
  }

  // CHECK-LABEL: func.func @different_scale_stays_unfused
  // CHECK: hip.dequantize_linear
  // CHECK: hip.transpose
  // CHECK: hip.quantize_linear
  func.func @different_scale_stays_unfused(%x: tensor<2x3xui16>)
      -> tensor<3x2xui16> {
    %in_scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %out_scale = "onnx.Constant"() {
      value = dense<2.500000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<7> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %in_scale, %zp)
        : (tensor<2x3xui16>, tensor<f32>, tensor<ui16>) -> tensor<2x3xf32>
    %t = "onnx.Transpose"(%dq) {perm = [1, 0]}
        : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %q = "onnx.QuantizeLinear"(%t, %out_scale, %zp)
        : (tensor<3x2xf32>, tensor<f32>, tensor<ui16>)
        -> tensor<3x2xui16>
    return %q : tensor<3x2xui16>
  }

  // CHECK-LABEL: func.func @different_zp_stays_unfused
  // CHECK: hip.dequantize_linear
  // CHECK: hip.transpose
  // CHECK: hip.quantize_linear
  func.func @different_zp_stays_unfused(%x: tensor<2x3xui16>)
      -> tensor<3x2xui16> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %in_zp = "onnx.Constant"() {
      value = dense<7> : tensor<ui16>
    } : () -> tensor<ui16>
    %out_zp = "onnx.Constant"() {
      value = dense<8> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %in_zp)
        : (tensor<2x3xui16>, tensor<f32>, tensor<ui16>) -> tensor<2x3xf32>
    %t = "onnx.Transpose"(%dq) {perm = [1, 0]}
        : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %q = "onnx.QuantizeLinear"(%t, %scale, %out_zp)
        : (tensor<3x2xf32>, tensor<f32>, tensor<ui16>)
        -> tensor<3x2xui16>
    return %q : tensor<3x2xui16>
  }

  // Squeeze is the same SISO QDQ skip: the quantized tensor is collapsed
  // without a dequant/quant round-trip.
  // CHECK-LABEL: func.func @qdq_squeeze
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NEXT: tensor.collapse_shape
  // CHECK-SAME: tensor<2x1x3xui16> into tensor<2x3xui16>
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return
  func.func @qdq_squeeze(%x: tensor<2x1x3xui16>) -> tensor<2x3xui16> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<32768> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<2x1x3xui16>, tensor<f32>, tensor<ui16>)
        -> tensor<2x1x3xf32>
    %axes = "onnx.Constant"() {
      value = dense<[1]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %s = "onnx.Squeeze"(%dq, %axes)
        : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x3xf32>
    %q = "onnx.QuantizeLinear"(%s, %scale, %zp)
        : (tensor<2x3xf32>, tensor<f32>, tensor<ui16>)
        -> tensor<2x3xui16>
    return %q : tensor<2x3xui16>
  }

  // INT8 with matching per-tensor params is the same skip; storage width is
  // not part of the match.
  // CHECK-LABEL: func.func @qdq_transpose_i8
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NEXT: tensor.empty
  // CHECK-NEXT: hip.transpose
  // CHECK-SAME: ins(%{{.*}} : tensor<2x3xi8>)
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return
  func.func @qdq_transpose_i8(%x: tensor<2x3xi8>) -> tensor<3x2xi8> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<0> : tensor<i8>
    } : () -> tensor<i8>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<2x3xi8>, tensor<f32>, tensor<i8>) -> tensor<2x3xf32>
    %t = "onnx.Transpose"(%dq) {perm = [1, 0]}
        : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %q = "onnx.QuantizeLinear"(%t, %scale, %zp)
        : (tensor<3x2xf32>, tensor<f32>, tensor<i8>) -> tensor<3x2xi8>
    return %q : tensor<3x2xi8>
  }

  // CHECK-LABEL: func.func @qdq_unsqueeze
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NEXT: tensor.expand_shape
  // CHECK-SAME: tensor<2x3xui16> into tensor<2x1x3xui16>
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return
  func.func @qdq_unsqueeze(%x: tensor<2x3xui16>) -> tensor<2x1x3xui16> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<32768> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<2x3xui16>, tensor<f32>, tensor<ui16>) -> tensor<2x3xf32>
    %axes = "onnx.Constant"() {
      value = dense<[1]> : tensor<1xi64>
    } : () -> tensor<1xi64>
    %u = "onnx.Unsqueeze"(%dq, %axes)
        : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
    %q = "onnx.QuantizeLinear"(%u, %scale, %zp)
        : (tensor<2x1x3xf32>, tensor<f32>, tensor<ui16>)
        -> tensor<2x1x3xui16>
    return %q : tensor<2x1x3xui16>
  }

  // CHECK-LABEL: func.func @qdq_reshape
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NEXT: tensor.collapse_shape
  // CHECK-SAME: tensor<1x2x3xui16> into tensor<1x6xui16>
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return
  func.func @qdq_reshape(%x: tensor<1x2x3xui16>) -> tensor<1x6xui16> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<32768> : tensor<ui16>
    } : () -> tensor<ui16>
    %shape = "onnx.Constant"() {
      value = dense<[1, 6]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<1x2x3xui16>, tensor<f32>, tensor<ui16>)
        -> tensor<1x2x3xf32>
    %r = "onnx.Reshape"(%dq, %shape)
        : (tensor<1x2x3xf32>, tensor<2xi64>) -> tensor<1x6xf32>
    %q = "onnx.QuantizeLinear"(%r, %scale, %zp)
        : (tensor<1x6xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x6xui16>
    return %q : tensor<1x6xui16>
  }

  // CHECK-LABEL: func.func @qdq_flatten
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NEXT: tensor.collapse_shape
  // CHECK-SAME: tensor<2x3x4xui16> into tensor<2x12xui16>
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return
  func.func @qdq_flatten(%x: tensor<2x3x4xui16>) -> tensor<2x12xui16> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<32768> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<2x3x4xui16>, tensor<f32>, tensor<ui16>)
        -> tensor<2x3x4xf32>
    %f = "onnx.Flatten"(%dq) {axis = 1 : si64}
        : (tensor<2x3x4xf32>) -> tensor<2x12xf32>
    %q = "onnx.QuantizeLinear"(%f, %scale, %zp)
        : (tensor<2x12xf32>, tensor<f32>, tensor<ui16>)
        -> tensor<2x12xui16>
    return %q : tensor<2x12xui16>
  }

  // CHECK-LABEL: func.func @qdq_identity
  // CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3xui16>
  // CHECK-NOT: hip.dequantize_linear
  // CHECK-NOT: onnx.Identity
  // CHECK-NOT: hip.quantize_linear
  // CHECK-NEXT: return %[[X]]
  func.func @qdq_identity(%x: tensor<2x3xui16>) -> tensor<2x3xui16> {
    %scale = "onnx.Constant"() {
      value = dense<1.250000e-01> : tensor<f32>
    } : () -> tensor<f32>
    %zp = "onnx.Constant"() {
      value = dense<32768> : tensor<ui16>
    } : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<2x3xui16>, tensor<f32>, tensor<ui16>) -> tensor<2x3xf32>
    %i = "onnx.Identity"(%dq) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %q = "onnx.QuantizeLinear"(%i, %scale, %zp)
        : (tensor<2x3xf32>, tensor<f32>, tensor<ui16>) -> tensor<2x3xui16>
    return %q : tensor<2x3xui16>
  }
}
