// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// DQ -> Transpose -> Q with identical UINT16 per-tensor parameters is just a
// Transpose over the quantized input. Mismatched or non-scalar parameters must
// retain the original QDQ operations.

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: %[[X:.*]]: tensor<1x2x3x4xui16>
  // CHECK-NOT: hip.dequantize_linear
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x3x4x2xui16>
  // CHECK: %[[T:.*]] = hip.transpose
  // CHECK-SAME: ins(%[[X]] : tensor<1x2x3x4xui16>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<1x3x4x2xui16>)
  // CHECK-SAME: perm = [0, 2, 3, 1]
  // CHECK-NOT: hip.quantize_linear
  // CHECK: return %[[T]]
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
  // CHECK: hip.transpose
  // CHECK-SAME: ins(%{{.*}} : tensor<1x2x3x4xui16>)
  // CHECK-SAME: perm = [0, 3, 1, 2]
  // CHECK-NOT: hip.quantize_linear
  // CHECK: return
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
}
