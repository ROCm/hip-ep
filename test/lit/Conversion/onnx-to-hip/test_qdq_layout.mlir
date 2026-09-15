// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: skip matching UINT16 QDQ around Unsqueeze / Squeeze / Reshape
//
// Matching per-tensor scale and zero point:
//   DequantizeLinear -> {Unsqueeze,Squeeze,Reshape} -> QuantizeLinear
// becomes the layout op on the quantized tensor, which convert-onnx-to-hip
// then lowers to tensor.expand_shape / collapse_shape (zero-cost metadata).
//
// A mismatched scale keeps the unfused Q/DQ chain.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128x2048xui16>) -> tensor<128x2048xui16> {
    return %arg0 : tensor<128x2048xui16>
  }

// CHECK-LABEL: func.func @qdq_unsqueeze
// CHECK-NOT:   hip.dequantize_linear
// CHECK-NOT:   hip.quantize_linear
// CHECK:       tensor.expand_shape
// CHECK-SAME:  tensor<128x2048xui16> into tensor<128x1x2048xui16>
// CHECK-NOT:   hip.quantize_linear
  func.func @qdq_unsqueeze(%x: tensor<128x2048xui16>) -> tensor<128x1x2048xui16> {
    %scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<5> : tensor<ui16>} : () -> tensor<ui16>
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<128x2048xui16>, tensor<f32>, tensor<ui16>) -> tensor<128x2048xf32>
    %u = "onnx.Unsqueeze"(%dq, %axes)
        : (tensor<128x2048xf32>, tensor<1xi64>) -> tensor<128x1x2048xf32>
    %q = "onnx.QuantizeLinear"(%u, %scale, %zp)
        : (tensor<128x1x2048xf32>, tensor<f32>, tensor<ui16>) -> tensor<128x1x2048xui16>
    return %q : tensor<128x1x2048xui16>
  }

// CHECK-LABEL: func.func @qdq_squeeze
// CHECK-NOT:   hip.dequantize_linear
// CHECK-NOT:   hip.quantize_linear
// CHECK:       tensor.collapse_shape
// CHECK-SAME:  tensor<128x1x2048xui16> into tensor<128x2048xui16>
// CHECK-NOT:   hip.quantize_linear
  func.func @qdq_squeeze(%x: tensor<128x1x2048xui16>) -> tensor<128x2048xui16> {
    %scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<5> : tensor<ui16>} : () -> tensor<ui16>
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<128x1x2048xui16>, tensor<f32>, tensor<ui16>) -> tensor<128x1x2048xf32>
    %s = "onnx.Squeeze"(%dq, %axes)
        : (tensor<128x1x2048xf32>, tensor<1xi64>) -> tensor<128x2048xf32>
    %q = "onnx.QuantizeLinear"(%s, %scale, %zp)
        : (tensor<128x2048xf32>, tensor<f32>, tensor<ui16>) -> tensor<128x2048xui16>
    return %q : tensor<128x2048xui16>
  }

// CHECK-LABEL: func.func @qdq_reshape
// CHECK-NOT:   hip.dequantize_linear
// CHECK-NOT:   hip.quantize_linear
// CHECK:       tensor.expand_shape
// CHECK-SAME:  tensor<128x2048xui16> into tensor<128x16x128xui16>
// CHECK-NOT:   hip.quantize_linear
  func.func @qdq_reshape(%x: tensor<128x2048xui16>) -> tensor<128x16x128xui16> {
    %scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<5> : tensor<ui16>} : () -> tensor<ui16>
    %shape = "onnx.Constant"() {value = dense<[128, 16, 128]> : tensor<3xi64>} : () -> tensor<3xi64>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<128x2048xui16>, tensor<f32>, tensor<ui16>) -> tensor<128x2048xf32>
    %r = "onnx.Reshape"(%dq, %shape)
        : (tensor<128x2048xf32>, tensor<3xi64>) -> tensor<128x16x128xf32>
    %q = "onnx.QuantizeLinear"(%r, %scale, %zp)
        : (tensor<128x16x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<128x16x128xui16>
    return %q : tensor<128x16x128xui16>
  }

// MorphiZen Custom QDQ must canonicalize before PDLL.
// CHECK-LABEL: func.func @custom_qdq_unsqueeze
// CHECK-NOT:   hip.dequantize_linear
// CHECK-NOT:   hip.quantize_linear
// CHECK:       tensor.expand_shape
// CHECK-NOT:   hip.quantize_linear
  func.func @custom_qdq_unsqueeze(%x: tensor<4xui16>) -> tensor<1x4xui16> {
    %scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<5> : tensor<ui16>} : () -> tensor<ui16>
    %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %dq = "onnx.Custom"(%x, %scale, %zp) {
      domain_name = "com.microsoft", function_name = "DequantizeLinear"
    } : (tensor<4xui16>, tensor<f32>, tensor<ui16>) -> tensor<4xf32>
    %u = "onnx.Unsqueeze"(%dq, %axes)
        : (tensor<4xf32>, tensor<1xi64>) -> tensor<1x4xf32>
    %q = "onnx.Custom"(%u, %scale, %zp) {
      domain_name = "com.microsoft", function_name = "QuantizeLinear"
    } : (tensor<1x4xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x4xui16>
    return %q : tensor<1x4xui16>
  }

// Different scale: the round-trip is not an identity.
// CHECK-LABEL: func.func @qdq_unsqueeze_mismatch_scale
// CHECK:       hip.dequantize_linear
// CHECK:       tensor.expand_shape
// CHECK:       hip.quantize_linear
  func.func @qdq_unsqueeze_mismatch_scale(%x: tensor<8x16xui16>) -> tensor<8x1x16xui16> {
    %in_s = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %out_s = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<0> : tensor<ui16>} : () -> tensor<ui16>
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %dq = "onnx.DequantizeLinear"(%x, %in_s, %zp)
        : (tensor<8x16xui16>, tensor<f32>, tensor<ui16>) -> tensor<8x16xf32>
    %u = "onnx.Unsqueeze"(%dq, %axes)
        : (tensor<8x16xf32>, tensor<1xi64>) -> tensor<8x1x16xf32>
    %q = "onnx.QuantizeLinear"(%u, %out_s, %zp)
        : (tensor<8x1x16xf32>, tensor<f32>, tensor<ui16>) -> tensor<8x1x16xui16>
    return %q : tensor<8x1x16xui16>
  }

// FP16 cannot represent every UINT16 code exactly, so its QDQ round-trip is
// not an identity and must not be removed.
// CHECK-LABEL: func.func @qdq_unsqueeze_fp16
// CHECK:       hip.dequantize_linear
// CHECK:       tensor.expand_shape
// CHECK:       hip.quantize_linear
  func.func @qdq_unsqueeze_fp16(%x: tensor<8x16xui16>) -> tensor<8x1x16xui16> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<f16>} : () -> tensor<f16>
    %zp = "onnx.Constant"() {value = dense<0> : tensor<ui16>} : () -> tensor<ui16>
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %dq = "onnx.DequantizeLinear"(%x, %scale, %zp)
        : (tensor<8x16xui16>, tensor<f16>, tensor<ui16>) -> tensor<8x16xf16>
    %u = "onnx.Unsqueeze"(%dq, %axes)
        : (tensor<8x16xf16>, tensor<1xi64>) -> tensor<8x1x16xf16>
    %q = "onnx.QuantizeLinear"(%u, %scale, %zp)
        : (tensor<8x1x16xf16>, tensor<f16>, tensor<ui16>) -> tensor<8x1x16xui16>
    return %q : tensor<8x1x16xui16>
  }
}
