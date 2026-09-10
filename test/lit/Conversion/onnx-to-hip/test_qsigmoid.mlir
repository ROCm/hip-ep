// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ Sigmoid fusion (UINT16)
//
// Pattern fuses:
//   onnx.DequantizeLinear -> onnx.Sigmoid -> onnx.QuantizeLinear
// into:
//   hip.qsigmoid
//
// Scales and zero points are deliberately different on the DQ vs Q sides so
// the checks prove both pairs fold into attributes (requant is required).
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {

// CHECK-LABEL: func.func @main_graph
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<1x128x2048xui16>
// CHECK-NEXT:    %[[QSIG:.*]] = hip.qsigmoid(%{{.*}}) ins(%{{.*}} : tensor<1x128x2048xui16>) outs(%[[EMPTY]] : tensor<1x128x2048xui16>) {input_scale = 1.000000e-01 : f32, input_zp = 35189 : i64, output_scale = 2.000000e-01 : f32, output_zp = 35274 : i64} : tensor<1x128x2048xui16>
// CHECK-NEXT:    return %[[QSIG]] : tensor<1x128x2048xui16>
  func.func @main_graph(%x: tensor<1x128x2048xui16>) -> tensor<1x128x2048xui16> {
    %in_s = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %in_z = "onnx.Constant"() {value = dense<35189> : tensor<ui16>} : () -> tensor<ui16>
    %out_s = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %out_z = "onnx.Constant"() {value = dense<35274> : tensor<ui16>} : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %in_s, %in_z)
        : (tensor<1x128x2048xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x2048xf32>
    %s = "onnx.Sigmoid"(%dq) : (tensor<1x128x2048xf32>) -> tensor<1x128x2048xf32>
    %q = "onnx.QuantizeLinear"(%s, %out_s, %out_z)
        : (tensor<1x128x2048xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x2048xui16>
    return %q : tensor<1x128x2048xui16>
  }

// CHECK-LABEL: func.func @custom_qdq_sigmoid
// CHECK:       hip.qsigmoid
// CHECK-SAME:  input_scale = 1.000000e-01 : f32
// CHECK-SAME:  input_zp = 33443 : i64
// CHECK-SAME:  output_scale = 2.000000e-01 : f32
// CHECK-SAME:  output_zp = 33422 : i64
  func.func @custom_qdq_sigmoid(%x: tensor<1x1x8x32xui16>) -> tensor<1x1x8x32xui16> {
    %in_s = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %in_z = "onnx.Constant"() {value = dense<33443> : tensor<ui16>} : () -> tensor<ui16>
    %out_s = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %out_z = "onnx.Constant"() {value = dense<33422> : tensor<ui16>} : () -> tensor<ui16>
    %dq = "onnx.Custom"(%x, %in_s, %in_z) {
      domain_name = "com.microsoft", function_name = "DequantizeLinear"
    } : (tensor<1x1x8x32xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x8x32xf32>
    %s = "onnx.Sigmoid"(%dq) : (tensor<1x1x8x32xf32>) -> tensor<1x1x8x32xf32>
    %q = "onnx.Custom"(%s, %out_s, %out_z) {
      domain_name = "com.microsoft", function_name = "QuantizeLinear"
    } : (tensor<1x1x8x32xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x8x32xui16>
    return %q : tensor<1x1x8x32xui16>
  }

// INT8 sandwiches stay unfused: the fused wrap is UINT16-only.
// CHECK-LABEL: func.func @sigmoid_i8
// CHECK-NOT:   hip.qsigmoid
  func.func @sigmoid_i8(%x: tensor<1x8x16xi8>) -> tensor<1x8x16xi8> {
    %in_s = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %in_z = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %out_s = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %out_z = "onnx.Constant"() {value = dense<1> : tensor<i8>} : () -> tensor<i8>
    %dq = "onnx.DequantizeLinear"(%x, %in_s, %in_z)
        : (tensor<1x8x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x8x16xf32>
    %s = "onnx.Sigmoid"(%dq) : (tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
    %q = "onnx.QuantizeLinear"(%s, %out_s, %out_z)
        : (tensor<1x8x16xf32>, tensor<f32>, tensor<i8>) -> tensor<1x8x16xi8>
    return %q : tensor<1x8x16xi8>
  }
}
