// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ LpNormalization fusion (UINT16 L2 / RMS)
//
// Pattern fuses:
//   onnx.DequantizeLinear -> onnx.LpNormalization -> onnx.QuantizeLinear
// into:
//   hip.qlpnormalization
//
// Scales and zero points are deliberately different on the DQ vs Q sides so
// the checks prove both pairs fold into attributes (requant is required).
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {

// CHECK-LABEL: func.func @main_graph
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<1x128x2048xui16>
// CHECK-NEXT:    %[[QLP:.*]] = hip.qlpnormalization(%{{.*}}) ins(%{{.*}} : tensor<1x128x2048xui16>) outs(%[[EMPTY]] : tensor<1x128x2048xui16>) {axis = -1 : i64, input_scale = 9.99999974E-6 : f32, input_zp = 35189 : i64, output_scale = 2.50000012E-5 : f32, output_zp = 35274 : i64, p = 2 : i64} : tensor<1x128x2048xui16>
// CHECK-NEXT:    return %[[QLP]] : tensor<1x128x2048xui16>
  func.func @main_graph(%x: tensor<1x128x2048xui16>) -> tensor<1x128x2048xui16> {
    %in_s = "onnx.Constant"() {value = dense<9.99999975E-6> : tensor<f32>} : () -> tensor<f32>
    %in_z = "onnx.Constant"() {value = dense<35189> : tensor<ui16>} : () -> tensor<ui16>
    %out_s = "onnx.Constant"() {value = dense<2.50000004E-5> : tensor<f32>} : () -> tensor<f32>
    %out_z = "onnx.Constant"() {value = dense<35274> : tensor<ui16>} : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %in_s, %in_z)
        : (tensor<1x128x2048xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x2048xf32>
    %n = "onnx.LpNormalization"(%dq) {axis = -1 : si64, p = 2 : si64}
        : (tensor<1x128x2048xf32>) -> tensor<1x128x2048xf32>
    %q = "onnx.QuantizeLinear"(%n, %out_s, %out_z)
        : (tensor<1x128x2048xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x2048xui16>
    return %q : tensor<1x128x2048xui16>
  }

// CHECK-LABEL: func.func @custom_qdq_lpnorm
// CHECK:       hip.qlpnormalization
// CHECK-SAME:  input_scale = 3.16281967E-6 : f32
// CHECK-SAME:  input_zp = 33443 : i64
// CHECK-SAME:  output_scale = 2.33214701E-6 : f32
// CHECK-SAME:  output_zp = 33422 : i64
// CHECK-SAME:  p = 2 : i64
  func.func @custom_qdq_lpnorm(%x: tensor<1x1x8x32xui16>) -> tensor<1x1x8x32xui16> {
    %in_s = "onnx.Constant"() {value = dense<3.16281967E-6> : tensor<f32>} : () -> tensor<f32>
    %in_z = "onnx.Constant"() {value = dense<33443> : tensor<ui16>} : () -> tensor<ui16>
    %out_s = "onnx.Constant"() {value = dense<2.33214700E-6> : tensor<f32>} : () -> tensor<f32>
    %out_z = "onnx.Constant"() {value = dense<33422> : tensor<ui16>} : () -> tensor<ui16>
    %dq = "onnx.Custom"(%x, %in_s, %in_z) {
      domain_name = "com.microsoft", function_name = "DequantizeLinear"
    } : (tensor<1x1x8x32xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x8x32xf32>
    %n = "onnx.LpNormalization"(%dq) {axis = -1 : si64, p = 2 : si64}
        : (tensor<1x1x8x32xf32>) -> tensor<1x1x8x32xf32>
    %q = "onnx.Custom"(%n, %out_s, %out_z) {
      domain_name = "com.microsoft", function_name = "QuantizeLinear"
    } : (tensor<1x1x8x32xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x8x32xui16>
    return %q : tensor<1x1x8x32xui16>
  }

// p=1 is not the fused RMS path.
// CHECK-LABEL: func.func @lpnorm_p1
// CHECK-NOT:   hip.qlpnormalization
  func.func @lpnorm_p1(%x: tensor<1x8x16xui16>) -> tensor<1x8x16xui16> {
    %in_s = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %in_z = "onnx.Constant"() {value = dense<0> : tensor<ui16>} : () -> tensor<ui16>
    %out_s = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %out_z = "onnx.Constant"() {value = dense<1> : tensor<ui16>} : () -> tensor<ui16>
    %dq = "onnx.DequantizeLinear"(%x, %in_s, %in_z)
        : (tensor<1x8x16xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x8x16xf32>
    %n = "onnx.LpNormalization"(%dq) {axis = -1 : si64, p = 1 : si64}
        : (tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
    %q = "onnx.QuantizeLinear"(%n, %out_s, %out_z)
        : (tensor<1x8x16xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x8x16xui16>
    return %q : tensor<1x8x16xui16>
  }
}
