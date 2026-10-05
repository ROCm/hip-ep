// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// com.microsoft QLinearAdd / QLinearMul / QLinearConcat /
// QLinearGlobalAveragePool arrive as onnx.Custom. They decompose into
// DequantizeLinear + the float op + QuantizeLinear before PDLL fusion.
// Per-tensor Add and Mul then fuse to hip.qadd / hip.qmul. Absent zero
// points are onnx.NoValue and fuse as zp 0.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x8xi8>) -> tensor<1x8xi8> {
    return %arg0 : tensor<1x8xi8>
  }

  func.func @qlinear_add(%a: tensor<1x8xi8>, %b: tensor<1x8xi8>) -> tensor<1x8xi8> {
    // CHECK-LABEL: func.func @qlinear_add
    %a_scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %b_scale = "onnx.Constant"() {value = dense<0.05> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<3> : tensor<i8>} : () -> tensor<i8>
    %c_scale = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %c_zp = "onnx.Constant"() {value = dense<7> : tensor<i8>} : () -> tensor<i8>
    %c = "onnx.Custom"(%a, %a_scale, %a_zp, %b, %b_scale, %b_zp, %c_scale, %c_zp) {
      domain_name = "com.microsoft", function_name = "QLinearAdd"
    } : (tensor<1x8xi8>, tensor<f32>, tensor<i8>, tensor<1x8xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<1x8xi8>

    // CHECK: hip.qadd
    // CHECK-SAME: lhs_scale = 1.000000e-01
    // CHECK-SAME: lhs_zp = -5
    // CHECK-SAME: output_scale = 2.000000e-01
    // CHECK-SAME: output_zp = 7
    // CHECK-SAME: rhs_scale = 5.000000e-02
    // CHECK-SAME: rhs_zp = 3
    // CHECK-NOT: onnx.Custom
    return %c : tensor<1x8xi8>
  }

  // Optional zero points are omitted (onnx.NoValue) and mean 0.
  func.func @qlinear_add_default_zp(%a: tensor<4xi8>, %b: tensor<4xi8>) -> tensor<4xi8> {
    // CHECK-LABEL: func.func @qlinear_add_default_zp
    %none = "onnx.NoValue"() {value} : () -> none
    %a_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %b_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %c_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %c = "onnx.Custom"(%a, %a_scale, %none, %b, %b_scale, %none, %c_scale, %none) {
      domain_name = "com.microsoft", function_name = "QLinearAdd"
    } : (tensor<4xi8>, tensor<f32>, none, tensor<4xi8>, tensor<f32>, none, tensor<f32>, none) -> tensor<4xi8>

    // CHECK: hip.qadd
    // CHECK-SAME: lhs_zp = 0
    // CHECK-SAME: output_zp = 0
    // CHECK-SAME: rhs_zp = 0
    // CHECK-NOT: onnx.Custom
    return %c : tensor<4xi8>
  }

  func.func @qlinear_mul(%a: tensor<1x4xui8>, %b: tensor<4xui8>) -> tensor<1x4xui8> {
    // CHECK-LABEL: func.func @qlinear_mul
    %a_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<12> : tensor<ui8>} : () -> tensor<ui8>
    %b_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<9> : tensor<ui8>} : () -> tensor<ui8>
    %c_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %c_zp = "onnx.Constant"() {value = dense<4> : tensor<ui8>} : () -> tensor<ui8>
    %c = "onnx.Custom"(%a, %a_scale, %a_zp, %b, %b_scale, %b_zp, %c_scale, %c_zp) {
      domain_name = "com.microsoft", function_name = "QLinearMul"
    } : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>, tensor<4xui8>, tensor<f32>, tensor<ui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>

    // CHECK: hip.qmul
    // CHECK-SAME: tensor<1x4xui8>, tensor<4xui8>
    // CHECK-SAME: lhs_zp = 12
    // CHECK-SAME: output_zp = 4
    // CHECK-SAME: rhs_zp = 9
    // CHECK-NOT: onnx.Custom
    return %c : tensor<1x4xui8>
  }

  func.func @qlinear_concat(%a: tensor<1x2xui8>, %b: tensor<1x3xui8>) -> tensor<1x5xui8> {
    // CHECK-LABEL: func.func @qlinear_concat
    %y_scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<1> : tensor<ui8>} : () -> tensor<ui8>
    %a_scale = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<2> : tensor<ui8>} : () -> tensor<ui8>
    %b_scale = "onnx.Constant"() {value = dense<0.3> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<3> : tensor<ui8>} : () -> tensor<ui8>
    %y = "onnx.Custom"(%y_scale, %y_zp, %a, %a_scale, %a_zp, %b, %b_scale, %b_zp) {
      axis = 1 : si64, domain_name = "com.microsoft", function_name = "QLinearConcat"
    } : (tensor<f32>, tensor<ui8>, tensor<1x2xui8>, tensor<f32>, tensor<ui8>, tensor<1x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x5xui8>

    // CHECK-NOT: onnx.Custom
    // CHECK-NOT: onnx.Concat
    // CHECK: hip.dequantize_linear
    // CHECK: tensor.insert_slice
    // CHECK: hip.quantize_linear
    return %y : tensor<1x5xui8>
  }

  func.func @qlinear_gap(%x: tensor<1x8x4x4xi8>) -> tensor<1x8x1x1xi8> {
    // CHECK-LABEL: func.func @qlinear_gap
    %x_scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %x_zp = "onnx.Constant"() {value = dense<-2> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.05> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<1> : tensor<i8>} : () -> tensor<i8>
    %y = "onnx.Custom"(%x, %x_scale, %x_zp, %y_scale, %y_zp) {
      channels_last = 0 : si64, domain_name = "com.microsoft", function_name = "QLinearGlobalAveragePool"
    } : (tensor<1x8x4x4xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<1x8x1x1xi8>

    // CHECK-NOT: onnx.Custom
    // CHECK: hip.dequantize_linear
    // CHECK: hip.global_pool
    // CHECK-SAME: mode = 0
    // CHECK: hip.quantize_linear
    return %y : tensor<1x8x1x1xi8>
  }

  // NHWC input. Transpose to NCHW, pool, transpose back, then quantize.
  func.func @qlinear_gap_channels_last(%x: tensor<1x4x4x8xui8>) -> tensor<1x1x1x8xui8> {
    // CHECK-LABEL: func.func @qlinear_gap_channels_last
    %x_scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %x_zp = "onnx.Constant"() {value = dense<2> : tensor<ui8>} : () -> tensor<ui8>
    %y_scale = "onnx.Constant"() {value = dense<0.05> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<1> : tensor<ui8>} : () -> tensor<ui8>
    %y = "onnx.Custom"(%x, %x_scale, %x_zp, %y_scale, %y_zp) {
      channels_last = 1 : si64, domain_name = "com.microsoft", function_name = "QLinearGlobalAveragePool"
    } : (tensor<1x4x4x8xui8>, tensor<f32>, tensor<ui8>, tensor<f32>, tensor<ui8>) -> tensor<1x1x1x8xui8>

    // CHECK-NOT: onnx.Custom
    // CHECK: hip.transpose
    // CHECK: hip.global_pool
    // CHECK: hip.transpose
    // CHECK: hip.quantize_linear
    return %y : tensor<1x1x1x8xui8>
  }
}
