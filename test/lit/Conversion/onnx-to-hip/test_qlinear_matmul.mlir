// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// onnx.QLinearMatMul -> hip.qlinear_matmul
//
// Covers the MobileNetV3 int8 form: uint8 [1, K] times int8 [K, N], scalar
// scales and zero points. Rank other than 2 and per-column scales stay
// onnx.QLinearMatMul.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }

  // CHECK-LABEL: func.func @qlinear_matmul_row
  // CHECK: hip.qlinear_matmul
  // CHECK-SAME: tensor<1x4xui8>
  // CHECK-SAME: tensor<4x3xi8>
  // CHECK-SAME: tensor<1x3xui8>
  // CHECK-NOT: onnx.QLinearMatMul
  func.func @qlinear_matmul_row(
      %a: tensor<1x4xui8>, %as: tensor<f32>, %az: tensor<ui8>,
      %b: tensor<4x3xi8>, %bs: tensor<f32>, %bz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>) -> tensor<1x3xui8> {
    %y = "onnx.QLinearMatMul"(%a, %as, %az, %b, %bs, %bz, %ys, %yz)
        : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>, tensor<4x3xi8>,
           tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>) -> tensor<1x3xui8>
    return %y : tensor<1x3xui8>
  }

  // CHECK-LABEL: func.func @qlinear_matmul_matrix
  // CHECK: hip.qlinear_matmul
  // CHECK-NOT: onnx.QLinearMatMul
  func.func @qlinear_matmul_matrix(
      %a: tensor<2x4xui8>, %as: tensor<f32>, %az: tensor<ui8>,
      %b: tensor<4x3xi8>, %bs: tensor<f32>, %bz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>) -> tensor<2x3xui8> {
    %y = "onnx.QLinearMatMul"(%a, %as, %az, %b, %bs, %bz, %ys, %yz)
        : (tensor<2x4xui8>, tensor<f32>, tensor<ui8>, tensor<4x3xi8>,
           tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>) -> tensor<2x3xui8>
    return %y : tensor<2x3xui8>
  }

  // CHECK-LABEL: func.func @qlinear_matmul_rank3_rejected
  // CHECK: onnx.QLinearMatMul
  // CHECK-NOT: hip.qlinear_matmul
  func.func @qlinear_matmul_rank3_rejected(
      %a: tensor<1x2x4xui8>, %as: tensor<f32>, %az: tensor<ui8>,
      %b: tensor<1x4x3xi8>, %bs: tensor<f32>, %bz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>) -> tensor<1x2x3xui8> {
    %y = "onnx.QLinearMatMul"(%a, %as, %az, %b, %bs, %bz, %ys, %yz)
        : (tensor<1x2x4xui8>, tensor<f32>, tensor<ui8>, tensor<1x4x3xi8>,
           tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>) -> tensor<1x2x3xui8>
    return %y : tensor<1x2x3xui8>
  }

  // CHECK-LABEL: func.func @qlinear_matmul_per_column_rejected
  // CHECK: onnx.QLinearMatMul
  // CHECK-NOT: hip.qlinear_matmul
  func.func @qlinear_matmul_per_column_rejected(
      %a: tensor<1x4xui8>, %as: tensor<f32>, %az: tensor<ui8>,
      %b: tensor<4x3xi8>, %bs: tensor<3xf32>, %bz: tensor<3xi8>,
      %ys: tensor<f32>, %yz: tensor<ui8>) -> tensor<1x3xui8> {
    %y = "onnx.QLinearMatMul"(%a, %as, %az, %b, %bs, %bz, %ys, %yz)
        : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>, tensor<4x3xi8>,
           tensor<3xf32>, tensor<3xi8>, tensor<f32>, tensor<ui8>) -> tensor<1x3xui8>
    return %y : tensor<1x3xui8>
  }
}
