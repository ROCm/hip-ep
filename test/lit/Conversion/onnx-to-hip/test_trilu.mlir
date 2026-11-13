// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<3x3xf16>) -> tensor<3x3xf16> {
    return %arg0 : tensor<3x3xf16>
  }

  // ChatGLM form: lower triangle, k = 0, dynamic batch and matrix.
  // CHECK-LABEL: func.func @trilu_lower_dynamic
  // CHECK-NOT: onnx.Trilu
  // CHECK: hip.trilu
  // CHECK-SAME: upper = 0 : si64
  func.func @trilu_lower_dynamic(%input: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %cst = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %output = "onnx.Trilu"(%input, %cst) {upper = 0 : si64} : (tensor<?x?x?xf32>, tensor<i64>) -> tensor<?x?x?xf32>
    return %output : tensor<?x?x?xf32>
  }

  // Upper triangle one diagonal above the main diagonal.
  // CHECK-LABEL: func.func @trilu_upper_k1
  // CHECK-NOT: onnx.Trilu
  // CHECK: hip.trilu
  // CHECK-SAME: k = 1 : si64
  func.func @trilu_upper_k1(%input: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
    %cst = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %output = "onnx.Trilu"(%input, %cst) {upper = 1 : si64} : (tensor<2x4x4xf32>, tensor<i64>) -> tensor<2x4x4xf32>
    return %output : tensor<2x4x4xf32>
  }

  // ChatGLM causal mask: Trilu(ConstantOfShape(1.0)) reads a rank-0 fill
  // instead of materialising the full-size splat.
  // CHECK-LABEL: func.func @trilu_constant_of_shape
  // CHECK-NOT: onnx.ConstantOfShape
  // CHECK-NOT: tensor.splat
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<f32>)
  // CHECK: hip.trilu(%{{.*}}) ins(%[[FILL]] : tensor<f32>) outs(%{{.*}} : tensor<?x?x?xf32>)
  func.func @trilu_constant_of_shape(%shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
    %ones = "onnx.ConstantOfShape"(%shape) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x?x?xf32>
    %cst = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %output = "onnx.Trilu"(%ones, %cst) {upper = 0 : si64} : (tensor<?x?x?xf32>, tensor<i64>) -> tensor<?x?x?xf32>
    return %output : tensor<?x?x?xf32>
  }

  // Omitted k is 0. Default upper stays 1 and is not printed.
  // CHECK-LABEL: func.func @trilu_default
  // CHECK-NOT: onnx.Trilu
  // CHECK: hip.trilu
  // CHECK-NOT: k =
  func.func @trilu_default(%input: tensor<3x3xf16>) -> tensor<3x3xf16> {
    %output = "onnx.Trilu"(%input) : (tensor<3x3xf16>) -> tensor<3x3xf16>
    return %output : tensor<3x3xf16>
  }
}
