// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Pre-lowering attention score prep optimizations:
//  - MatMul -> Mul(scale) folds into Mul(A,scale) -> MatMul
//  - broadcast Add(constant bias) expands the constant to same-shape Add

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
    return %arg0 : tensor<1xf16>
  }

  func.func @test_fold_matmul_scale(
      %a: tensor<2x3x4x8xf16>, %b: tensor<2x3x8x4xf16>)
      -> tensor<2x3x4x4xf16> {
    %scale = "onnx.Constant"() {
      value = dense<[[[2.000000e+00]], [[4.000000e+00]], [[8.000000e+00]]]> :
        tensor<3x1x1xf16>
    } : () -> tensor<3x1x1xf16>
    %scores = "onnx.MatMul"(%a, %b)
        : (tensor<2x3x4x8xf16>, tensor<2x3x8x4xf16>) -> tensor<2x3x4x4xf16>
    %out = "onnx.Mul"(%scores, %scale)
        : (tensor<2x3x4x4xf16>, tensor<3x1x1xf16>) -> tensor<2x3x4x4xf16>
    return %out : tensor<2x3x4x4xf16>
  }

  func.func @test_expand_constant_bias(
      %act: tensor<2x3x4x4xf16>) -> tensor<2x3x4x4xf16> {
    %bias = "onnx.Constant"() {
      value = dense<1.0> : tensor<1x3x4x4xf16>
    } : () -> tensor<1x3x4x4xf16>
    %out = "onnx.Add"(%act, %bias)
        : (tensor<2x3x4x4xf16>, tensor<1x3x4x4xf16>) -> tensor<2x3x4x4xf16>
    return %out : tensor<2x3x4x4xf16>
  }
}

// MatMul->Mul(scale) should disappear; one Mul on A remains, then hip.matmul.
// CHECK-LABEL: func.func @test_fold_matmul_scale
// CHECK-NOT: onnx.MatMul{{.*}}onnx.Mul
// CHECK: hip.mul
// CHECK: hip.matmul

// Broadcast bias constant should expand to 2x3x4x4 before hip.add.
// CHECK-LABEL: func.func @test_expand_constant_bias
// CHECK: "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2x3x4x4xf16>}
// CHECK: hip.add
