// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Range static fold for f32 scalars (ONNX Range type T).
//===----------------------------------------------------------------------===//
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
    return %arg0 : tensor<1xf16>
  }

  func.func @test_range_fold_f32() -> tensor<3xf32> {
    %start = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %limit = "onnx.Constant"() {value = dense<1.5> : tensor<f32>} : () -> tensor<f32>
    %delta = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %0 = "onnx.Range"(%start, %limit, %delta)
        : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

// CHECK-LABEL: func.func @test_range_fold_f32
// CHECK-NOT: onnx.Range
// CHECK: arith.constant
// CHECK: tensor.from_elements
