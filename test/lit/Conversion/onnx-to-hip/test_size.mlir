// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Size compile-time handling.
//
// Verifies:
// - Static-ranked input is folded to scalar tensor constant IR.
// - Dynamic input dimension is not folded and onnx.Size remains.
//===----------------------------------------------------------------------===//
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x3x4xf16>) -> tensor<2x3x4xf16> {
    return %arg0 : tensor<2x3x4xf16>
  }

  func.func @test_size_static(%data: tensor<2x3x4xf16>) -> tensor<i64> {
    %0 = "onnx.Size"(%data) : (tensor<2x3x4xf16>) -> tensor<i64>
    return %0 : tensor<i64>
  }

  func.func @test_size_dynamic(%data: tensor<?x3x4xf16>) -> tensor<i64> {
    %0 = "onnx.Size"(%data) : (tensor<?x3x4xf16>) -> tensor<i64>
    return %0 : tensor<i64>
  }
}

// CHECK-LABEL: func.func @test_size_static
// CHECK-NOT: onnx.Size
// CHECK: arith.constant 24 : i64
// CHECK: tensor.from_elements

// CHECK-LABEL: func.func @test_size_dynamic
// CHECK: "onnx.Size"
