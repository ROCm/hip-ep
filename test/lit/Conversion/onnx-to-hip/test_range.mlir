// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Constant-folded integer Range (numpy.arange semantics).
  func.func @test_range_i32_fold() -> tensor<4xi32> {
    %s = arith.constant dense<2> : tensor<i32>
    %l = arith.constant dense<10> : tensor<i32>
    %d = arith.constant dense<2> : tensor<i32>
    %r = "onnx.Range"(%s, %l, %d) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  // CHECK-LABEL: func.func @test_range_i32_fold
  // CHECK-SAME: !hip.context
  // CHECK: arith.constant dense<[2, 4, 6, 8]> : tensor<4xi32>
  // CHECK: return

  // Dynamic operands lower to tensor.empty + scf.for.
  func.func @test_range_i32_dynamic(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?xi32> {
    %r = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
    return %r : tensor<?xi32>
  }
  // CHECK-LABEL: func.func @test_range_i32_dynamic
  // CHECK: tensor.empty
  // CHECK: scf.for
}
