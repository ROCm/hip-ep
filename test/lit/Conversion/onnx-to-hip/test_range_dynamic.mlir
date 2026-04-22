// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Range with non-constant scalars lowers to scf/tensor/arith.
//===----------------------------------------------------------------------===//
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf16>) -> tensor<1xf16> {
    return %arg0 : tensor<1xf16>
  }

  func.func @test_range_dynamic(
      %start: tensor<i64>,
      %limit: tensor<i64>,
      %delta: tensor<i64>) -> tensor<?xi64> {
    %0 = "onnx.Range"(%start, %limit, %delta)
        : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    return %0 : tensor<?xi64>
  }
}

// CHECK-LABEL: func.func @test_range_dynamic
// CHECK-NOT: onnx.Range
// CHECK-DAG: tensor.extract
// CHECK-DAG: arith.sitofp
// CHECK-DAG: arith.subf
// CHECK-DAG: arith.divf
// CHECK-DAG: math.ceil
// CHECK-DAG: arith.fptosi
// CHECK-DAG: tensor.empty
// CHECK: scf.for
// CHECK: tensor.insert
