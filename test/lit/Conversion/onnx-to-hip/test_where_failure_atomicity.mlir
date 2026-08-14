// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

// A static imported result extent that contradicts the statically inferred
// broadcast shape must fail before broadcast reification emits shape IR.
module {
  func.func @main_graph(
      %cond: tensor<4xi1>, %x: tensor<4xf32>, %y: tensor<4xf32>)
      -> tensor<7xf32> {
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: tensor.dim
    // CHECK-NOT: arith.cmpi
    // CHECK-NOT: arith.select
    // CHECK: "onnx.Where"
    %result = "onnx.Where"(%cond, %x, %y) :
        (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<7xf32>
    return %result : tensor<7xf32>
  }

  // A contradictory imported rank is rejected by the same pure check.
  func.func @where_result_rank_mismatch(
      %cond: tensor<?xi1>, %x: tensor<?xf32>, %y: tensor<?xf32>)
      -> tensor<?x?xf32> {
    // CHECK-LABEL: func.func @where_result_rank_mismatch
    // CHECK-NOT: tensor.dim
    // CHECK-NOT: arith.cmpi
    // CHECK-NOT: arith.select
    // CHECK: "onnx.Where"
    %result = "onnx.Where"(%cond, %x, %y) :
        (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}
