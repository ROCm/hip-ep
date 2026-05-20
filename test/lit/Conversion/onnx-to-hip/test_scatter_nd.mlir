// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify onnx.ScatterND -> hip.scatter_nd conversion.
// The runtime is a logging-only stub today (no kernel), but the IR shape
// must still be correct so the bufferize / LLVM lowering passes stay
// happy.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // 1-D ScatterND, default reduction ("none"), scalar updates.
  // Note: `reduction = "none"` is the default value of a
  // `DefaultValuedStrAttr` and is therefore elided by MLIR's pretty
  // printer — the absence of the attribute in IR IS the "none" case.
  func.func @test_scatter_nd_1d_default(
      %data: tensor<8xf32>,
      %indices: tensor<4x1xi64>,
      %updates: tensor<4xf32>) -> tensor<8xf32> {
    // CHECK-LABEL: func.func @test_scatter_nd_1d_default
    %r = "onnx.ScatterND"(%data, %indices, %updates)
        : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>

    // CHECK-NOT: onnx.ScatterND
    // CHECK: tensor.empty() : tensor<8xf32>
    // CHECK: hip.scatter_nd({{.*}}) ins(
    // CHECK-SAME: : tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>)
    // CHECK-SAME: outs({{.*}} : tensor<8xf32>)
    // CHECK-NOT: reduction

    return %r : tensor<8xf32>
  }

  // 3-D ScatterND with k=1: each update is a 2-D slice (default reduction
  // is elided — see note in @test_scatter_nd_1d_default above).
  func.func @test_scatter_nd_3d_slice(
      %data: tensor<4x4x4xf32>,
      %indices: tensor<2x1xi64>,
      %updates: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
    // CHECK-LABEL: func.func @test_scatter_nd_3d_slice
    %r = "onnx.ScatterND"(%data, %indices, %updates)
        : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>)
          -> tensor<4x4x4xf32>

    // CHECK-NOT: onnx.ScatterND
    // CHECK: hip.scatter_nd({{.*}}) ins(
    // CHECK-SAME: : tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>)
    // CHECK-SAME: outs({{.*}} : tensor<4x4x4xf32>)
    // CHECK-NOT: reduction

    return %r : tensor<4x4x4xf32>
  }

  // Explicit reduction="add" — must be forwarded verbatim.
  func.func @test_scatter_nd_reduction_add(
      %data: tensor<8xf32>,
      %indices: tensor<4x1xi64>,
      %updates: tensor<4xf32>) -> tensor<8xf32> {
    // CHECK-LABEL: func.func @test_scatter_nd_reduction_add
    %r = "onnx.ScatterND"(%data, %indices, %updates)
        {reduction = "add"}
        : (tensor<8xf32>, tensor<4x1xi64>, tensor<4xf32>) -> tensor<8xf32>

    // CHECK-NOT: onnx.ScatterND
    // CHECK: hip.scatter_nd({{.*}}) ins(
    // CHECK-SAME: outs({{.*}} : tensor<8xf32>)
    // CHECK-SAME: {reduction = "add"}

    return %r : tensor<8xf32>
  }

  // reduction="max" works too — exercises another enum value.
  func.func @test_scatter_nd_reduction_max(
      %data: tensor<4x4xf32>,
      %indices: tensor<2x2xi64>,
      %updates: tensor<2xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: func.func @test_scatter_nd_reduction_max
    %r = "onnx.ScatterND"(%data, %indices, %updates)
        {reduction = "max"}
        : (tensor<4x4xf32>, tensor<2x2xi64>, tensor<2xf32>) -> tensor<4x4xf32>

    // CHECK-NOT: onnx.ScatterND
    // CHECK: hip.scatter_nd({{.*}}) ins(
    // CHECK-SAME: outs({{.*}} : tensor<4x4xf32>)
    // CHECK-SAME: {reduction = "max"}

    return %r : tensor<4x4xf32>
  }

  // Dynamic data shape (output.rank == data.rank invariant): every
  // dynamic output dim is sourced from the matching data dim via
  // tensor.dim.
  func.func @test_scatter_nd_dyn_data(
      %data: tensor<?x?xf32>,
      %indices: tensor<2x2xi64>,
      %updates: tensor<2xf32>) -> tensor<?x?xf32> {
    %r = "onnx.ScatterND"(%data, %indices, %updates)
        : (tensor<?x?xf32>, tensor<2x2xi64>, tensor<2xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @test_scatter_nd_dyn_data
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<?x?xf32>,
  // CHECK-DAG: %[[A0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[D]], %[[A0]] : tensor<?x?xf32>
  // CHECK-DAG: %[[A1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[D1:.*]] = tensor.dim %[[D]], %[[A1]] : tensor<?x?xf32>
  // CHECK: tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32>
  // CHECK: hip.scatter_nd({{.*}}) ins({{.*}}, {{.*}}, {{.*}} : tensor<?x?xf32>, tensor<2x2xi64>, tensor<2xf32>) outs({{.*}} : tensor<?x?xf32>)
}
