// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @equal_f32(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @equal_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xf32>, %[[B:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.equal(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)

  func.func @equal_i64(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<4xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @equal_i64
  // CHECK: hip.equal({{.*}}) ins({{.*}}, {{.*}} : tensor<4xi64>, tensor<4xi64>) outs({{.*}} : tensor<4xi1>)

  func.func @equal_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
    return %result : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @equal_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[A3:.*]]: tensor<?x?xf32>, %[[B3:.*]]: tensor<?x?xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: tensor.dim
  // CHECK: tensor.dim
  // CHECK: tensor.empty({{.*}}) : tensor<?x?xi1>
  // CHECK: hip.equal(%[[CTX3]]) ins(%[[A3]], %[[B3]] : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xi1>)

  // Regression: Equal(dyn-rank-2, rank-0 scalar) MUST broadcast the
  // scalar to the dynamic result shape via hip.expand. Before Phase 2a
  // this skipped broadcastToShape entirely (resultIsStatic guard) and
  // the kernel read past the 1-element scalar buffer at runtime.
  // The expected pattern: tensor.dim(%a, 0/1) -> arith.index_cast ->
  // tensor.from_elements -> hip.expand -> hip.equal.
  func.func @equal_dyn_broadcast_scalar(%a: tensor<?x?xi64>, %b: tensor<i64>) -> tensor<?x?xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi1>
    return %result : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @equal_dyn_broadcast_scalar
  // CHECK-SAME: (%[[CTX4:.*]]: !hip.context, %[[A4:.*]]: tensor<?x?xi64>, %[[B4:.*]]: tensor<i64>)
  // The wide operand %A4 must be the dim source so SSA dominance is
  // preserved (tensor.dim ops created at the user op's location).
  // CHECK-DAG: tensor.dim %[[A4]]
  // CHECK-DAG: tensor.dim %[[A4]]
  // CHECK: tensor.from_elements
  // CHECK: hip.expand(%[[CTX4]]) ins(%[[B4]]
  // CHECK: hip.equal(%[[CTX4]]) ins(%[[A4]],

  // Regression: Equal(rank-0 scalar, dyn-rank-2) — same as above but
  // with the operands swapped. The wide-side detection must pick %b,
  // not blindly take operand 0.
  func.func @equal_dyn_broadcast_scalar_swapped(%a: tensor<i64>, %b: tensor<?x?xi64>) -> tensor<?x?xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<i64>, tensor<?x?xi64>) -> tensor<?x?xi1>
    return %result : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @equal_dyn_broadcast_scalar_swapped
  // CHECK-SAME: (%[[CTX5:.*]]: !hip.context, %[[A5:.*]]: tensor<i64>, %[[B5:.*]]: tensor<?x?xi64>)
  // CHECK-DAG: tensor.dim %[[B5]]
  // CHECK-DAG: tensor.dim %[[B5]]
  // CHECK: tensor.from_elements
  // CHECK: hip.expand(%[[CTX5]]) ins(%[[A5]]
  // CHECK: hip.equal(%[[CTX5]]) ins({{.*}}, %[[B5]]
}
