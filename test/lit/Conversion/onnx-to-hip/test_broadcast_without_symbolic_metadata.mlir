// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --canonicalize --cse | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s --check-prefix=RAW

// Independent dynamic operands require the exact broadcast expression.
// Tensor names do not prove equality between their extents.
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[L:.*]] = tensor.dim %arg1,
// CHECK: %[[R:.*]] = tensor.dim %arg2,
// CHECK: %[[UNIT:.*]] = arith.cmpi eq, %[[L]], %[[ONE]] : index
// CHECK: %[[EXTENT:.*]] = arith.select %[[UNIT]], %[[R]], %[[L]] : index
// CHECK: %[[INIT:.*]] = tensor.empty(%[[EXTENT]]) : tensor<?xf32>
// CHECK: hip.add({{.*}}) ins({{.*}}) outs(%[[INIT]] : tensor<?xf32>)
func.func @main_graph(%lhs: tensor<?xf32> {onnx.name = "lhs"},
                      %rhs: tensor<?xf32> {onnx.name = "rhs"}) -> tensor<?xf32> {
  %r = "onnx.Add"(%lhs, %rhs) {node.outputs = ["sum"]}
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// Repeated SSA operands still share their mixed shape without frontend facts.
// RAW-LABEL: func.func @same_value
// RAW-NOT: arith.select
// RAW: tensor.dim
// RAW-NOT: tensor.dim
// RAW-NOT: arith.select
// RAW: return
// CHECK-LABEL: func.func @same_value
// CHECK-NOT: arith.select
// CHECK: tensor.empty
// CHECK-NOT: arith.select
// CHECK: hip.mul
// CHECK-NOT: arith.select
// CHECK: return
func.func @same_value(%x: tensor<?xf32>) -> tensor<?xf32> {
  %r = "onnx.Mul"(%x, %x) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// Where merges all three independent contributors.
// CHECK-LABEL: func.func @three_contributors
// CHECK: arith.select
// CHECK: arith.select
// CHECK: tensor.empty
// CHECK: hip.where
func.func @three_contributors(%cond: tensor<?xi1>, %x: tensor<?xf32>,
                              %y: tensor<?xf32>) -> tensor<?xf32> {
  %r = "onnx.Where"(%cond, %x, %y)
      : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// Rank padding contributes a static one; the aligned dynamic axis still merges.
// CHECK-LABEL: func.func @rank_padding
// CHECK: arith.select
// CHECK: tensor.empty
// CHECK: hip.add
func.func @rank_padding(%x: tensor<?x?xf32>, %y: tensor<?xf32>)
    -> tensor<?x?xf32> {
  %r = "onnx.Add"(%x, %y)
      : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// Zero broadcast with one remains zero, not max(0, 1).
// CHECK-LABEL: func.func @zero_and_one
// CHECK-NOT: arith.select
// CHECK: tensor.empty() : tensor<0xf32>
// CHECK: hip.add
func.func @zero_and_one(%x: tensor<0xf32>, %y: tensor<1xf32>) -> tensor<0xf32> {
  %r = "onnx.Add"(%x, %y) : (tensor<0xf32>, tensor<1xf32>) -> tensor<0xf32>
  return %r : tensor<0xf32>
}
