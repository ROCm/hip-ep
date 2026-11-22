// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --hip-infer-shapes --canonicalize --cse | FileCheck %s

func.func @main_graph(%a: tensor<f32>) -> tensor<f32> {
  return %a : tensor<f32>
}

// Broadcasting 0 and 1 must produce 0, including a right-aligned rank mismatch.
// CHECK-LABEL: func.func @zero_one
// CHECK: tensor.empty() : tensor<0x4xf32>
// CHECK-NOT: arith.select
// CHECK: hip.add
func.func @zero_one(%a: tensor<0x4xf32>, %b: tensor<1xf32>) -> tensor<?x4xf32> {
  %r = "onnx.Add"(%a, %b) : (tensor<0x4xf32>, tensor<1xf32>) -> tensor<?x4xf32>
  return %r : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @one_zero
// CHECK: tensor.empty() : tensor<0x4xf32>
// CHECK-NOT: arith.select
// CHECK: hip.add
func.func @one_zero(%a: tensor<1xf32>, %b: tensor<0x4xf32>) -> tensor<?x4xf32> {
  %r = "onnx.Add"(%a, %b) : (tensor<1xf32>, tensor<0x4xf32>) -> tensor<?x4xf32>
  return %r : tensor<?x4xf32>
}

// Repeated operands need one dimension query and no broadcast selection.
// CHECK-LABEL: func.func @repeated_operand
// CHECK: %[[DIM:.*]] = tensor.dim
// CHECK-NOT: tensor.dim
// CHECK-NOT: arith.select
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// CHECK: hip.mul{{.*}}outs(%[[INIT]] : tensor<?xf32>)
func.func @repeated_operand(%a: tensor<?xf32>) -> tensor<?xf32> {
  %r = "onnx.Mul"(%a, %a) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// Greater swaps its inputs to use hip.less. The destination stays exact.
// CHECK-LABEL: func.func @greater_swapped
// CHECK-SAME: %[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[AD:.*]] = tensor.dim %[[A]],
// CHECK-DAG: %[[BD:.*]] = tensor.dim %[[B]],
// CHECK: %[[ISONE:.*]] = arith.cmpi eq, %[[BD]], %[[ONE]] : index
// CHECK: %[[SIZE:.*]] = arith.select %[[ISONE]], %[[AD]], %[[BD]] : index
// CHECK: %[[INIT:.*]] = tensor.empty(%[[SIZE]]) : tensor<?xi1>
// CHECK: hip.less(%[[CTX]]) ins(%[[B]], %[[A]] : tensor<?xf32>, tensor<?xf32>) outs(%[[INIT]] : tensor<?xi1>)
func.func @greater_swapped(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xi1> {
  %r = "onnx.Greater"(%a, %b) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
  return %r : tensor<?xi1>
}
