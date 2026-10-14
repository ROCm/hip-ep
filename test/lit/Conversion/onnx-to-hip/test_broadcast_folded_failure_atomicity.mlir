// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-onnx-to-hip | FileCheck %s --implicit-check-not=hip.min --implicit-check-not=hip.max --implicit-check-not=hip.where --implicit-check-not=tensor.dim --implicit-check-not=arith.select --implicit-check-not=arith.cmpi --implicit-check-not=tensor.empty

// Dynamic slice types hide constant extents from the type-only check.
// A failed destination check must leave no shape operations or partial chain.
func.func @main_graph(%ctx: !hip.context, %c: tensor<?xi1>, %x: tensor<?xf32>, %y: tensor<?xf32>) -> tensor<3xf32> {
  %two = arith.constant 2 : index
  %a = tensor.extract_slice %x[0] [%two] [1] : tensor<?xf32> to tensor<?xf32>
  %b = tensor.extract_slice %y[0] [%two] [1] : tensor<?xf32> to tensor<?xf32>
  %cond = tensor.extract_slice %c[0] [%two] [1] : tensor<?xi1> to tensor<?xi1>
  // CHECK-LABEL: func.func @main_graph
  // CHECK: "onnx.Where"
  // CHECK: return
  %r = "onnx.Where"(%cond, %a, %b) : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<3xf32>
  return %r : tensor<3xf32>
}

// A conflict on the second axis must discard queries and a merge for the first.
// CHECK-LABEL: func.func @where_late_axis_result_conflict
// CHECK: "onnx.Where"
// CHECK: return
func.func @where_late_axis_result_conflict(%ctx: !hip.context, %m: index, %n: index, %cond: tensor<?x?xi1>, %x: tensor<?x?xf32>, %y: tensor<?x?xf32>) -> tensor<?x3xf32> {
  %two = arith.constant 2 : index
  %a = tensor.extract_slice %x[0, 0] [%m, %two] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %b = tensor.extract_slice %y[0, 0] [%n, %two] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %c = tensor.extract_slice %cond[0, 0] [%m, %two] [1, 1] : tensor<?x?xi1> to tensor<?x?xi1>
  %r = "onnx.Where"(%c, %a, %b) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x3xf32>
  return %r : tensor<?x3xf32>
}

// The final result contradicts a folded extent of the last operand.
// The first pair is valid, but it must not remain after the final pair fails.
// CHECK-LABEL: func.func @late_min_conflict
// CHECK: "onnx.Min"
// CHECK: return
func.func @late_min_conflict(%ctx: !hip.context, %a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>) -> tensor<3xf32> {
  %two = arith.constant 2 : index
  %last = tensor.extract_slice %c[0] [%two] [1] : tensor<?xf32> to tensor<?xf32>
  %r = "onnx.Min"(%a, %b, %last) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<3xf32>
  return %r : tensor<3xf32>
}

// CHECK-LABEL: func.func @late_max_conflict
// CHECK: "onnx.Max"
// CHECK: return
func.func @late_max_conflict(%ctx: !hip.context, %a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>) -> tensor<3xf32> {
  %two = arith.constant 2 : index
  %last = tensor.extract_slice %c[0] [%two] [1] : tensor<?xf32> to tensor<?xf32>
  %r = "onnx.Max"(%a, %b, %last) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<3xf32>
  return %r : tensor<3xf32>
}
