// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Equal to hipsr.equal. Rejected forms live in
// equal-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The shape region stays empty here: hipsr.equal is DPS, so
// hipsr-populate-shape-region has a recipe to dispatch on.
// CHECK-LABEL: func.func @broadcast_scalar(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context, %[[IDS:.*]]: tensor<?x?xi64>,
// CHECK-SAME:    %[[TOKEN:.*]]: tensor<i64>) -> tensor<?x?xui8> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[IDS]], %[[TOKEN]] : tensor<?x?xi64>, tensor<i64>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?xui8>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.equal(%[[CTX]]) ins(%[[IDS]], %[[TOKEN]] : tensor<?x?xi64>, tensor<i64>) outs(%[[INIT]] : tensor<?x?xui8>) : tensor<?x?xui8>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?xui8>
func.func @broadcast_scalar(%ctx: !hipsr.context, %ids: tensor<?x?xi64>,
                            %token: tensor<i64>) -> tensor<?x?xui8> {
  %0 = "onnx.Equal"(%ids, %token)
      : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xui8>
  return %0 : tensor<?x?xui8>
}

// -----

// An i1 mask is accepted alongside the ui8 spelling.
// CHECK-LABEL: func.func @same_shape(
// CHECK:         hipsr.equal(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<2x3xf16>, tensor<2x3xf16>) outs(%{{.*}} : tensor<2x3xi1>) : tensor<2x3xi1>
func.func @same_shape(%ctx: !hipsr.context, %lhs: tensor<2x3xf16>,
                      %rhs: tensor<2x3xf16>) -> tensor<2x3xi1> {
  %0 = "onnx.Equal"(%lhs, %rhs)
      : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xi1>
  return %0 : tensor<2x3xi1>
}
