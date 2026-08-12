// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Gather to hipsr.gather. Rejected forms live in
// gather-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The shape region stays empty here: hipsr.gather is DPS, so
// hipsr-populate-shape-region has a recipe to dispatch on.
// CHECK-LABEL: func.func @embedding_lookup(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context, %[[TABLE:.*]]: tensor<8x4096xf16>,
// CHECK-SAME:    %[[IDS:.*]]: tensor<?x?xi64>) -> tensor<?x?x4096xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4096xf16>, tensor<?x?xi64>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.gather(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4096xf16>, tensor<?x?xi64>) outs(%[[INIT]] : tensor<?x?x4096xf16>) {axis = 0 : i64} : tensor<?x?x4096xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x4096xf16>
func.func @embedding_lookup(%ctx: !hipsr.context, %table: tensor<8x4096xf16>,
                            %ids: tensor<?x?xi64>) -> tensor<?x?x4096xf16> {
  %0 = "onnx.Gather"(%table, %ids) {axis = 0 : si64}
      : (tensor<8x4096xf16>, tensor<?x?xi64>) -> tensor<?x?x4096xf16>
  return %0 : tensor<?x?x4096xf16>
}

// -----

// Reading one extent out of an onnx.Shape result converts like any other
// gather; where that value lives is a memory-space question, not a
// legalization one.
// CHECK-LABEL: func.func @shape_extent(
// CHECK:         hipsr.gather(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<2xi64>, tensor<i64>) outs(%{{.*}} : tensor<i64>) {axis = 0 : i64} : tensor<i64>
func.func @shape_extent(%ctx: !hipsr.context, %shape: tensor<2xi64>,
                        %index: tensor<i64>) -> tensor<i64> {
  %0 = "onnx.Gather"(%shape, %index) {axis = 0 : si64}
      : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}

// -----

// A negative axis counts back from the end and reaches the dialect normalized.
// CHECK-LABEL: func.func @negative_axis(
// CHECK:         hipsr.gather(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<2x3x4xf16>, tensor<5xi64>) outs(%{{.*}} : tensor<2x3x5xf16>) {axis = 2 : i64} : tensor<2x3x5xf16>
func.func @negative_axis(%ctx: !hipsr.context, %data: tensor<2x3x4xf16>,
                         %indices: tensor<5xi64>) -> tensor<2x3x5xf16> {
  %0 = "onnx.Gather"(%data, %indices) {axis = -1 : si64}
      : (tensor<2x3x4xf16>, tensor<5xi64>) -> tensor<2x3x5xf16>
  return %0 : tensor<2x3x5xf16>
}
