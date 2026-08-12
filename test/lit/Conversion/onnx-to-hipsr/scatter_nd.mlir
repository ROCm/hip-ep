// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.ScatterND to hipsr.scatter_nd. Rejected forms live in
// scatter_nd-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The output takes the data's shape, so the data alone reaches the
// placeholder. Its shape region stays empty: hipsr.scatter_nd is DPS, so
// hipsr-populate-shape-region has a recipe to dispatch on.
// CHECK-LABEL: func.func @scatter_elements(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context, %[[DATA:.*]]: tensor<?x?x4096xf16>,
// CHECK-SAME:    %[[IDS:.*]]: tensor<?x3xi64>, %[[UPDATES:.*]]: tensor<?xf16>) -> tensor<?x?x4096xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]] : tensor<?x?x4096xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.scatter_nd(%[[CTX]]) ins(%[[DATA]], %[[IDS]], %[[UPDATES]] : tensor<?x?x4096xf16>, tensor<?x3xi64>, tensor<?xf16>) outs(%[[INIT]] : tensor<?x?x4096xf16>) : tensor<?x?x4096xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x4096xf16>
func.func @scatter_elements(%ctx: !hipsr.context, %data: tensor<?x?x4096xf16>,
                            %ids: tensor<?x3xi64>,
                            %updates: tensor<?xf16>) -> tensor<?x?x4096xf16> {
  %0 = "onnx.ScatterND"(%data, %ids, %updates) {reduction = "none"}
      : (tensor<?x?x4096xf16>, tensor<?x3xi64>, tensor<?xf16>)
      -> tensor<?x?x4096xf16>
  return %0 : tensor<?x?x4096xf16>
}

// -----

// ONNX defaults the reduction to overwriting, so an absent attribute converts
// like an explicit one.
// CHECK-LABEL: func.func @default_reduction(
// CHECK:         hipsr.scatter_nd(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8x2xf16>, tensor<5x1xi64>, tensor<5x8x2xf16>) outs(%{{.*}} : tensor<4x8x2xf16>) : tensor<4x8x2xf16>
func.func @default_reduction(%ctx: !hipsr.context, %data: tensor<4x8x2xf16>,
                             %ids: tensor<5x1xi64>,
                             %updates: tensor<5x8x2xf16>) -> tensor<4x8x2xf16> {
  %0 = "onnx.ScatterND"(%data, %ids, %updates)
      : (tensor<4x8x2xf16>, tensor<5x1xi64>, tensor<5x8x2xf16>)
      -> tensor<4x8x2xf16>
  return %0 : tensor<4x8x2xf16>
}
