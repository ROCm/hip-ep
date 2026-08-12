// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Transpose to hipsr.transpose. Rejected forms live in
// transpose-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The shape region stays empty here: hipsr.transpose is DPS, so
// hipsr-populate-shape-region has a recipe to dispatch on.
// CHECK-LABEL: func.func @explicit_perm(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<3x?xi64>) -> tensor<?x3xi64> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<3x?xi64>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x3xi64>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.transpose(%[[CTX]]) ins(%[[INPUT]] : tensor<3x?xi64>) outs(%[[INIT]] : tensor<?x3xi64>) {perm = array<i64: 1, 0>} : tensor<?x3xi64>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x3xi64>
func.func @explicit_perm(%ctx: !hipsr.context,
                         %input: tensor<3x?xi64>) -> tensor<?x3xi64> {
  %0 = "onnx.Transpose"(%input) {perm = [1, 0]}
      : (tensor<3x?xi64>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>
}

// -----

// An absent perm means the reverse permutation, which the hipsr op spells out.
// CHECK-LABEL: func.func @default_perm(
// CHECK:         hipsr.transpose(%{{.*}}) ins(%{{.*}} : tensor<2x3x4xf16>) outs(%{{.*}} : tensor<4x3x2xf16>) {perm = array<i64: 2, 1, 0>} : tensor<4x3x2xf16>
func.func @default_perm(%ctx: !hipsr.context,
                        %input: tensor<2x3x4xf16>) -> tensor<4x3x2xf16> {
  %0 = "onnx.Transpose"(%input) : (tensor<2x3x4xf16>) -> tensor<4x3x2xf16>
  return %0 : tensor<4x3x2xf16>
}
