// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Expand to a placeholder-backed hipsr.expand. Rejected forms
// live in expand-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The requested extents are read at runtime, so the init is a barrier
// placeholder over both the input and the shape operand.
// CHECK-LABEL: func.func @expand(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<2xi64>) -> tensor<?x?xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<?x3xf16>, tensor<2xi64>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x?xf16>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<?x3xf16>, tensor<2xi64>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?xf16>
//
// The conversion leaves the region empty; hipsr-populate-shape-region fills it.
// CHECK-NOT:     shape_region
func.func @expand(%ctx: !hipsr.context, %input: tensor<?x3xf16>,
                  %shape: tensor<2xi64>) -> tensor<?x?xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<?x3xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// A shape longer than the input rank raises the output rank, and the leading
// extents come from the shape operand alone.
// CHECK-LABEL: func.func @expand_broadcast_rank(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<4xi64>) -> tensor<?x?x?x?xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<2x3xf16>, tensor<4xi64>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x?x?x?xf16>
// CHECK-NEXT:    hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<2x3xf16>, tensor<4xi64>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x?x?x?xf16>) : tensor<?x?x?x?xf16>
func.func @expand_broadcast_rank(%ctx: !hipsr.context,
                                 %input: tensor<2x3xf16>,
                                 %shape: tensor<4xi64>)
    -> tensor<?x?x?x?xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<4xi64>) -> tensor<?x?x?x?xf16>
  return %0 : tensor<?x?x?x?xf16>
}
