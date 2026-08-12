// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Unsqueeze to a hipsr.compute wrapping tensor.expand_shape,
// with the placeholder's shape region populated inline. Rejected forms live in
// unsqueeze-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The axes operand goes unread: matching the result extents against the input
// extents in order recovers the same grouping.
// CHECK-LABEL: func.func @trailing_axis(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?xui8>,
// CHECK-SAME:    %{{.*}}: tensor<1xi64>) -> tensor<?x?x1xui8> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?xui8>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x1xui8> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[AXIS0:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE0:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS0]]
// CHECK-NEXT:      %[[EXT0:.*]] = shape.size_to_index %[[SIZE0]] : !shape.size
// CHECK-NEXT:      %[[AXIS1:.*]] = shape.const_size 1
// CHECK-NEXT:      %[[SIZE1:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS1]]
// CHECK-NEXT:      %[[EXT1:.*]] = shape.size_to_index %[[SIZE1]] : !shape.size
// CHECK-NEXT:      %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[EXT0]], %[[EXT1]], %[[ONE]] : index, index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?xui8>) outs(%[[INIT]] : tensor<?x?x1xui8>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?xui8>, %{{.*}}: tensor<?x?x1xui8>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?xui8>
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?xui8>
// CHECK-NEXT:      %[[EXPANDED:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0], [1, 2]] output_shape {{\[}}%[[DIM0]], %[[DIM1]], 1] : tensor<?x?xui8> into tensor<?x?x1xui8>
// CHECK-NEXT:      hipsr.compute_yield %[[EXPANDED]] : tensor<?x?x1xui8>
// CHECK-NEXT:    } : tensor<?x?x1xui8>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x1xui8>
func.func @trailing_axis(%ctx: !hipsr.context, %input: tensor<?x?xui8>,
                         %axes: tensor<1xi64>) -> tensor<?x?x1xui8> {
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<?x?xui8>, tensor<1xi64>) -> tensor<?x?x1xui8>
  return %0 : tensor<?x?x1xui8>
}

// -----

// A rank-0 input has no group for the inserted axis to join, which is the
// empty reassociation tensor.expand_shape uses for that case.
// CHECK-LABEL: func.func @scalar(
// CHECK:         tensor.expand_shape %{{.*}} [] output_shape [1] : tensor<i64> into tensor<1xi64>
func.func @scalar(%ctx: !hipsr.context, %input: tensor<i64>,
                  %axes: tensor<1xi64>) -> tensor<1xi64> {
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}

// -----

// A leading inserted axis joins the first group, and a static input needs no
// extent query at all: every line below is a literal or the expansion itself,
// so no tensor.dim appears in either region.
// CHECK-LABEL: func.func @leading_axis(
// CHECK:         hipsr.placeholder
// CHECK-SAME:      : tensor<1x4x8xf16> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[D0:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[D1:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[D2:.*]] = arith.constant 8 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[D0]], %[[D1]], %[[D2]] : index, index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<4x8xf16>, %{{.*}}: tensor<1x4x8xf16>):
// CHECK-NEXT:      %[[EXPANDED:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1], [2]] output_shape [1, 4, 8] : tensor<4x8xf16> into tensor<1x4x8xf16>
// CHECK-NEXT:      hipsr.compute_yield %[[EXPANDED]] : tensor<1x4x8xf16>
// CHECK-NEXT:    } : tensor<1x4x8xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<1x4x8xf16>
func.func @leading_axis(%ctx: !hipsr.context, %input: tensor<4x8xf16>,
                        %axes: tensor<1xi64>) -> tensor<1x4x8xf16> {
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<4x8xf16>, tensor<1xi64>) -> tensor<1x4x8xf16>
  return %0 : tensor<1x4x8xf16>
}

// -----

// Two inserted axes around a dynamic extent; only that extent costs a query.
// CHECK-LABEL: func.func @two_axes(
// CHECK:         ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x128xf16>, %{{.*}}: tensor<1x?x128x1xf16>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x128xf16>
// CHECK-NEXT:      tensor.expand_shape %[[IN]] {{\[}}[0, 1], [2, 3]] output_shape [1, %[[DIM]], 128, 1] : tensor<?x128xf16> into tensor<1x?x128x1xf16>
func.func @two_axes(%ctx: !hipsr.context, %input: tensor<?x128xf16>,
                    %axes: tensor<2xi64>) -> tensor<1x?x128x1xf16> {
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<?x128xf16>, tensor<2xi64>) -> tensor<1x?x128x1xf16>
  return %0 : tensor<1x?x128x1xf16>
}

// -----

// An unsqueeze that inserts nothing leaves the input untouched.
// CHECK-LABEL: func.func @no_axes(
// CHECK-SAME:    %{{.*}}: !hipsr.context, %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %{{.*}}: tensor<0xi64>) -> tensor<2x3xf16> {
// CHECK-NEXT:    return %[[INPUT]] : tensor<2x3xf16>
func.func @no_axes(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                   %axes: tensor<0xi64>) -> tensor<2x3xf16> {
  %0 = "onnx.Unsqueeze"(%input, %axes)
      : (tensor<2x3xf16>, tensor<0xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}
