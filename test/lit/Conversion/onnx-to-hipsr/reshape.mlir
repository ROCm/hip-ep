// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Reshape to a hipsr.compute wrapping tensor.collapse_shape or
// tensor.expand_shape. Like the onnx.Shape conversion, it populates the
// placeholder's shape region itself. Rejected forms live in
// reshape-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The shape operand goes unread: the result type already carries the extents,
// and a dynamic one follows from the preserved element count.
// CHECK-LABEL: func.func @collapse_dynamic(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4096xf16>,
// CHECK-SAME:    %{{.*}}: tensor<1xi64>) -> tensor<?xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[AXIS:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS]]
// CHECK-NEXT:      %[[EXTENT:.*]] = shape.size_to_index %[[SIZE]] : !shape.size
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[COUNT:.*]] = arith.muli %[[EXTENT]], %[[COLS]] : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[COUNT]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x4096xf16>) outs(%[[INIT]] : tensor<?xf16>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x4096xf16>, %{{.*}}: tensor<?xf16>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<?x4096xf16> into tensor<?xf16>
// CHECK-NEXT:      hipsr.compute_yield %[[FLAT]] : tensor<?xf16>
// CHECK-NEXT:    } : tensor<?xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?xf16>
func.func @collapse_dynamic(%ctx: !hipsr.context, %input: tensor<?x4096xf16>,
                            %shape: tensor<1xi64>) -> tensor<?xf16> {
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x4096xf16>, tensor<1xi64>) -> tensor<?xf16>
  return %0 : tensor<?xf16>
}

// -----

// A static result needs no extent arithmetic, and collapse_shape derives its
// extents from the grouping alone.
// CHECK-LABEL: func.func @collapse_static(
// CHECK:         hipsr.placeholder
// CHECK-SAME:      : tensor<6x4xf16> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[D0:.*]] = arith.constant 6 : index
// CHECK-NEXT:      %[[D1:.*]] = arith.constant 4 : index
// CHECK-NEXT:      shape.from_extents %[[D0]], %[[D1]] : index, index
// CHECK:         ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<2x3x4xf16>, %{{.*}}: tensor<6x4xf16>):
// CHECK-NEXT:      tensor.collapse_shape %[[IN]] {{\[}}[0, 1], [2]] : tensor<2x3x4xf16> into tensor<6x4xf16>
func.func @collapse_static(%ctx: !hipsr.context, %input: tensor<2x3x4xf16>,
                           %shape: tensor<2xi64>) -> tensor<6x4xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3x4xf16>, tensor<2xi64>) -> tensor<6x4xf16>
  return %0 : tensor<6x4xf16>
}

// -----

// Expanding splits one input extent over a group, which the input type cannot
// pin down, so the body recomputes the dynamic extent for output_shape.
// CHECK-LABEL: func.func @expand_dynamic(
// CHECK:         hipsr.placeholder
// CHECK:           %[[EXTENT:.*]] = shape.size_to_index
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.divui %[[EXTENT]], %[[COLS]] : index
// CHECK-NEXT:      %[[COLS2:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      shape.from_extents %[[ROWS]], %[[COLS2]] : index, index
// CHECK:         ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?xf16>, %{{.*}}: tensor<?x4096xf16>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?xf16>
// CHECK-NEXT:      %[[BODY_COLS:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[BODY_ROWS:.*]] = arith.divui %[[DIM]], %[[BODY_COLS]] : index
// CHECK-NEXT:      tensor.expand_shape %[[IN]] {{\[}}[0, 1]] output_shape {{\[}}%[[BODY_ROWS]], 4096] : tensor<?xf16> into tensor<?x4096xf16>
func.func @expand_dynamic(%ctx: !hipsr.context, %input: tensor<?xf16>,
                          %shape: tensor<2xi64>) -> tensor<?x4096xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<?xf16>, tensor<2xi64>) -> tensor<?x4096xf16>
  return %0 : tensor<?x4096xf16>
}

// -----

// A fully static expansion carries its output_shape as literals. Both regions
// are checked line by line to show neither divides to recover an extent.
// CHECK-LABEL: func.func @expand_static(
// CHECK:         hipsr.placeholder
// CHECK-SAME:      : tensor<2x3xf16> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[D0:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[D1:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[D0]], %[[D1]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<6xf16>, %{{.*}}: tensor<2x3xf16>):
// CHECK-NEXT:      %[[EXPANDED:.*]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1]] output_shape [2, 3] : tensor<6xf16> into tensor<2x3xf16>
// CHECK-NEXT:      hipsr.compute_yield %[[EXPANDED]] : tensor<2x3xf16>
// CHECK-NEXT:    } : tensor<2x3xf16>
// CHECK-NEXT:    return %[[RESULT]] : tensor<2x3xf16>
func.func @expand_static(%ctx: !hipsr.context, %input: tensor<6xf16>,
                         %shape: tensor<2xi64>) -> tensor<2x3xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<6xf16>, tensor<2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// A reshape to the input's own type reinterprets nothing, so it leaves no
// destination behind.
// CHECK-LABEL: func.func @identity(
// CHECK-SAME:    %{{.*}}: !hipsr.context, %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %{{.*}}: tensor<2xi64>) -> tensor<2x3xf16> {
// CHECK-NEXT:    return %[[INPUT]] : tensor<2x3xf16>
func.func @identity(%ctx: !hipsr.context, %input: tensor<2x3xf16>,
                    %shape: tensor<2xi64>) -> tensor<2x3xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<2x3xf16>, tensor<2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}
