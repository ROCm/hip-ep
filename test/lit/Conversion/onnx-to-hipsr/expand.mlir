// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Converts supported onnx.Expand ops to placeholder-backed hipsr.expand ops.
// Unsupported forms remain unchanged for another conversion path.

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s
// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr -canonicalize | FileCheck %s --check-prefix=CANONICALIZE

// CHECK-LABEL: func.func @expand(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<2xi64>) -> tensor<?x?xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder : tensor<?x?xf16>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<?x3xf16>, tensor<2xi64>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NOT:     shape_region
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?xf16>
func.func @expand(%ctx: !hipsr.context, %input: tensor<?x3xf16>,
                  %shape: tensor<2xi64>) -> tensor<?x?xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<?x3xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Conversion always preserves the shape operand. Canonicalization folds a
// ConstantLike shape into shape_attr and removes the now-unused constant.
// CHECK-LABEL: func.func @expand_constant_shape(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<1x3xf16>) -> tensor<4x3xf16> {
// CHECK-NEXT:  %[[SHAPE:.*]] = arith.constant dense<[4, 3]> : tensor<2xi64>
// CHECK-NEXT:  %[[INIT:.*]] = hipsr.placeholder : tensor<4x3xf16>
// CHECK-NEXT:  %[[RESULT:.*]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<1x3xf16>, tensor<2xi64>)
// CHECK-SAME:    outs(%[[INIT]] : tensor<4x3xf16>) : tensor<4x3xf16>
// CHECK-NEXT:  return %[[RESULT]] : tensor<4x3xf16>
// CANONICALIZE-LABEL: func.func @expand_constant_shape(
// CANONICALIZE-SAME:    %[[CTX:.*]]: !hipsr.context,
// CANONICALIZE-SAME:    %[[INPUT:.*]]: tensor<1x3xf16>) -> tensor<4x3xf16> {
// CANONICALIZE-NEXT:  %[[INIT:.*]] = hipsr.placeholder : tensor<4x3xf16>
// CANONICALIZE-NEXT:  %[[RESULT:.*]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]] : tensor<1x3xf16>)
// CANONICALIZE-SAME:    outs(%[[INIT]] : tensor<4x3xf16>) {shape_attr = array<i64: 4, 3>} : tensor<4x3xf16>
// CANONICALIZE-NEXT:  return %[[RESULT]] : tensor<4x3xf16>
func.func @expand_constant_shape(%ctx: !hipsr.context,
                                 %input: tensor<1x3xf16>)
    -> tensor<4x3xf16> {
  %shape = arith.constant dense<[4, 3]> : tensor<2xi64>
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<1x3xf16>, tensor<2xi64>) -> tensor<4x3xf16>
  return %0 : tensor<4x3xf16>
}

// -----

// Shape must be a rank-1 extent vector.
// CHECK-LABEL: func.func @expand_shape_rank(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<1x2xi64>) -> tensor<2x3xf16> {
// CHECK-NOT:   hipsr.expand
// CHECK-NEXT:  %[[RESULT:.*]] = "onnx.Expand"(%[[INPUT]], %[[SHAPE]]) : (tensor<2x3xf16>, tensor<1x2xi64>) -> tensor<2x3xf16>
// CHECK-NEXT:  return %[[RESULT]] : tensor<2x3xf16>
func.func @expand_shape_rank(%ctx: !hipsr.context,
                             %input: tensor<2x3xf16>,
                             %shape: tensor<1x2xi64>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1x2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// ONNX shape extents use i64 elements.
// CHECK-LABEL: func.func @expand_shape_element_type(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<2xi32>) -> tensor<2x3xf16> {
// CHECK-NOT:   hipsr.expand
// CHECK-NEXT:  %[[RESULT:.*]] = "onnx.Expand"(%[[INPUT]], %[[SHAPE]]) : (tensor<2x3xf16>, tensor<2xi32>) -> tensor<2x3xf16>
// CHECK-NEXT:  return %[[RESULT]] : tensor<2x3xf16>
func.func @expand_shape_element_type(%ctx: !hipsr.context,
                                     %input: tensor<2x3xf16>,
                                     %shape: tensor<2xi32>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<2xi32>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// The static shape length determines the output rank.
// CHECK-LABEL: func.func @expand_dynamic_shape_length(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<?xi64>) -> tensor<2x3xf16> {
// CHECK-NOT:   hipsr.expand
// CHECK-NEXT:  %[[RESULT:.*]] = "onnx.Expand"(%[[INPUT]], %[[SHAPE]]) : (tensor<2x3xf16>, tensor<?xi64>) -> tensor<2x3xf16>
// CHECK-NEXT:  return %[[RESULT]] : tensor<2x3xf16>
func.func @expand_dynamic_shape_length(%ctx: !hipsr.context,
                                       %input: tensor<2x3xf16>,
                                       %shape: tensor<?xi64>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<?xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// Expand preserves the input element type.
// CHECK-LABEL: func.func @expand_element_mismatch(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<2xi64>) -> tensor<2x3xf32> {
// CHECK-NOT:   hipsr.expand
// CHECK-NEXT:  %[[RESULT:.*]] = "onnx.Expand"(%[[INPUT]], %[[SHAPE]]) : (tensor<2x3xf16>, tensor<2xi64>) -> tensor<2x3xf32>
// CHECK-NEXT:  return %[[RESULT]] : tensor<2x3xf32>
func.func @expand_element_mismatch(%ctx: !hipsr.context,
                                   %input: tensor<2x3xf16>,
                                   %shape: tensor<2xi64>)
    -> tensor<2x3xf32> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<2xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// Output rank is max(input rank, requested shape length).
// CHECK-LABEL: func.func @expand_output_rank(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<4xi64>) -> tensor<2x3xf16> {
// CHECK-NOT:   hipsr.expand
// CHECK-NEXT:  %[[RESULT:.*]] = "onnx.Expand"(%[[INPUT]], %[[SHAPE]]) : (tensor<2x3xf16>, tensor<4xi64>) -> tensor<2x3xf16>
// CHECK-NEXT:  return %[[RESULT]] : tensor<2x3xf16>
func.func @expand_output_rank(%ctx: !hipsr.context,
                              %input: tensor<2x3xf16>,
                              %shape: tensor<4xi64>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<4xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}
