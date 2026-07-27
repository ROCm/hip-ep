// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Converts supported onnx.Expand ops to placeholder-backed hipsr.expand ops.
// Unsupported forms remain unchanged for another conversion path.

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

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

// Requested shape can add leading dimensions.
// CHECK-LABEL: func.func @expand_leading_rank(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x8xf32>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<4xi64>) -> tensor<?x?x?x8xf32> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder : tensor<?x?x?x8xf32>
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<?x8xf32>, tensor<4xi64>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x?x?x8xf32>) : tensor<?x?x?x8xf32>
// CHECK-NOT:     shape_region
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?x?x8xf32>
func.func @expand_leading_rank(%ctx: !hipsr.context,
                               %input: tensor<?x8xf32>,
                               %shape: tensor<4xi64>)
    -> tensor<?x?x?x8xf32> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<?x8xf32>, tensor<4xi64>) -> tensor<?x?x?x8xf32>
  return %0 : tensor<?x?x?x8xf32>
}

// -----

// Shape must be a rank-1 extent vector.
// CHECK-LABEL: func.func @expand_shape_rank
// CHECK-NOT:   hipsr.expand
// CHECK:       %[[RESULT:.+]] = "onnx.Expand"
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
// CHECK-LABEL: func.func @expand_shape_element_type
// CHECK-NOT:   hipsr.expand
// CHECK:       %[[RESULT:.+]] = "onnx.Expand"
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
// CHECK-LABEL: func.func @expand_dynamic_shape_length
// CHECK-NOT:   hipsr.expand
// CHECK:       %[[RESULT:.+]] = "onnx.Expand"
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
// CHECK-LABEL: func.func @expand_element_mismatch
// CHECK-NOT:   hipsr.expand
// CHECK:       %[[RESULT:.+]] = "onnx.Expand"
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
// CHECK-LABEL: func.func @expand_output_rank
// CHECK-NOT:   hipsr.expand
// CHECK:       %[[RESULT:.+]] = "onnx.Expand"
// CHECK-NEXT:  return %[[RESULT]] : tensor<2x3xf16>
func.func @expand_output_rank(%ctx: !hipsr.context,
                              %input: tensor<2x3xf16>,
                              %shape: tensor<4xi64>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<4xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}
