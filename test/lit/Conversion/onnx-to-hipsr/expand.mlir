// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Converts onnx.Expand to a placeholder-backed hipsr.expand with an empty
// shape region.

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// CHECK-LABEL: func.func @expand(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x3xf16>,
// CHECK-SAME:    %[[SHAPE:.*]]: tensor<2xi64>) -> tensor<?x?xf16> {
// CHECK:         %[[INIT:.*]] = hipsr.placeholder : tensor<?x?xf16>
// CHECK:         %[[RESULT:.*]] = hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : tensor<?x3xf16>, tensor<2xi64>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NOT:     shape_region
// CHECK:         return %[[RESULT]]
func.func @expand(%ctx: !hipsr.context, %input: tensor<?x3xf16>,
                  %shape: tensor<2xi64>) -> tensor<?x?xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<?x3xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Requested shape can add leading dimensions.
// CHECK-LABEL: func.func @expand_leading_rank(
// CHECK:         %[[INIT:.*]] = hipsr.placeholder : tensor<?x?x?x8xf32>
// CHECK:         hipsr.expand(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<?x8xf32>, tensor<4xi64>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x?x?x8xf32>) : tensor<?x?x?x8xf32>
func.func @expand_leading_rank(%ctx: !hipsr.context,
                               %input: tensor<?x8xf32>,
                               %shape: tensor<4xi64>)
    -> tensor<?x?x?x8xf32> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<?x8xf32>, tensor<4xi64>) -> tensor<?x?x?x8xf32>
  return %0 : tensor<?x?x?x8xf32>
}

// -----

// Unsupported shape forms remain unchanged for another conversion path.
// CHECK-LABEL: func.func @expand_shape_rank
// CHECK: "onnx.Expand"
// CHECK-NOT: hipsr.expand
func.func @expand_shape_rank(%ctx: !hipsr.context,
                             %input: tensor<2x3xf16>,
                             %shape: tensor<1x2xi64>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<1x2xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// CHECK-LABEL: func.func @expand_shape_element_type
// CHECK: "onnx.Expand"
// CHECK-NOT: hipsr.expand
func.func @expand_shape_element_type(%ctx: !hipsr.context,
                                     %input: tensor<2x3xf16>,
                                     %shape: tensor<2xi32>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<2xi32>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// -----

// CHECK-LABEL: func.func @expand_dynamic_shape_length
// CHECK: "onnx.Expand"
// CHECK-NOT: hipsr.expand
func.func @expand_dynamic_shape_length(%ctx: !hipsr.context,
                                       %input: tensor<2x3xf16>,
                                       %shape: tensor<?xi64>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Expand"(%input, %shape)
      : (tensor<2x3xf16>, tensor<?xi64>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}
