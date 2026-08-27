// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// CHECK-LABEL: func.func @min_two_inputs
// CHECK-SAME:  (%[[CTX:[^:]*]]: !hipsr.context, %[[A:[^:]*]]: tensor<4x1024xf16, #hipsr.mem<device>>, %[[B:[^:]*]]: tensor<4x1024xf16, #hipsr.mem<device>>) -> tensor<4x1024xf16, #hipsr.mem<device>> {
// CHECK-NEXT:  %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<4x1024xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[RESULT:.+]] = hipsr.min(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<4x1024xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<4x1024xf16, #hipsr.mem<device>>) : tensor<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  return %[[RESULT]] : tensor<4x1024xf16, #hipsr.mem<device>>
func.func @min_two_inputs(%ctx: !hipsr.context, %a: tensor<4x1024xf16>,
                          %b: tensor<4x1024xf16>) -> tensor<4x1024xf16> {
  %0 = "onnx.Min"(%a, %b) : (tensor<4x1024xf16>, tensor<4x1024xf16>)
      -> tensor<4x1024xf16>
  "onnx.Return"(%0) : (tensor<4x1024xf16>) -> ()
}

// -----

// CHECK-LABEL: func.func @min_three_inputs
// CHECK-SAME:  (%[[CTX:[^:]*]]: !hipsr.context, %[[A:[^:]*]]: tensor<4x1024xf16, #hipsr.mem<device>>, %[[B:[^:]*]]: tensor<4x1024xf16, #hipsr.mem<device>>, %[[C:[^:]*]]: tensor<4x1024xf16, #hipsr.mem<device>>) -> tensor<4x1024xf16, #hipsr.mem<device>> {
// CHECK-NEXT:  %[[FIRST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<4x1024xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[FIRST:.+]] = hipsr.min(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<4x1024xf16, #hipsr.mem<device>>) outs(%[[FIRST_INIT]] : tensor<4x1024xf16, #hipsr.mem<device>>) : tensor<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[SECOND_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[FIRST]], %[[C]] : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<4x1024xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[SECOND:.+]] = hipsr.min(%[[CTX]]) ins(%[[FIRST]], %[[C]] : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<4x1024xf16, #hipsr.mem<device>>) outs(%[[SECOND_INIT]] : tensor<4x1024xf16, #hipsr.mem<device>>) : tensor<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  return %[[SECOND]] : tensor<4x1024xf16, #hipsr.mem<device>>
func.func @min_three_inputs(%ctx: !hipsr.context, %a: tensor<4x1024xf16>,
                            %b: tensor<4x1024xf16>, %c: tensor<4x1024xf16>)
    -> tensor<4x1024xf16> {
  %0 = "onnx.Min"(%a, %b, %c)
      : (tensor<4x1024xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>)
      -> tensor<4x1024xf16>
  "onnx.Return"(%0) : (tensor<4x1024xf16>) -> ()
}

// -----

// CHECK-LABEL: func.func @min_three_inputs_broadcast
// CHECK-SAME:  (%[[CTX:[^:]*]]: !hipsr.context, %[[A:[^:]*]]: tensor<2x1xf16, #hipsr.mem<device>>, %[[B:[^:]*]]: tensor<1x3xf16, #hipsr.mem<device>>, %[[C:[^:]*]]: tensor<2x3xf16, #hipsr.mem<device>>) -> tensor<2x3xf16, #hipsr.mem<device>> {
// CHECK-NEXT:  %[[FIRST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<2x1xf16, #hipsr.mem<device>>, tensor<1x3xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[FIRST:.+]] = hipsr.min(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<2x1xf16, #hipsr.mem<device>>, tensor<1x3xf16, #hipsr.mem<device>>) outs(%[[FIRST_INIT]] : tensor<2x3xf16, #hipsr.mem<device>>) : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[SECOND_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[FIRST]], %[[C]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x3xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[SECOND:.+]] = hipsr.min(%[[CTX]]) ins(%[[FIRST]], %[[C]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x3xf16, #hipsr.mem<device>>) outs(%[[SECOND_INIT]] : tensor<2x3xf16, #hipsr.mem<device>>) : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:  return %[[SECOND]] : tensor<2x3xf16, #hipsr.mem<device>>
func.func @min_three_inputs_broadcast(%ctx: !hipsr.context, %a: tensor<2x1xf16>,
                                      %b: tensor<1x3xf16>, %c: tensor<2x3xf16>)
    -> tensor<2x3xf16> {
  %0 = "onnx.Min"(%a, %b, %c)
      : (tensor<2x1xf16>, tensor<1x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  "onnx.Return"(%0) : (tensor<2x3xf16>) -> ()
}

// -----

// CHECK-LABEL: func.func @min_single_input
// CHECK-SAME:  (%[[CTX:[^:]*]]: !hipsr.context, %[[A:[^:]*]]: tensor<4x1024xf16, #hipsr.mem<device>>) -> tensor<4x1024xf16, #hipsr.mem<device>> {
// CHECK-NEXT:  return %[[A]] : tensor<4x1024xf16, #hipsr.mem<device>>
func.func @min_single_input(%ctx: !hipsr.context, %a: tensor<4x1024xf16>)
    -> tensor<4x1024xf16> {
  %0 = "onnx.Min"(%a) : (tensor<4x1024xf16>) -> tensor<4x1024xf16>
  "onnx.Return"(%0) : (tensor<4x1024xf16>) -> ()
}

// -----

// CHECK-LABEL: func.func @min_broadcast
// CHECK-SAME:  (%[[CTX:[^:]*]]: !hipsr.context, %[[A:[^:]*]]: tensor<?x1024xf16, #hipsr.mem<device>>, %[[B:[^:]*]]: tensor<1024xf16, #hipsr.mem<device>>) -> tensor<?x1024xf16, #hipsr.mem<device>> {
// CHECK-NEXT:  %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  %[[RESULT:.+]] = hipsr.min(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x1024xf16, #hipsr.mem<device>>) : tensor<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT:  return %[[RESULT]] : tensor<?x1024xf16, #hipsr.mem<device>>
func.func @min_broadcast(%ctx: !hipsr.context, %a: tensor<?x1024xf16>,
                         %b: tensor<1024xf16>) -> tensor<?x1024xf16> {
  %0 = "onnx.Min"(%a, %b) : (tensor<?x1024xf16>, tensor<1024xf16>)
      -> tensor<?x1024xf16>
  "onnx.Return"(%0) : (tensor<?x1024xf16>) -> ()
}
