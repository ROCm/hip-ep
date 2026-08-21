// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --allow-unregistered-dialect --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

#encoding = #test.encoding

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2xf32, #test.encoding>
// CHECK: hip.where
// CHECK-SAME: outs(%[[INIT]] : tensor<2xf32, #test.encoding>)
func.func @main_graph(
    %cond: tensor<2xi1>, %x: tensor<2xf32, #encoding>,
    %y: tensor<2xf32, #encoding>) -> tensor<2xf32, #encoding> {
  %result = "onnx.Where"(%cond, %x, %y) :
      (tensor<2xi1>, tensor<2xf32, #encoding>,
       tensor<2xf32, #encoding>) -> tensor<2xf32, #encoding>
  return %result : tensor<2xf32, #encoding>
}
