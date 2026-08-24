// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// The whole hipsr pipeline over the smallest graph it has to handle: one
// onnx.MatMul against a constant weight. The CHECK block is the expected
// output of the pipeline as it currently stands, extended by each pass added
// to buildHipsrPipeline.
//
// Two constraints shaped the graph and are worth preserving when editing it.
// Rank-2 operands drive MatMul's batch split and broadcast with empty batch
// shapes, the cheapest form of its shape graph. And the weight's elements are
// spelled out because hipsr.constant rejects a splat with more than one
// element, so the dense<1.0> shorthand a larger weight would need is not
// available; an external dense_resource weight belongs in its own case.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// The placeholder's shape region is still empty here, and the CHECK-NEXT chain
// is what says so: a populated region prints as extra lines, which would leave
// the matmul off the line the chain expects it on.
// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:      %[[A:.+]]: tensor<2x3xf16, #hipsr.mem<device>> {onnx.name = "a"})
// CHECK-SAME:      -> (tensor<2x4xf16, #hipsr.mem<device>> {onnx.name = "y"})
// CHECK-SAME:      attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:    %[[B:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x4xf16>} : tensor<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<3x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[Y:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<3x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2x4xf16, #hipsr.mem<device>>) : tensor<2x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[Y]] : tensor<2x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @main_graph(%a: tensor<2x3xf16> {onnx.name = "a"})
    -> (tensor<2x4xf16> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %b = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 1.0, 2.0, 3.0]]> : tensor<3x4xf16>}
      : () -> tensor<3x4xf16>
  %y = "onnx.MatMul"(%a, %b) : (tensor<2x3xf16>, tensor<3x4xf16>)
      -> tensor<2x4xf16>
  return %y : tensor<2x4xf16>
}
