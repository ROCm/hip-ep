// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// The hipsr pipeline over a small ONNX graph, every extent static.
// sample_dynamic.mlir is the same graph with a dynamic leading extent.
//
// The CHECK block is the pipeline's output as it stands today, and grows with
// each pass added to buildHipsrPipeline.
//
// Editing notes. Keep the graph a chain, so that the shape and data graphs stay
// distinguishable: each init takes the previous placeholder, each operation the
// previous result. Keep the expand, which is what makes a barrier placeholder
// appear, and keep feeding its extents from onnx.Shape -- hipsr.expand requires
// them in the host space, and a graph input would arrive in the device space.
// Spell out constant elements, because hipsr.constant rejects a splat with more
// than one element.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// Regions whose contents come from an operation's own recipe are matched at
// their boundaries -- block signature, terminator, closing brace -- because
// each recipe is pinned under
// test/lit/Dialect/Hipsr/Transforms/PopulateShapeRegion/ and
// test/lit/Conversion/onnx-to-hipsr/. The two trivial regions, the cast's
// identity and the shape's constant, are matched whole.
// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:      %[[A:.+]]: tensor<2x3xf16, #hipsr.mem<device>> {onnx.name = "a"},
// CHECK-SAME:      %[[B:.+]]: tensor<2x4xf32, #hipsr.mem<device>> {onnx.name = "b"})
// CHECK-SAME:      -> (tensor<2x2xf32, #hipsr.mem<device>> {onnx.name = "y"})
// CHECK-SAME:      attributes {onnx.graph.name = "main_graph"} {

// CHECK-NEXT:    %[[W1:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x1xf16>} : tensor<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[MM1_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[W1]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x1xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.+}}: !shape.shape, %{{.+}}: !shape.shape):
// CHECK:           hipsr.shape_yield %{{.+}} : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[MM1:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[W1]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>) outs(%[[MM1_INIT]] : tensor<2x1xf16, #hipsr.mem<device>>) : tensor<2x1xf16, #hipsr.mem<device>>

// CHECK-NEXT:    %[[CAST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[MM1_INIT]] : tensor<2x1xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x1xf32, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[MM1_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT:      hipsr.shape_yield %[[MM1_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[CAST:.+]] = hipsr.cast(%[[CTX]]) ins(%[[MM1]] : tensor<2x1xf16, #hipsr.mem<device>>) outs(%[[CAST_INIT]] : tensor<2x1xf32, #hipsr.mem<device>>) : tensor<2x1xf32, #hipsr.mem<device>>

// onnx.Shape has no shape-graph inputs of its own: the result length is fixed by
// the operand's rank, so the init takes no ins and its region is a constant. The
// extents land in the host space, which is what hipsr.expand needs.
// CHECK-NEXT:    %[[SHAPE_INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[RANK:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[SHAPE_SHAPE:.+]] = shape.from_extents %[[RANK]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[SHAPE:.+]] = hipsr.compute(%[[CTX]]) ins(%[[B]] : tensor<2x4xf32, #hipsr.mem<device>>) outs(%[[SHAPE_INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %{{.+}}: tensor<2x4xf32, #hipsr.mem<device>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK:           hipsr.compute_yield %{{.+}} : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<2xi64, #hipsr.mem<host>>

// CHECK-NEXT:    %[[EXPAND_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[CAST_INIT]], %[[SHAPE_INIT]] : tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<2x4xf32, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %{{.+}}: tensor<2x1xf32, #hipsr.mem<device>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK:           hipsr.shape_yield %{{.+}} : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[EXPAND:.+]] = hipsr.expand(%[[CTX]]) ins(%[[CAST]], %[[SHAPE]] : tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[EXPAND_INIT]] : tensor<2x4xf32, #hipsr.mem<device>>) : tensor<2x4xf32, #hipsr.mem<device>>

// CHECK-NEXT:    %[[W2:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<4x2xf32>} : tensor<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    %[[MM2_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[EXPAND_INIT]], %[[W2]] : tensor<2x4xf32, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x2xf32, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.+}}: !shape.shape, %{{.+}}: !shape.shape):
// CHECK:           hipsr.shape_yield %{{.+}} : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[Y:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[EXPAND]], %[[W2]] : tensor<2x4xf32, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) outs(%[[MM2_INIT]] : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>

// CHECK-NEXT:    return %[[Y]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @main_graph(%a: tensor<2x3xf16> {onnx.name = "a"},
                      %b: tensor<2x4xf32> {onnx.name = "b"})
    -> (tensor<2x2xf32> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %w1 = "onnx.Constant"() {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : () -> tensor<3x1xf16>
  %mm1 = "onnx.MatMul"(%a, %w1) : (tensor<2x3xf16>, tensor<3x1xf16>)
      -> tensor<2x1xf16>
  %cast = "onnx.Cast"(%mm1) {to = f32} : (tensor<2x1xf16>) -> tensor<2x1xf32>
  %shape = "onnx.Shape"(%b) : (tensor<2x4xf32>) -> tensor<2xi64>
  %expand = "onnx.Expand"(%cast, %shape)
      : (tensor<2x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  %w2 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0],
                                          [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>}
      : () -> tensor<4x2xf32>
  %y = "onnx.MatMul"(%expand, %w2) : (tensor<2x4xf32>, tensor<4x2xf32>)
      -> tensor<2x2xf32>
  "onnx.Return"(%y) : (tensor<2x2xf32>) -> ()
}
