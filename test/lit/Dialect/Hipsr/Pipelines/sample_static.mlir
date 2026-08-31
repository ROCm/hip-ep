// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// A sample for testing --hipsr-pipeline, with every extent static.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// Every line of the output is checked, which is also what shows that no shape
// dialect op survives: one would have to appear on a line of its own.

// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:      %[[A:.+]]: tensor<2x3xf16, #hipsr.mem<device>> {onnx.name = "a"},
// CHECK-SAME:      %[[B:.+]]: tensor<2x4xf32, #hipsr.mem<device>> {onnx.name = "b"})
// CHECK-SAME:      -> (tensor<2x2xf32, #hipsr.mem<device>> {onnx.name = "y"})
// CHECK-SAME:      attributes {onnx.graph.name = "main_graph"} {

// The barrier the expand needs splits the graph in two. Everything up to it
// lands in domain 0, which yields both halves of the shape and data graphs for
// domain 1 to pick up. Inside a domain the constants come first, then the shape
// computations, then the allocations they size, then the data ops, and last a
// hipsr.preserve_shape tying each shape to the value filling the buffer it
// sized.
//
// Every extent here is static, so no allocation reads a shape back. The shape
// graph sizes nothing, yet it survives because preserve_shape names it. Without
// that use it would fold away entirely.
// CHECK-NEXT:    %[[D0:.+]]:5 = hipsr.pool_domain(%[[CTX]], %[[A]], %[[B]] : !hipsr.context, tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[D0_CTX:.+]]: !hipsr.context, %[[D0_A:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[D0_B:.+]]: tensor<2x4xf32, #hipsr.mem<device>>):

// Everything constant-like leads the domain, and here that is the whole shape
// graph. Neither matmul operand has batch dimensions, so the result shape is
// the two matrix extents, both from the operand types: one dense vector.
// CHECK-NEXT:      %[[EXTENTS_SHAPE_S:.+]] = arith.constant dense<2> : tensor<1xindex>
// CHECK-NEXT:      %[[W1:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x1xf16>} : tensor<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[W2:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<4x2xf32>} : tensor<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MM1_SHAPE:.+]] = arith.constant dense<{{\[}}2, 1]> : tensor<2xindex>

// Every extent is in the types, so no allocation reads a shape back.
// CHECK-NEXT:      %[[MM1_INIT:.+]] = tensor.empty() : tensor<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST_INIT:.+]] = tensor.empty() : tensor<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXTENTS_INIT:.+]] = tensor.empty() : tensor<2xi64, #hipsr.mem<host>>

// The expand reads an extent vector, the one place a shape reaches the data
// graph, and it arrives as a host buffer. Its operand is fully static, so
// canonicalization folds the two extents into one dense vector.
// CHECK-NEXT:      %[[MM1:.+]] = hipsr.matmul(%[[D0_CTX]]) ins(%[[D0_A]], %[[W1]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>) outs(%[[MM1_INIT]] : tensor<2x1xf16, #hipsr.mem<device>>) : tensor<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST:.+]] = hipsr.cast(%[[D0_CTX]]) ins(%[[MM1]] : tensor<2x1xf16, #hipsr.mem<device>>) outs(%[[CAST_INIT]] : tensor<2x1xf32, #hipsr.mem<device>>) : tensor<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXTENTS:.+]] = hipsr.compute(%[[D0_CTX]]) ins(%[[D0_B]] : tensor<2x4xf32, #hipsr.mem<device>>) outs(%[[EXTENTS_INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.+}}: !hipsr.context, %{{.+}}: tensor<2x4xf32, #hipsr.mem<device>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[EXTENT_VECTOR:.+]] = arith.constant dense<{{\[}}2, 4]> : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENT_VECTOR]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<2xi64, #hipsr.mem<host>>

// Each shape closes the domain next to the value that filled the buffer it
// sized. Its length is part of its type. The matmul and the cast share one
// shape, because the cast copies its input's extents.
// CHECK-NEXT:      hipsr.preserve_shape %[[MM1_SHAPE]], %[[MM1]] : tensor<2xindex>, tensor<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[MM1_SHAPE]], %[[CAST]] : tensor<2xindex>, tensor<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[EXTENTS_SHAPE_S]], %[[EXTENTS]] : tensor<1xindex>, tensor<2xi64, #hipsr.mem<host>>

// The second weight has no use in this domain beyond the yield. It crosses the
// barrier as a domain result, like the buffers do.
// CHECK-NEXT:      hipsr.pool_domain_yield %[[CAST_INIT]], %[[CAST]], %[[EXTENTS_INIT]], %[[EXTENTS]], %[[W2]] : tensor<2x1xf32, #hipsr.mem<device>>, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<2x1xf32, #hipsr.mem<device>>, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>> {domain_id = 0 : i64}

// Domain 1 takes the shape-graph results first and the data results after, so
// the barrier keeps reading the buffers the shape graph names while the
// operations keep following results.
// CHECK-NEXT:    %[[D1:.+]] = hipsr.pool_domain(%[[CTX]], %[[D0]]#0, %[[D0]]#2, %[[D0]]#1, %[[D0]]#3, %[[D0]]#4 : !hipsr.context, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[D1_CTX:.+]]: !hipsr.context, %[[D1_CAST_INIT:.+]]: tensor<2x1xf32, #hipsr.mem<device>>, %[[D1_EXTENTS_INIT:.+]]: tensor<2xi64, #hipsr.mem<host>>, %[[D1_CAST:.+]]: tensor<2x1xf32, #hipsr.mem<device>>, %[[D1_EXTENTS:.+]]: tensor<2xi64, #hipsr.mem<host>>, %[[D1_W2:.+]]: tensor<4x2xf32, #hipsr.mem<device>>):

// CHECK-NEXT:      %[[D1_C2:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[D1_C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[D1_C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[IN_SHAPE:.+]] = arith.constant dense<{{\[}}2, 1]> : tensor<2xindex>

// The expand's own shape broadcasts its input shape against the requested
// extents, read out of the host buffer domain 0 handed over. That buffer stays
// opaque, so these two reads survive everything static around them.
// CHECK-NEXT:      %[[E0:.+]] = tensor.extract %[[D1_EXTENTS_INIT]]{{\[}}%[[D1_C0]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X0:.+]] = arith.index_cast %[[E0]] : i64 to index
// CHECK-NEXT:      %[[E1:.+]] = tensor.extract %[[D1_EXTENTS_INIT]]{{\[}}%[[D1_C1]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X1:.+]] = arith.index_cast %[[E1]] : i64 to index
// CHECK-NEXT:      %[[REQ:.+]] = tensor.from_elements %[[X0]], %[[X1]] : tensor<2xindex>
// Broadcasting the two is a per-dimension select, so upstream writes it as a
// generator body: out of range on either side yields 1, and a requested extent
// of 1 keeps the input's.
// CHECK-NEXT:      %[[EXPAND_SHAPE:.+]] = tensor.generate {
// CHECK-NEXT:      ^bb0(%[[DIM:.+]]: index):
// CHECK-NEXT:        %[[IN_OOB:.+]] = arith.cmpi ult, %[[DIM]], %[[D1_C0]] : index
// CHECK-NEXT:        %[[IN_EXTENT:.+]] = scf.if %[[IN_OOB]] -> (index) {
// CHECK-NEXT:          scf.yield %[[D1_C1]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[IN_E:.+]] = tensor.extract %[[IN_SHAPE]]{{\[}}%[[DIM]]] : tensor<2xindex>
// CHECK-NEXT:          scf.yield %[[IN_E]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[REQ_OOB:.+]] = arith.cmpi ult, %[[DIM]], %[[D1_C0]] : index
// CHECK-NEXT:        %[[OUT_EXTENT:.+]] = scf.if %[[REQ_OOB]] -> (index) {
// CHECK-NEXT:          scf.yield %[[IN_EXTENT]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[REQ_E:.+]] = tensor.extract %[[REQ]]{{\[}}%[[DIM]]] : tensor<2xindex>
// CHECK-NEXT:          %[[REQ_IS_ONE:.+]] = arith.cmpi eq, %[[REQ_E]], %[[D1_C1]] : index
// CHECK-NEXT:          %[[PICK:.+]] = arith.select %[[REQ_IS_ONE]], %[[IN_EXTENT]], %[[REQ_E]] : index
// CHECK-NEXT:          scf.yield %[[PICK]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        tensor.yield %[[OUT_EXTENT]] : index
// CHECK-NEXT:      } : tensor<2xindex>

// The second matmul has no batch dimensions either. Its shape is M broadcast
// against the requested leading extent, then the weight's trailing extent. The
// static side of that select is the input's own extent.
// CHECK-NEXT:      %[[M2_IS_ONE:.+]] = arith.cmpi eq, %[[X0]], %[[D1_C1]] : index
// CHECK-NEXT:      %[[M2:.+]] = arith.select %[[M2_IS_ONE]], %[[D1_C2]], %[[X0]] : index
// CHECK-NEXT:      %[[MM2_SHAPE:.+]] = tensor.from_elements %[[M2]], %[[D1_C2]] : tensor<2xindex>

// CHECK-NEXT:      %[[EXPAND_INIT:.+]] = tensor.empty() : tensor<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MM2_INIT:.+]] = tensor.empty() : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXPAND:.+]] = hipsr.expand(%[[D1_CTX]]) ins(%[[D1_CAST]], %[[D1_EXTENTS]] : tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[EXPAND_INIT]] : tensor<2x4xf32, #hipsr.mem<device>>) : tensor<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MM2:.+]] = hipsr.matmul(%[[D1_CTX]]) ins(%[[EXPAND]], %[[D1_W2]] : tensor<2x4xf32, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) outs(%[[MM2_INIT]] : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[EXPAND_SHAPE]], %[[EXPAND]] : tensor<2xindex>, tensor<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[MM2_SHAPE]], %[[MM2]] : tensor<2xindex>, tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[MM2]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<2x2xf32, #hipsr.mem<device>> {domain_id = 1 : i64}

// CHECK-NEXT:    return %[[D1]] : tensor<2x2xf32, #hipsr.mem<device>>
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
