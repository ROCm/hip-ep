// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// A sample for testing --hipsr-pipeline, with a dynamic leading extent.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:      %[[A:.+]]: tensor<?x3xf16, #hipsr.mem<device>> {onnx.name = "a"},
// CHECK-SAME:      %[[B:.+]]: tensor<?x4xf32, #hipsr.mem<device>> {onnx.name = "b"})
// CHECK-SAME:      -> (tensor<?x2xf32, #hipsr.mem<device>> {onnx.name = "y"})
// CHECK-SAME:      attributes {onnx.graph.name = "main_graph"} {

// The domains split exactly where they do in sample_static.mlir, and so do the
// five zones inside each one: neither the barrier's position nor the shape
// graph depends on whether the extents are known. sample_static.mlir carries
// the commentary on the shape graph itself; what is worth reading here is where
// an extent gets read back out of it, which is the only thing that differs.
// CHECK-NEXT:    %[[D0:.+]]:5 = hipsr.pool_domain(%[[CTX]], %[[A]], %[[B]] : !hipsr.context, tensor<?x3xf16, #hipsr.mem<device>>, tensor<?x4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[D0_CTX:.+]]: !hipsr.context, %[[D0_A:.+]]: tensor<?x3xf16, #hipsr.mem<device>>, %[[D0_B:.+]]: tensor<?x4xf32, #hipsr.mem<device>>):

// CHECK-NEXT:      %[[EXTENTS_SHAPE:.+]] = shape.const_shape [2] : tensor<1xindex>
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[W1:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x1xf16>} : tensor<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[W2:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<4x2xf32>} : tensor<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[A_SHAPE:.+]] = shape.shape_of %[[D0_A]] : tensor<?x3xf16, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[W1_SHAPE:.+]] = shape.shape_of %[[W1]] : tensor<3x1xf16, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[A_BATCH:.+]], %[[A_TAIL:.+]] = "shape.split_at"(%[[A_SHAPE]], %[[C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[W1_BATCH:.+]], %[[W1_TAIL:.+]] = "shape.split_at"(%[[W1_SHAPE]], %[[C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[BATCH:.+]] = shape.broadcast %[[A_BATCH]], %[[W1_BATCH]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:      %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[C0]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[N:.+]] = shape.get_extent %[[W1_SHAPE]], %[[C1]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[MATRIX:.+]] = tensor.from_elements %[[M]], %[[N]] : tensor<2xindex>
// CHECK-NEXT:      %[[MM1_SHAPE:.+]] = tensor.concat dim(0) %[[BATCH]], %[[MATRIX]] : (tensor<?xindex>, tensor<2xindex>) -> tensor<?xindex>

// This is the difference from sample_static.mlir. The rows extent is not in the
// type, so each allocation reads it back out of the shape that sized it, with
// its own shape.get_extent. Both reads name the same shape, because the cast
// forwards its operand's; CSE is left to run later rather than being built in
// here. The host extent vector is still fully static, so it needs no read.
// CHECK-NEXT:      %[[MM1_ROWS:.+]] = shape.get_extent %[[MM1_SHAPE]], %[[C0]] : tensor<?xindex>, index -> index
// CHECK-NEXT:      %[[MM1_INIT:.+]] = tensor.empty(%[[MM1_ROWS]]) : tensor<?x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST_ROWS:.+]] = shape.get_extent %[[MM1_SHAPE]], %[[C0]] : tensor<?xindex>, index -> index
// CHECK-NEXT:      %[[CAST_INIT:.+]] = tensor.empty(%[[CAST_ROWS]]) : tensor<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXTENTS_INIT:.+]] = tensor.empty() : tensor<2xi64, #hipsr.mem<host>>

// CHECK-NEXT:      %[[MM1:.+]] = hipsr.matmul(%[[D0_CTX]]) ins(%[[D0_A]], %[[W1]] : tensor<?x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>) outs(%[[MM1_INIT]] : tensor<?x1xf16, #hipsr.mem<device>>) : tensor<?x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST:.+]] = hipsr.cast(%[[D0_CTX]]) ins(%[[MM1]] : tensor<?x1xf16, #hipsr.mem<device>>) outs(%[[CAST_INIT]] : tensor<?x1xf32, #hipsr.mem<device>>) : tensor<?x1xf32, #hipsr.mem<device>>

// The onnx.Shape lowering reads the dynamic extent off the data operand rather
// than folding to a constant vector the way it does in sample_static.mlir.
// CHECK-NEXT:      %[[EXTENTS:.+]] = hipsr.compute(%[[D0_CTX]]) ins(%[[D0_B]] : tensor<?x4xf32, #hipsr.mem<device>>) outs(%[[EXTENTS_INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.+}}: !hipsr.context, %[[C_B:.+]]: tensor<?x4xf32, #hipsr.mem<device>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[D1_EXT:.+]] = arith.constant 4 : i64
// CHECK-NEXT:        %[[DIM_IDX:.+]] = arith.constant 0 : index
// CHECK-NEXT:        %[[DIM:.+]] = tensor.dim %[[C_B]], %[[DIM_IDX]] : tensor<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[D0_EXT:.+]] = arith.index_cast %[[DIM]] : index to i64
// CHECK-NEXT:        %[[EXTENT_VECTOR:.+]] = tensor.from_elements %[[D0_EXT]], %[[D1_EXT]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENT_VECTOR]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<2xi64, #hipsr.mem<host>>

// CHECK-NEXT:      hipsr.preserve_shape %[[MM1_SHAPE]], %[[MM1]] : tensor<?xindex>, tensor<?x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[MM1_SHAPE]], %[[CAST]] : tensor<?xindex>, tensor<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[EXTENTS_SHAPE]], %[[EXTENTS]] : tensor<1xindex>, tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[CAST_INIT]], %[[CAST]], %[[EXTENTS_INIT]], %[[EXTENTS]], %[[W2]] : tensor<?x1xf32, #hipsr.mem<device>>, tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<?x1xf32, #hipsr.mem<device>>, tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>> {domain_id = 0 : i64}

// CHECK-NEXT:    %[[D1:.+]] = hipsr.pool_domain(%[[CTX]], %[[D0]]#0, %[[D0]]#2, %[[D0]]#1, %[[D0]]#3, %[[D0]]#4 : !hipsr.context, tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[D1_CTX:.+]]: !hipsr.context, %[[D1_CAST_INIT:.+]]: tensor<?x1xf32, #hipsr.mem<device>>, %[[D1_EXTENTS_INIT:.+]]: tensor<2xi64, #hipsr.mem<host>>, %[[D1_CAST:.+]]: tensor<?x1xf32, #hipsr.mem<device>>, %[[D1_EXTENTS:.+]]: tensor<2xi64, #hipsr.mem<host>>, %[[D1_W2:.+]]: tensor<4x2xf32, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[D1_C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[D1_C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[EXTRACT_I1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[EXTRACT_I0:.+]] = arith.constant 0 : index

// The barrier reads the host vector domain 0 wrote, which is where the dynamic
// extent crosses back into the shape graph.
// CHECK-NEXT:      %[[IN_SHAPE:.+]] = shape.shape_of %[[D1_CAST_INIT]] : tensor<?x1xf32, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[E0:.+]] = tensor.extract %[[D1_EXTENTS_INIT]]{{\[}}%[[EXTRACT_I0]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X0:.+]] = arith.index_cast %[[E0]] : i64 to index
// CHECK-NEXT:      %[[E1:.+]] = tensor.extract %[[D1_EXTENTS_INIT]]{{\[}}%[[EXTRACT_I1]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X1:.+]] = arith.index_cast %[[E1]] : i64 to index
// CHECK-NEXT:      %[[REQ:.+]] = tensor.from_elements %[[X0]], %[[X1]] : tensor<2xindex>
// CHECK-NEXT:      %[[EXPAND_SHAPE:.+]] = shape.broadcast %[[IN_SHAPE]], %[[REQ]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>

// CHECK-NEXT:      %[[W2_SHAPE:.+]] = shape.shape_of %[[D1_W2]] : tensor<4x2xf32, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[E_BATCH:.+]], %[[E_TAIL:.+]] = "shape.split_at"(%[[EXPAND_SHAPE]], %[[D1_C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[W2_BATCH:.+]], %[[W2_TAIL:.+]] = "shape.split_at"(%[[W2_SHAPE]], %[[D1_C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[BATCH2:.+]] = shape.broadcast %[[E_BATCH]], %[[W2_BATCH]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:      %[[M2:.+]] = shape.get_extent %[[EXPAND_SHAPE]], %[[D1_C0]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[N2:.+]] = shape.get_extent %[[W2_SHAPE]], %[[D1_C1]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[MATRIX2:.+]] = tensor.from_elements %[[M2]], %[[N2]] : tensor<2xindex>
// CHECK-NEXT:      %[[MM2_SHAPE:.+]] = tensor.concat dim(0) %[[BATCH2]], %[[MATRIX2]] : (tensor<?xindex>, tensor<2xindex>) -> tensor<?xindex>

// Each allocation reads the shape that sized it, so the expand's rows come from
// the broadcast and the matmul's from the concat.
// CHECK-NEXT:      %[[EXPAND_ROWS:.+]] = shape.get_extent %[[EXPAND_SHAPE]], %[[D1_C0]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[EXPAND_INIT:.+]] = tensor.empty(%[[EXPAND_ROWS]]) : tensor<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MM2_ROWS:.+]] = shape.get_extent %[[MM2_SHAPE]], %[[D1_C0]] : tensor<?xindex>, index -> index
// CHECK-NEXT:      %[[MM2_INIT:.+]] = tensor.empty(%[[MM2_ROWS]]) : tensor<?x2xf32, #hipsr.mem<device>>

// CHECK-NEXT:      %[[EXPAND:.+]] = hipsr.expand(%[[D1_CTX]]) ins(%[[D1_CAST]], %[[D1_EXTENTS]] : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[EXPAND_INIT]] : tensor<?x4xf32, #hipsr.mem<device>>) : tensor<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MM2:.+]] = hipsr.matmul(%[[D1_CTX]]) ins(%[[EXPAND]], %[[D1_W2]] : tensor<?x4xf32, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) outs(%[[MM2_INIT]] : tensor<?x2xf32, #hipsr.mem<device>>) : tensor<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[EXPAND_SHAPE]], %[[EXPAND]] : tensor<2xindex>, tensor<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[MM2_SHAPE]], %[[MM2]] : tensor<?xindex>, tensor<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[MM2]] : tensor<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<?x2xf32, #hipsr.mem<device>> {domain_id = 1 : i64}

// CHECK-NEXT:    return %[[D1]] : tensor<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @main_graph(%a: tensor<?x3xf16> {onnx.name = "a"},
                      %b: tensor<?x4xf32> {onnx.name = "b"})
    -> (tensor<?x2xf32> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %w1 = "onnx.Constant"() {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : () -> tensor<3x1xf16>
  %mm1 = "onnx.MatMul"(%a, %w1) : (tensor<?x3xf16>, tensor<3x1xf16>)
      -> tensor<?x1xf16>
  %cast = "onnx.Cast"(%mm1) {to = f32} : (tensor<?x1xf16>) -> tensor<?x1xf32>
  %shape = "onnx.Shape"(%b) : (tensor<?x4xf32>) -> tensor<2xi64>
  %expand = "onnx.Expand"(%cast, %shape)
      : (tensor<?x1xf32>, tensor<2xi64>) -> tensor<?x4xf32>
  %w2 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0],
                                          [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>}
      : () -> tensor<4x2xf32>
  %y = "onnx.MatMul"(%expand, %w2) : (tensor<?x4xf32>, tensor<4x2xf32>)
      -> tensor<?x2xf32>
  "onnx.Return"(%y) : (tensor<?x2xf32>) -> ()
}
