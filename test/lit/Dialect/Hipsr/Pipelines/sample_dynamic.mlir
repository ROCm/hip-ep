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

// CHECK-NEXT:    %[[W1:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x1xf16>} : tensor<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:    %[[MM1_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[W1]] : tensor<?x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x1xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[A_SHAPE:.+]]: !shape.shape, %[[W1_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT:      %[[A_K_IDX:.+]] = shape.const_size 1
// CHECK-NEXT:      %[[A_K:.+]] = shape.get_extent %[[A_SHAPE]], %[[A_K_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[A_K_SHAPE:.+]] = shape.from_extents %[[A_K]] : !shape.size
// CHECK-NEXT:      %[[W1_K_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[W1_K:.+]] = shape.get_extent %[[W1_SHAPE]], %[[W1_K_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[W1_K_SHAPE:.+]] = shape.from_extents %[[W1_K]] : !shape.size
// CHECK-NEXT:      %[[K_WITNESS:.+]] = shape.cstr_eq %[[A_K_SHAPE]], %[[W1_K_SHAPE]] : !shape.shape, !shape.shape
// CHECK-NEXT:      %[[A_SPLIT_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[A_BATCH:.+]], %[[A_TAIL:.+]] = "shape.split_at"(%[[A_SHAPE]], %[[A_SPLIT_IDX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:      %[[W1_SPLIT_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[W1_BATCH:.+]], %[[W1_TAIL:.+]] = "shape.split_at"(%[[W1_SHAPE]], %[[W1_SPLIT_IDX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:      %[[BATCH_WITNESS:.+]] = shape.cstr_broadcastable %[[A_BATCH]], %[[W1_BATCH]] : !shape.shape, !shape.shape
// CHECK-NEXT:      %[[WITNESS:.+]] = shape.assuming_all %[[K_WITNESS]], %[[BATCH_WITNESS]]
// CHECK-NEXT:      %[[MM1_SHAPE_VAL:.+]] = shape.assuming %[[WITNESS]] -> (!shape.shape) {
// CHECK-NEXT:        %[[BATCH:.+]] = shape.broadcast %[[A_BATCH]], %[[W1_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:        %[[M_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:        %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[M_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[N_IDX:.+]] = shape.const_size 1
// CHECK-NEXT:        %[[N:.+]] = shape.get_extent %[[W1_SHAPE]], %[[N_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[MATRIX:.+]] = shape.from_extents %[[M]], %[[N]] : !shape.size, !shape.size
// CHECK-NEXT:        %[[RESULT:.+]] = shape.concat %[[BATCH]], %[[MATRIX]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:        shape.assuming_yield %[[RESULT]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      hipsr.shape_yield %[[MM1_SHAPE_VAL]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[MM1:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[W1]] : tensor<?x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>) outs(%[[MM1_INIT]] : tensor<?x1xf16, #hipsr.mem<device>>) : tensor<?x1xf16, #hipsr.mem<device>>

// CHECK-NEXT:    %[[CAST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[MM1_INIT]] : tensor<?x1xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x1xf32, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[MM1_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT:      hipsr.shape_yield %[[MM1_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[CAST:.+]] = hipsr.cast(%[[CTX]]) ins(%[[MM1]] : tensor<?x1xf16, #hipsr.mem<device>>) outs(%[[CAST_INIT]] : tensor<?x1xf32, #hipsr.mem<device>>) : tensor<?x1xf32, #hipsr.mem<device>>

// CHECK-NEXT:    %[[SHAPE_INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:      %[[RANK:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[RANK_SHAPE:.+]] = shape.from_extents %[[RANK]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[RANK_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[SHAPE:.+]] = hipsr.compute(%[[CTX]]) ins(%[[B]] : tensor<?x4xf32, #hipsr.mem<device>>) outs(%[[SHAPE_INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[BODY_B:.+]]: tensor<?x4xf32, #hipsr.mem<device>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[DIM_IDX:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM:.+]] = tensor.dim %[[BODY_B]], %[[DIM_IDX]] : tensor<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[D0:.+]] = arith.index_cast %[[DIM]] : index to i64
// CHECK-NEXT:      %[[D1:.+]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[EXTENTS:.+]] = tensor.from_elements %[[D0]], %[[D1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:    } : tensor<2xi64, #hipsr.mem<host>>

// CHECK-NEXT:    %[[EXPAND_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[CAST_INIT]], %[[SHAPE_INIT]] : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x4xf32, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[IN:.+]]: tensor<?x1xf32, #hipsr.mem<device>>, %[[EXT:.+]]: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[IN_SHAPE:.+]] = shape.shape_of %[[IN]] : tensor<?x1xf32, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[I0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[E0:.+]] = tensor.extract %[[EXT]][%[[I0]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X0:.+]] = arith.index_cast %[[E0]] : i64 to index
// CHECK-NEXT:      %[[I1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[E1:.+]] = tensor.extract %[[EXT]][%[[I1]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X1:.+]] = arith.index_cast %[[E1]] : i64 to index
// CHECK-NEXT:      %[[REQ:.+]] = shape.from_extents %[[X0]], %[[X1]] : index, index
// CHECK-NEXT:      %[[EXPAND_WITNESS:.+]] = shape.cstr_broadcastable %[[IN_SHAPE]], %[[REQ]] : tensor<2xindex>, !shape.shape
// CHECK-NEXT:      %[[EXPAND_SHAPE:.+]] = shape.assuming %[[EXPAND_WITNESS]] -> (!shape.shape) {
// CHECK-NEXT:        %[[BROADCAST:.+]] = shape.broadcast %[[IN_SHAPE]], %[[REQ]] : tensor<2xindex>, !shape.shape -> !shape.shape
// CHECK-NEXT:        shape.assuming_yield %[[BROADCAST]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      hipsr.shape_yield %[[EXPAND_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[EXPAND:.+]] = hipsr.expand(%[[CTX]]) ins(%[[CAST]], %[[SHAPE]] : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[EXPAND_INIT]] : tensor<?x4xf32, #hipsr.mem<device>>) : tensor<?x4xf32, #hipsr.mem<device>>

// CHECK-NEXT:    %[[W2:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<4x2xf32>} : tensor<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    %[[MM2_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[EXPAND_INIT]], %[[W2]] : tensor<?x4xf32, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x2xf32, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[E_SHAPE:.+]]: !shape.shape, %[[W2_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT:      %[[E_K_IDX:.+]] = shape.const_size 1
// CHECK-NEXT:      %[[E_K:.+]] = shape.get_extent %[[E_SHAPE]], %[[E_K_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[E_K_SHAPE:.+]] = shape.from_extents %[[E_K]] : !shape.size
// CHECK-NEXT:      %[[W2_K_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[W2_K:.+]] = shape.get_extent %[[W2_SHAPE]], %[[W2_K_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[W2_K_SHAPE:.+]] = shape.from_extents %[[W2_K]] : !shape.size
// CHECK-NEXT:      %[[K2_WITNESS:.+]] = shape.cstr_eq %[[E_K_SHAPE]], %[[W2_K_SHAPE]] : !shape.shape, !shape.shape
// CHECK-NEXT:      %[[E_SPLIT_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[E_BATCH:.+]], %[[E_TAIL:.+]] = "shape.split_at"(%[[E_SHAPE]], %[[E_SPLIT_IDX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:      %[[W2_SPLIT_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:      %[[W2_BATCH:.+]], %[[W2_TAIL:.+]] = "shape.split_at"(%[[W2_SHAPE]], %[[W2_SPLIT_IDX]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:      %[[BATCH2_WITNESS:.+]] = shape.cstr_broadcastable %[[E_BATCH]], %[[W2_BATCH]] : !shape.shape, !shape.shape
// CHECK-NEXT:      %[[WITNESS2:.+]] = shape.assuming_all %[[K2_WITNESS]], %[[BATCH2_WITNESS]]
// CHECK-NEXT:      %[[MM2_SHAPE_VAL:.+]] = shape.assuming %[[WITNESS2]] -> (!shape.shape) {
// CHECK-NEXT:        %[[BATCH2:.+]] = shape.broadcast %[[E_BATCH]], %[[W2_BATCH]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:        %[[M2_IDX:.+]] = shape.const_size 0
// CHECK-NEXT:        %[[M2:.+]] = shape.get_extent %[[E_SHAPE]], %[[M2_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[N2_IDX:.+]] = shape.const_size 1
// CHECK-NEXT:        %[[N2:.+]] = shape.get_extent %[[W2_SHAPE]], %[[N2_IDX]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:        %[[MATRIX2:.+]] = shape.from_extents %[[M2]], %[[N2]] : !shape.size, !shape.size
// CHECK-NEXT:        %[[RESULT2:.+]] = shape.concat %[[BATCH2]], %[[MATRIX2]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:        shape.assuming_yield %[[RESULT2]] : !shape.shape
// CHECK-NEXT:      }
// CHECK-NEXT:      hipsr.shape_yield %[[MM2_SHAPE_VAL]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[Y:.+]] = hipsr.matmul(%[[CTX]]) ins(%[[EXPAND]], %[[W2]] : tensor<?x4xf32, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) outs(%[[MM2_INIT]] : tensor<?x2xf32, #hipsr.mem<device>>) : tensor<?x2xf32, #hipsr.mem<device>>

// CHECK-NEXT:    return %[[Y]] : tensor<?x2xf32, #hipsr.mem<device>>
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
