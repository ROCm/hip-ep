// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The same graph as sample_static.mlir, but with a dynamic leading extent. It
// enters the shape graph as a memref.dim, and every allocation reads its row
// count back out of the shape buffer that sized it.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// CHECK-LABEL:   func.func @main_graph(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<?x3xf16, #hipsr.mem<device>> {onnx.name = "a"},
// CHECK-SAME:      %[[ARG2:.*]]: memref<?x4xf32, #hipsr.mem<device>> {onnx.name = "b"}) -> (memref<?x2xf32, #hipsr.mem<device>> {onnx.name = "y"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]]:5 = hipsr.pool_domain(%[[ARG0]], %[[ARG1]], %[[ARG2]] : !hipsr.context, memref<?x3xf16, #hipsr.mem<device>>, memref<?x4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<?x3xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: memref<?x4xf32, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[CONSTANT_0:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[CONSTANT_1:.*]] = hipsr.constant {value = dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<4x2xf32>} : memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CONSTANT_2:.*]] = hipsr.constant {value = dense<{{\[\[}}1.000000e+00], [2.000000e+00], [3.000000e+00]]> : tensor<3x1xf16>} : memref<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[ALLOC_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_5]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_4]]] : memref<1xindex>
// CHECK-NEXT:        %[[DIM_0:.*]] = memref.dim %[[VAL_1]], %[[CONSTANT_4]] : memref<?x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_4]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_3]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_0]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_4]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_3]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_3]]] : memref<2xindex>
// CHECK-NEXT:        %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_1]]{{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_3]]] : memref<2xindex> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:        %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_2]]{{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_3]]] : memref<2xindex> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:        %[[ALLOC_3:.*]] = memref.alloc(%[[CONSTANT_4]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        scf.for %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_4]] step %[[CONSTANT_3]] {
// CHECK-NEXT:          %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-NEXT:          %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:            scf.yield %[[CONSTANT_3]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_0:.*]] = memref.load %[[SUBVIEW_0]]{{\[}}%[[VAL_3]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:            scf.yield %[[LOAD_0]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-NEXT:          %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (index) {
// CHECK-NEXT:            scf.yield %[[IF_0]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_1:.*]] = memref.load %[[SUBVIEW_1]]{{\[}}%[[VAL_3]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:            %[[CMPI_2:.*]] = arith.cmpi eq, %[[LOAD_1]], %[[CONSTANT_3]] : index
// CHECK-NEXT:            %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[IF_0]], %[[LOAD_1]] : index
// CHECK-NEXT:            scf.yield %[[SELECT_0]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.store %[[IF_1]], %[[ALLOC_3]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[DIM_1:.*]] = memref.dim %[[VAL_1]], %[[CONSTANT_4]] : memref<?x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_1]], %[[ALLOC_4]]{{\[}}%[[CONSTANT_4]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_3]], %[[ALLOC_4]]{{\[}}%[[CONSTANT_3]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_5:.*]] = memref.alloc(%[[CONSTANT_5]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        %[[SUBVIEW_2:.*]] = memref.subview %[[ALLOC_5]][0] {{\[}}%[[CONSTANT_4]]] [1] : memref<?xindex> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        memref.copy %[[ALLOC_3]], %[[SUBVIEW_2]] : memref<?xindex> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        %[[SUBVIEW_3:.*]] = memref.subview %[[ALLOC_5]]{{\[}}%[[CONSTANT_4]]] [2] [1] : memref<?xindex> to memref<2xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        memref.copy %[[ALLOC_4]], %[[SUBVIEW_3]] : memref<2xindex> to memref<2xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[LOAD_2:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_4]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_6:.*]] = memref.alloc(%[[LOAD_2]]) {alignment = 64 : i64} : memref<?x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[LOAD_3:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_4]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_7:.*]] = memref.alloc(%[[LOAD_3]]) {alignment = 64 : i64} : memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.matmul(%[[VAL_0]]) ins(%[[VAL_1]], %[[CONSTANT_2]] : memref<?x3xf16, #hipsr.mem<device>>, memref<3x1xf16, #hipsr.mem<device>>) outs(%[[ALLOC_6]] : memref<?x1xf16, #hipsr.mem<device>>)
// CHECK-NEXT:        %[[ALLOC_9:.*]] = memref.alloc(%[[LOAD_3]]) {alignment = 64 : i64} : memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.cast(%[[VAL_0]]) ins(%[[ALLOC_6]] : memref<?x1xf16, #hipsr.mem<device>>) outs(%[[ALLOC_9]] : memref<?x1xf32, #hipsr.mem<device>>)
// CHECK-NEXT:        %[[COMPUTE_0:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[VAL_2]] : memref<?x4xf32, #hipsr.mem<device>>) outs(%[[ALLOC_8]] : memref<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_4:.*]]: !hipsr.context, %[[VAL_5:.*]]: memref<?x4xf32, #hipsr.mem<device>>, %[[VAL_6:.*]]: memref<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:          %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[CONSTANT_7:.*]] = arith.constant 4 : i64
// CHECK-NEXT:          %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[DIM_2:.*]] = memref.dim %[[VAL_5]], %[[CONSTANT_8]] : memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:          %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK-NEXT:          %[[ALLOC_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:          memref.store %[[INDEX_CAST_0]], %[[ALLOC_10]]{{\[}}%[[CONSTANT_8]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:          memref.store %[[CONSTANT_7]], %[[ALLOC_10]]{{\[}}%[[CONSTANT_6]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:          hipsr.compute_yield %[[ALLOC_10]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        } : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_5]], %[[ALLOC_6]] : memref<?xindex>, memref<?x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_5]], %[[ALLOC_9]] : memref<?xindex>, memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_0]], %[[COMPUTE_0]] : memref<1xindex>, memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ALLOC_7]], %[[ALLOC_9]], %[[ALLOC_8]], %[[COMPUTE_0]], %[[CONSTANT_1]] : memref<?x1xf32, #hipsr.mem<device>>, memref<?x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>, memref<2xi64, #hipsr.mem<host>>, memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      } -> memref<?x1xf32, #hipsr.mem<device>>, memref<?x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>, memref<2xi64, #hipsr.mem<host>>, memref<4x2xf32, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_1:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]]#0, %[[POOL_DOMAIN_0]]#2, %[[POOL_DOMAIN_0]]#1, %[[POOL_DOMAIN_0]]#3, %[[POOL_DOMAIN_0]]#4 : !hipsr.context, memref<?x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>, memref<?x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>, memref<4x2xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_8:.*]]: !hipsr.context, %[[VAL_9:.*]]: memref<?x1xf32, #hipsr.mem<device>>, %[[VAL_10:.*]]: memref<2xi64, #hipsr.mem<host>>, %[[VAL_11:.*]]: memref<?x1xf32, #hipsr.mem<device>>, %[[VAL_12:.*]]: memref<2xi64, #hipsr.mem<host>>, %[[VAL_13:.*]]: memref<4x2xf32, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[CONSTANT_9:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[CONSTANT_10:.*]] = arith.constant 4 : index
// CHECK-NEXT:        %[[CONSTANT_11:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[CONSTANT_12:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[DIM_3:.*]] = memref.dim %[[VAL_9]], %[[CONSTANT_11]] : memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_11:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_3]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_12]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:        %[[LOAD_4:.*]] = memref.load %[[VAL_10]]{{\[}}%[[CONSTANT_11]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_1:.*]] = arith.index_cast %[[LOAD_4]] : i64 to index
// CHECK-NEXT:        %[[LOAD_5:.*]] = memref.load %[[VAL_10]]{{\[}}%[[CONSTANT_12]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_2:.*]] = arith.index_cast %[[LOAD_5]] : i64 to index
// CHECK-NEXT:        %[[ALLOC_12:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_1]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_2]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_13:.*]] = memref.alloc(%[[CONSTANT_9]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        scf.for %[[VAL_14:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_9]] step %[[CONSTANT_12]] {
// CHECK-NEXT:          %[[CMPI_3:.*]] = arith.cmpi ult, %[[VAL_14]], %[[CONSTANT_11]] : index
// CHECK-NEXT:          %[[IF_2:.*]] = scf.if %[[CMPI_3]] -> (index) {
// CHECK-NEXT:            scf.yield %[[CONSTANT_12]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_6:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[VAL_14]]] : memref<2xindex>
// CHECK-NEXT:            scf.yield %[[LOAD_6]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[CMPI_4:.*]] = arith.cmpi ult, %[[VAL_14]], %[[CONSTANT_11]] : index
// CHECK-NEXT:          %[[IF_3:.*]] = scf.if %[[CMPI_4]] -> (index) {
// CHECK-NEXT:            scf.yield %[[IF_2]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_7:.*]] = memref.load %[[ALLOC_12]]{{\[}}%[[VAL_14]]] : memref<2xindex>
// CHECK-NEXT:            %[[CMPI_5:.*]] = arith.cmpi eq, %[[LOAD_7]], %[[CONSTANT_12]] : index
// CHECK-NEXT:            %[[SELECT_1:.*]] = arith.select %[[CMPI_5]], %[[IF_2]], %[[LOAD_7]] : index
// CHECK-NEXT:            scf.yield %[[SELECT_1]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.store %[[IF_3]], %[[ALLOC_13]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CAST_0:.*]] = memref.cast %[[ALLOC_13]] : memref<?xindex> to memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_14:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_10]], %[[ALLOC_14]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_9]], %[[ALLOC_14]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:        %[[SUBVIEW_4:.*]] = memref.subview %[[CAST_0]]{{\[}}%[[CONSTANT_11]]] {{\[}}%[[CONSTANT_11]]] {{\[}}%[[CONSTANT_12]]] : memref<2xindex> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:        %[[SUBVIEW_5:.*]] = memref.subview %[[ALLOC_14]]{{\[}}%[[CONSTANT_11]]] {{\[}}%[[CONSTANT_11]]] {{\[}}%[[CONSTANT_12]]] : memref<2xindex> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:        %[[ALLOC_15:.*]] = memref.alloc(%[[CONSTANT_11]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        scf.for %[[VAL_15:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_11]] step %[[CONSTANT_12]] {
// CHECK-NEXT:          %[[CMPI_6:.*]] = arith.cmpi ult, %[[VAL_15]], %[[CONSTANT_11]] : index
// CHECK-NEXT:          %[[IF_4:.*]] = scf.if %[[CMPI_6]] -> (index) {
// CHECK-NEXT:            scf.yield %[[CONSTANT_12]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_8:.*]] = memref.load %[[SUBVIEW_4]]{{\[}}%[[VAL_15]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:            scf.yield %[[LOAD_8]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[CMPI_7:.*]] = arith.cmpi ult, %[[VAL_15]], %[[CONSTANT_11]] : index
// CHECK-NEXT:          %[[IF_5:.*]] = scf.if %[[CMPI_7]] -> (index) {
// CHECK-NEXT:            scf.yield %[[IF_4]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_9:.*]] = memref.load %[[SUBVIEW_5]]{{\[}}%[[VAL_15]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:            %[[CMPI_8:.*]] = arith.cmpi eq, %[[LOAD_9]], %[[CONSTANT_12]] : index
// CHECK-NEXT:            %[[SELECT_2:.*]] = arith.select %[[CMPI_8]], %[[IF_4]], %[[LOAD_9]] : index
// CHECK-NEXT:            scf.yield %[[SELECT_2]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.store %[[IF_5]], %[[ALLOC_15]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[LOAD_10:.*]] = memref.load %[[ALLOC_13]]{{\[}}%[[CONSTANT_11]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_16:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[LOAD_10]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_9]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_17:.*]] = memref.alloc(%[[CONSTANT_9]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        %[[SUBVIEW_6:.*]] = memref.subview %[[ALLOC_17]][0] {{\[}}%[[CONSTANT_11]]] [1] : memref<?xindex> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        memref.copy %[[ALLOC_15]], %[[SUBVIEW_6]] : memref<?xindex> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        %[[SUBVIEW_7:.*]] = memref.subview %[[ALLOC_17]]{{\[}}%[[CONSTANT_11]]] [2] [1] : memref<?xindex> to memref<2xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        memref.copy %[[ALLOC_16]], %[[SUBVIEW_7]] : memref<2xindex> to memref<2xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[LOAD_11:.*]] = memref.load %[[ALLOC_13]]{{\[}}%[[CONSTANT_11]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_18:.*]] = memref.alloc(%[[LOAD_11]]) {alignment = 64 : i64} : memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[LOAD_12:.*]] = memref.load %[[ALLOC_17]]{{\[}}%[[CONSTANT_11]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_19:.*]] = memref.alloc(%[[LOAD_12]]) {alignment = 64 : i64} : memref<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.expand(%[[VAL_8]]) ins(%[[VAL_11]], %[[VAL_12]] : memref<?x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>) outs(%[[ALLOC_18]] : memref<?x4xf32, #hipsr.mem<device>>)
// CHECK-NEXT:        hipsr.matmul(%[[VAL_8]]) ins(%[[ALLOC_18]], %[[VAL_13]] : memref<?x4xf32, #hipsr.mem<device>>, memref<4x2xf32, #hipsr.mem<device>>) outs(%[[ALLOC_19]] : memref<?x2xf32, #hipsr.mem<device>>)
// CHECK-NEXT:        hipsr.preserve_shape %[[CAST_0]], %[[ALLOC_18]] : memref<2xindex>, memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_17]], %[[ALLOC_19]] : memref<?xindex>, memref<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ALLOC_19]] : memref<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      } -> memref<?x2xf32, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_1]] : memref<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    }
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
