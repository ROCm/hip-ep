// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The same graph as sample_static.mlir, but with a dynamic leading extent. It
// enters the shape graph as a memref.dim, and every allocation reads its row
// count back out of the shape buffer that sized it.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// The checks cover every output line, so a new alloc or copy fails the test.
// CHECK-LABEL:   func.func @main_graph(
// CHECK-SAME:      %[[ARG0:[^:,]*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:[^:,]*]]: memref<?x3xf16, #hipsr.mem<device>> {onnx.name = "a"},
// CHECK-SAME:      %[[ARG2:[^:,]*]]: memref<?x4xf32, #hipsr.mem<device>> {onnx.name = "b"}) -> (memref<?x2xf32,
// CHECK-SAME:      #hipsr.mem<device>> {onnx.name = "y"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:      %[[CONSTANT_0:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[CONSTANT_1:.*]] = hipsr.constant {value = dense<{{\[}}{{\[}}1.000000e+00, 2.000000e+00], {{\[}}3.000000e+00, 4.000000e+00], {{\[}}5.000000e+00, 6.000000e+00], {{\[}}7.000000e+00, 8.000000e+00]]> : tensor<4x2xf32>} : memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_2:.*]] = hipsr.constant {value = dense<{{\[}}{{\[}}1.000000e+00], {{\[}}2.000000e+00], {{\[}}3.000000e+00]]> : tensor<3x1xf16>} : memref<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[ALLOC_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_5]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_4]]] : memref<1xindex>
// CHECK-NEXT:      %[[DIM_0:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_4]] : memref<?x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_4]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_3]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_0]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_4]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_3]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_3]]] : memref<2xindex>
// CHECK-NEXT:      %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_1]]{{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_3]]] : memref<2xindex> to memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:      %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_2]]{{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_4]]] {{\[}}%[[CONSTANT_3]]] : memref<2xindex> to memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:      %[[ALLOC_3:.*]] = memref.alloc(%[[CONSTANT_4]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_4]] step %[[CONSTANT_3]] {
// CHECK-NEXT:        %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:        %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[CONSTANT_3]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_0:.*]] = memref.load %[[SUBVIEW_0]]{{\[}}%[[VAL_0]]] : memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:          scf.yield %[[LOAD_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:        %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (index) {
// CHECK-NEXT:          scf.yield %[[IF_0]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_0:.*]] = memref.load %[[SUBVIEW_1]]{{\[}}%[[VAL_0]]] : memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:          %[[CMPI_2:.*]] = arith.cmpi eq, %[[LOAD_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:          %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[IF_0]], %[[LOAD_0]] : index
// CHECK-NEXT:          scf.yield %[[SELECT_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %[[IF_1]], %[[ALLOC_3]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[DIM_1:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_4]] : memref<?x3xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_1]], %[[ALLOC_4]]{{\[}}%[[CONSTANT_4]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_3]], %[[ALLOC_4]]{{\[}}%[[CONSTANT_3]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_5:.*]] = memref.alloc(%[[CONSTANT_5]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      %[[SUBVIEW_2:.*]] = memref.subview %[[ALLOC_5]]{{\[}}0] {{\[}}%[[CONSTANT_4]]] {{\[}}1] : memref<?xindex> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      memref.copy %[[ALLOC_3]], %[[SUBVIEW_2]] : memref<?xindex> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      %[[SUBVIEW_3:.*]] = memref.subview %[[ALLOC_5]]{{\[}}%[[CONSTANT_4]]] {{\[}}2] {{\[}}1] : memref<?xindex> to memref<2xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      memref.copy %[[ALLOC_4]], %[[SUBVIEW_3]] : memref<2xindex> to memref<2xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      %[[LOAD_1:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_4]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_2:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_4]]] : memref<?xindex>
// CHECK-NEXT:      %[[CONSTANT_6:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[MULI_0:.*]] = arith.muli %[[CONSTANT_6]], %[[LOAD_1]] : index
// CHECK-NEXT:      %[[CONSTANT_7:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_8:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[CONSTANT_8]] : index
// CHECK-NEXT:      %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_7]] : index
// CHECK-NEXT:      %[[MULI_1:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_7]] : index
// CHECK-NEXT:      %[[CONSTANT_9:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[MULI_2:.*]] = arith.muli %[[CONSTANT_9]], %[[LOAD_2]] : index
// CHECK-NEXT:      %[[CONSTANT_10:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_11:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_1:.*]] = arith.addi %[[MULI_2]], %[[CONSTANT_11]] : index
// CHECK-NEXT:      %[[DIVUI_1:.*]] = arith.divui %[[ADDI_1]], %[[CONSTANT_10]] : index
// CHECK-NEXT:      %[[MULI_3:.*]] = arith.muli %[[DIVUI_1]], %[[CONSTANT_10]] : index
// CHECK-NEXT:      %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ADDI_2:.*]] = arith.addi %[[MULI_1]], %[[MULI_3]] : index
// CHECK-NEXT:      %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[ARG0]], %[[ADDI_2]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_12]]]{{\[}}%[[LOAD_1]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_1:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[MULI_1]]]{{\[}}%[[LOAD_2]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.matmul(%[[ARG0]]) ins(%[[ARG1]], %[[CONSTANT_2]] : memref<?x3xf16, #hipsr.mem<device>>, memref<3x1xf16, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<?x1xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.cast(%[[ARG0]]) ins(%[[VIEW_0]] : memref<?x1xf16, #hipsr.mem<device>>) outs(%[[VIEW_1]] : memref<?x1xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      %[[CONSTANT_13:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_14:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[CONSTANT_15:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM_2:.*]] = memref.dim %[[ARG2]], %[[CONSTANT_15]] : memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK-NEXT:      memref.store %[[INDEX_CAST_0]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_15]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[CONSTANT_14]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_13]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_5]], %[[VIEW_0]] : memref<?xindex>, memref<?x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_5]], %[[VIEW_1]] : memref<?xindex>, memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_0]], %[[ALLOC_6]] : memref<1xindex>, memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[CONSTANT_16:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_17:.*]] = arith.constant 4 : index
// CHECK-NEXT:      %[[CONSTANT_18:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[CONSTANT_19:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM_3:.*]] = memref.dim %[[VIEW_1]], %[[CONSTANT_18]] : memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_3]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_18]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_19]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_19]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_3:.*]] = memref.load %[[ALLOC_6]]{{\[}}%[[CONSTANT_18]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_1:.*]] = arith.index_cast %[[LOAD_3]] : i64 to index
// CHECK-NEXT:      %[[LOAD_4:.*]] = memref.load %[[ALLOC_6]]{{\[}}%[[CONSTANT_19]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_2:.*]] = arith.index_cast %[[LOAD_4]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_1]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_18]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_2]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_19]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_9:.*]] = memref.alloc(%[[CONSTANT_16]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      scf.for %[[VAL_0:.*]] = %[[CONSTANT_18]] to %[[CONSTANT_16]] step %[[CONSTANT_19]] {
// CHECK-NEXT:        %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_18]] : index
// CHECK-NEXT:        %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[CONSTANT_19]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_0:.*]] = memref.load %[[ALLOC_7]]{{\[}}%[[VAL_0]]] : memref<2xindex>
// CHECK-NEXT:          scf.yield %[[LOAD_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_18]] : index
// CHECK-NEXT:        %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (index) {
// CHECK-NEXT:          scf.yield %[[IF_0]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_0:.*]] = memref.load %[[ALLOC_8]]{{\[}}%[[VAL_0]]] : memref<2xindex>
// CHECK-NEXT:          %[[CMPI_2:.*]] = arith.cmpi eq, %[[LOAD_0]], %[[CONSTANT_19]] : index
// CHECK-NEXT:          %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[IF_0]], %[[LOAD_0]] : index
// CHECK-NEXT:          scf.yield %[[SELECT_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %[[IF_1]], %[[ALLOC_9]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[CAST_0:.*]] = memref.cast %[[ALLOC_9]] : memref<?xindex> to memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_17]], %[[ALLOC_10]]{{\[}}%[[CONSTANT_18]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_16]], %[[ALLOC_10]]{{\[}}%[[CONSTANT_19]]] : memref<2xindex>
// CHECK-NEXT:      %[[SUBVIEW_4:.*]] = memref.subview %[[CAST_0]]{{\[}}%[[CONSTANT_18]]] {{\[}}%[[CONSTANT_18]]] {{\[}}%[[CONSTANT_19]]] : memref<2xindex> to memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:      %[[SUBVIEW_5:.*]] = memref.subview %[[ALLOC_10]]{{\[}}%[[CONSTANT_18]]] {{\[}}%[[CONSTANT_18]]] {{\[}}%[[CONSTANT_19]]] : memref<2xindex> to memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:      %[[ALLOC_11:.*]] = memref.alloc(%[[CONSTANT_18]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      scf.for %[[VAL_0:.*]] = %[[CONSTANT_18]] to %[[CONSTANT_18]] step %[[CONSTANT_19]] {
// CHECK-NEXT:        %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_18]] : index
// CHECK-NEXT:        %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[CONSTANT_19]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_0:.*]] = memref.load %[[SUBVIEW_4]]{{\[}}%[[VAL_0]]] : memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:          scf.yield %[[LOAD_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_18]] : index
// CHECK-NEXT:        %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (index) {
// CHECK-NEXT:          scf.yield %[[IF_0]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_0:.*]] = memref.load %[[SUBVIEW_5]]{{\[}}%[[VAL_0]]] : memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:          %[[CMPI_2:.*]] = arith.cmpi eq, %[[LOAD_0]], %[[CONSTANT_19]] : index
// CHECK-NEXT:          %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[IF_0]], %[[LOAD_0]] : index
// CHECK-NEXT:          scf.yield %[[SELECT_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %[[IF_1]], %[[ALLOC_11]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[LOAD_5:.*]] = memref.load %[[ALLOC_9]]{{\[}}%[[CONSTANT_18]]] : memref<?xindex>
// CHECK-NEXT:      %[[ALLOC_12:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[LOAD_5]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_18]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_16]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_19]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_13:.*]] = memref.alloc(%[[CONSTANT_16]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      %[[SUBVIEW_6:.*]] = memref.subview %[[ALLOC_13]]{{\[}}0] {{\[}}%[[CONSTANT_18]]] {{\[}}1] : memref<?xindex> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      memref.copy %[[ALLOC_11]], %[[SUBVIEW_6]] : memref<?xindex> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      %[[SUBVIEW_7:.*]] = memref.subview %[[ALLOC_13]]{{\[}}%[[CONSTANT_18]]] {{\[}}2] {{\[}}1] : memref<?xindex> to memref<2xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      memref.copy %[[ALLOC_12]], %[[SUBVIEW_7]] : memref<2xindex> to memref<2xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      %[[LOAD_6:.*]] = memref.load %[[ALLOC_9]]{{\[}}%[[CONSTANT_18]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_7:.*]] = memref.load %[[ALLOC_13]]{{\[}}%[[CONSTANT_18]]] : memref<?xindex>
// CHECK-NEXT:      %[[CONSTANT_20:.*]] = arith.constant 16 : index
// CHECK-NEXT:      %[[MULI_4:.*]] = arith.muli %[[CONSTANT_20]], %[[LOAD_6]] : index
// CHECK-NEXT:      %[[CONSTANT_21:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_22:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_3:.*]] = arith.addi %[[MULI_4]], %[[CONSTANT_22]] : index
// CHECK-NEXT:      %[[DIVUI_2:.*]] = arith.divui %[[ADDI_3]], %[[CONSTANT_21]] : index
// CHECK-NEXT:      %[[MULI_5:.*]] = arith.muli %[[DIVUI_2]], %[[CONSTANT_21]] : index
// CHECK-NEXT:      %[[CONSTANT_23:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[GET_POOL_1:.*]] = hipsr.get_pool(%[[ARG0]], %[[MULI_5]]) {domain_id = 1 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_2:.*]] = memref.view %[[GET_POOL_1]]{{\[}}%[[CONSTANT_23]]]{{\[}}%[[LOAD_6]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_24:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LOAD_8:.*]] = memref.load %[[ALLOC_13]]{{\[}}%[[CONSTANT_24]]] : memref<?xindex>
// CHECK-NEXT:      %[[ALLOC_OUTPUT_0:.*]] = hipsr.alloc_output(%[[ARG0]], %[[LOAD_8]]) {out_idx = 0 : i64} : memref<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.expand(%[[ARG0]]) ins(%[[VIEW_1]], %[[ALLOC_6]] : memref<?x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>) outs(%[[VIEW_2]] : memref<?x4xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.matmul(%[[ARG0]]) ins(%[[VIEW_2]], %[[CONSTANT_1]] : memref<?x4xf32, #hipsr.mem<device>>, memref<4x2xf32, #hipsr.mem<device>>) outs(%[[ALLOC_OUTPUT_0]] : memref<?x2xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_0]], %[[VIEW_2]] : memref<2xindex>, memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_13]], %[[ALLOC_OUTPUT_0]] : memref<?xindex>, memref<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      return %[[ALLOC_OUTPUT_0]] : memref<?x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    }


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
