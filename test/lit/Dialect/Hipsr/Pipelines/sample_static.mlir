// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The hipsr pipeline on a graph where every extent is static. Each extent is a
// constant, so no allocation reads its size back out of a shape buffer.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// The checks cover every output line, so a new alloc or copy fails the test.
// CHECK-LABEL:   func.func @main_graph(
// CHECK-SAME:      %[[ARG0:[^:,]*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:[^:,]*]]: memref<2x3xf16, #hipsr.mem<device>> {onnx.name = "a"},
// CHECK-SAME:      %[[ARG2:[^:,]*]]: memref<2x4xf32, #hipsr.mem<device>> {onnx.name = "b"}) -> (memref<2x2xf32, #hipsr.mem<device>> {onnx.name = "y"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:      %[[CONSTANT_0:.*]] = arith.constant 512 : index
// CHECK-NEXT:      %[[CONSTANT_1:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_2:.*]] = arith.constant 2 : i64
// CHECK-NEXT:      %[[CONSTANT_3:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[CONSTANT_4:.*]] = hipsr.constant {value = dense<{{\[}}{{\[}}1.000000e+00, 2.000000e+00], {{\[}}3.000000e+00, 4.000000e+00], {{\[}}5.000000e+00, 6.000000e+00], {{\[}}7.000000e+00, 8.000000e+00]]> : tensor<4x2xf32>} : memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_5:.*]] = hipsr.constant {value = dense<{{\[}}{{\[}}1.000000e+00], {{\[}}2.000000e+00], {{\[}}3.000000e+00]]> : tensor<3x1xf16>} : memref<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[CONSTANT_8:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[ALLOC_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_8]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_7]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_8]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_7]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_6]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      %[[CAST_0:.*]] = memref.cast %[[ALLOC_2]] : memref<2xindex> to memref<?xindex>
// CHECK-NEXT:      memref.copy %[[ALLOC_1]], %[[ALLOC_2]] : memref<2xindex> to memref<2xindex>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_1]] : memref<2xindex>
// CHECK-NEXT:      %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[ARG0]], %[[CONSTANT_0]]) {bufferization.manual_deallocation, domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_7]]]{{\[}}] : memref<?xi8, #hipsr.mem<device>> to memref<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_1:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_1]]]{{\[}}] : memref<?xi8, #hipsr.mem<device>> to memref<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.matmul(%[[ARG0]]) ins(%[[ARG1]], %[[CONSTANT_5]] : memref<2x3xf16, #hipsr.mem<device>>, memref<3x1xf16, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<2x1xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.cast(%[[ARG0]]) ins(%[[VIEW_0]] : memref<2x1xf16, #hipsr.mem<device>>) outs(%[[VIEW_1]] : memref<2x1xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.store %[[CONSTANT_2]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_7]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[CONSTANT_3]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_6]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_0]], %[[VIEW_0]] : memref<?xindex>, memref<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_0]], %[[VIEW_1]] : memref<?xindex>, memref<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_2]] : memref<2xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_0]], %[[ALLOC_3]] : memref<1xindex>, memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_0]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_8]], %[[ALLOC_4]]{{\[}}%[[CONSTANT_7]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_6]], %[[ALLOC_4]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_0:.*]] = memref.load %[[ALLOC_3]]{{\[}}%[[CONSTANT_7]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_0:.*]] = arith.index_cast %[[LOAD_0]] : i64 to index
// CHECK-NEXT:      %[[LOAD_1:.*]] = memref.load %[[ALLOC_3]]{{\[}}%[[CONSTANT_6]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_1:.*]] = arith.index_cast %[[LOAD_1]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_0]], %[[ALLOC_5]]{{\[}}%[[CONSTANT_7]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_1]], %[[ALLOC_5]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      scf.for %arg3 = %[[CONSTANT_7]] to %[[CONSTANT_8]] step %[[CONSTANT_6]] {
// CHECK-NEXT:      %[[CMPI_0:.*]] = arith.cmpi ult, %arg3, %[[CONSTANT_7]] : index
// CHECK-NEXT:      %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:      scf.yield %[[CONSTANT_6]] : index
// CHECK-NEXT:      } else {
// CHECK-NEXT:      %[[LOAD_2:.*]] = memref.load %[[ALLOC_4]]{{\[}}%arg3] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_3:.*]] = memref.load %[[ALLOC_5]]{{\[}}%arg3] : memref<2xindex>
// CHECK-NEXT:      %[[CMPI_1:.*]] = arith.cmpi eq, %[[LOAD_3]], %[[CONSTANT_6]] : index
// CHECK-NEXT:      %[[ARITH_SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[LOAD_2]], %[[LOAD_3]] : index
// CHECK-NEXT:      scf.yield %[[ARITH_SELECT_0]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %[[IF_0]], %[[ALLOC_6]]{{\[}}%arg3] : memref<2xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[ALLOC_5]] : memref<2xindex>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_4]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_4:.*]] = memref.load %[[ALLOC_6]]{{\[}}%[[CONSTANT_7]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[LOAD_4]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_7]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_8]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      %[[CAST_1:.*]] = memref.cast %[[ALLOC_8]] : memref<2xindex> to memref<?xindex>
// CHECK-NEXT:      memref.copy %[[ALLOC_7]], %[[ALLOC_8]] : memref<2xindex> to memref<2xindex>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_7]] : memref<2xindex>
// CHECK-NEXT:      %[[GET_POOL_1:.*]] = hipsr.get_pool(%[[ARG0]], %[[CONSTANT_1]]) {bufferization.manual_deallocation, domain_id = 1 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_2:.*]] = memref.view %[[GET_POOL_1]]{{\[}}%[[CONSTANT_7]]]{{\[}}] : memref<?xi8, #hipsr.mem<device>> to memref<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_OUTPUT_0:.*]] = hipsr.alloc_output(%[[ARG0]]) {out_idx = 0 : i64} : memref<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.expand(%[[ARG0]]) ins(%[[VIEW_1]], %[[ALLOC_3]] : memref<2x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>) outs(%[[VIEW_2]] : memref<2x4xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.dealloc %[[ALLOC_3]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.matmul(%[[ARG0]]) ins(%[[VIEW_2]], %[[CONSTANT_4]] : memref<2x4xf32, #hipsr.mem<device>>, memref<4x2xf32, #hipsr.mem<device>>) outs(%[[ALLOC_OUTPUT_0]] : memref<2x2xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_6]], %[[VIEW_2]] : memref<2xindex>, memref<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_6]] : memref<2xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_1]], %[[ALLOC_OUTPUT_0]] : memref<?xindex>, memref<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_8]] : memref<2xindex>
// CHECK-NEXT:      return %[[ALLOC_OUTPUT_0]] : memref<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      }

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
