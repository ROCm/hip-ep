// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The hipsr pipeline on an embedding graph with dynamic shapes. NonZero makes
// the scatter index count depend on the data, so the pipeline cuts five pool
// domains. Each domain starts with a shape computation that reads a host
// buffer an earlier domain filled.
//
// Pool domains and what cuts them
// -------------------------------
//
// A domain is one pool allocation, so every buffer inside it must be sized
// before it runs. The pipeline therefore starts a new domain wherever a shape
// depends on a value the host cannot know until the previous domain has
// finished.
//
//   domain 0   collapse(image_features)              -> flat
//              equal(input_ids, 248056) -> unsqueeze -> mask
//              gather(table, input_ids)              -> embeds
//              shape(embeds)                         -> extents  host 3xi64
//                   |
//                   |  extents: the broadcast destination is not in any type
//                   v
//   domain 1   expand(mask, extents)                 -> mask3d
//                   |
//                   |  extents: the second broadcast reads them again
//                   v
//   domain 2   expand(mask3d, extents)               -> mask3d'
//              nonzero(mask3d')                      -> coords 3x?, count
//              copy_d2h(count)                       -> count    host 1xi64
//                   |
//                   |  count: how many coordinates the search actually found
//                   v
//   domain 3   extract_slice(coords, count)          -> coords 3x?
//              transpose(coords)                     -> coords ?x3
//              shape -> gather -> unsqueeze          -> window   host 1xi64
//                   |
//                   |  window: where the update slice ends
//                   v
//   domain 4   slice(flat, window)                   -> updates
//              scatter_nd(embeds, coords, updates)   -> inputs_embeds
//
// This is the only pipeline test that reaches shape.num_elements. Its two
// reducing scf.for loops show that shape-to-shape-lowering ran.
//
// The embedding table lives in an external file, so the RUN line creates a
// file of the right length to map. Only the length matters; nothing reads the
// weights.

// RUN: %python %S/../../../Inputs/make_external_data.py %t/embedding.onnx.data 2034237440 && cd %t && hip-mlir-opt --onnx-dialect=modeled --hipsr-pipeline --mlir-elide-resource-strings-if-larger=32 %s | FileCheck %s

// The checks cover every output line, so a new alloc or copy fails the test.
// CHECK-LABEL:   func.func @main_graph(
// CHECK-SAME:      %[[ARG0:[^:,]*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:[^:,]*]]: memref<?x?xi64, #hipsr.mem<device>> {onnx.name = "input_ids"},
// CHECK-SAME:      %[[ARG2:[^:,]*]]: memref<?x4096xf16, #hipsr.mem<device>> {onnx.name = "image_features"}) -> (memref<?x?x4096xf16, #hipsr.mem<device>> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:      %[[CONSTANT_0:.*]] = arith.constant 3 : i64
// CHECK-NEXT:      %[[CONSTANT_1:.*]] = arith.constant 24 : index
// CHECK-NEXT:      %[[CONSTANT_2:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:      %[[CONSTANT_3:.*]] = arith.constant 8192 : index
// CHECK-NEXT:      %[[CONSTANT_4:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[CONSTANT_5:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_6:.*]] = arith.constant 248320 : index
// CHECK-NEXT:      %[[CONSTANT_7:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_8:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[CONSTANT_9:.*]] = hipsr.constant {value = dense<248056> : tensor<i64>} : memref<i64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_10:.*]] = hipsr.constant {value = dense_resource<"file|embedding.onnx.data|0"> : tensor<248320x4096xf16, #hipsr.mem<device>>} : memref<248320x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[CONSTANT_13:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[ALLOC_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_13]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_12]]] : memref<1xindex>
// CHECK-NEXT:      %[[MEMREF_DIM_0:.*]] = memref.dim %[[ARG2]], %[[CONSTANT_12]] : memref<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[MEMREF_DIM_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_8]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[FOR_0:.*]] = scf.for %arg3 = %[[CONSTANT_12]] to %[[CONSTANT_7]] step %[[CONSTANT_11]] iter_args(%arg4 = %[[CONSTANT_11]]) -> (index) {
// CHECK-NEXT:      %[[LOAD_0:.*]] = memref.load %[[ALLOC_1]]{{\[}}%arg3] : memref<2xindex>
// CHECK-NEXT:      %[[MULI_0:.*]] = arith.muli %[[LOAD_0]], %arg4 : index
// CHECK-NEXT:      scf.yield %[[MULI_0]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[ALLOC_1]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[FOR_0]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_12]]] : memref<1xindex>
// CHECK-NEXT:      %[[MEMREF_DIM_1:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_12]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MEMREF_DIM_2:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_11]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[MEMREF_DIM_1]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[MEMREF_DIM_2]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<0xindex>
// CHECK-NEXT:      %[[ALLOC_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      scf.for %arg3 = %[[CONSTANT_12]] to %[[CONSTANT_7]] step %[[CONSTANT_11]] {
// CHECK-NEXT:      %[[LOAD_0:.*]] = arith.cmpi ult, %arg3, %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[MULI_0:.*]] = scf.if %[[LOAD_0]] -> (index) {
// CHECK-NEXT:      scf.yield %[[CONSTANT_11]] : index
// CHECK-NEXT:      } else {
// CHECK-NEXT:      %[[LOAD_1:.*]] = memref.load %[[ALLOC_3]]{{\[}}%arg3] : memref<2xindex>
// CHECK-NEXT:      scf.yield %[[LOAD_1]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[CMPI_0:.*]] = arith.cmpi ult, %arg3, %[[CONSTANT_7]] : index
// CHECK-NEXT:      %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:      scf.yield %[[MULI_0]] : index
// CHECK-NEXT:      } else {
// CHECK-NEXT:      %[[LOAD_1:.*]] = arith.subi %arg3, %[[CONSTANT_7]] : index
// CHECK-NEXT:      %[[LOAD_2:.*]] = memref.load %[[ALLOC_4]]{{\[}}%[[LOAD_1]]] : memref<0xindex>
// CHECK-NEXT:      %[[CMPI_1:.*]] = arith.cmpi eq, %[[LOAD_2]], %[[CONSTANT_11]] : index
// CHECK-NEXT:      %[[ARITH_SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[MULI_0]], %[[LOAD_2]] : index
// CHECK-NEXT:      scf.yield %[[ARITH_SELECT_0]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %[[IF_0]], %[[ALLOC_5]]{{\[}}%arg3] : memref<2xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[ALLOC_4]] : memref<0xindex>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_3]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_3:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_4:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_3]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_4]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_11]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_6]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_8]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[MEMREF_DIM_1]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[MEMREF_DIM_2]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_7]]{{\[}}1] {{\[}}1] {{\[}}1] : memref<2xindex> to memref<1xindex, strided<{{\[}}1], offset: 1>>
// CHECK-NEXT:      %[[ALLOC_9:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.copy %[[ALLOC_8]], %[[ALLOC_9]] : memref<2xindex> to memref<2xindex>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_8]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      %[[CAST_0:.*]] = memref.cast %[[ALLOC_10]] : memref<3xindex> to memref<?xindex>
// CHECK-NEXT:      %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_10]]{{\[}}0] {{\[}}2] {{\[}}1] : memref<3xindex> to memref<2xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      memref.copy %[[ALLOC_9]], %[[SUBVIEW_1]] : memref<2xindex> to memref<2xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_9]] : memref<2xindex>
// CHECK-NEXT:      %[[SUBVIEW_2:.*]] = memref.subview %[[ALLOC_10]]{{\[}}2] {{\[}}1] {{\[}}1] : memref<3xindex> to memref<1xindex, strided<{{\[}}1], offset: 2>>
// CHECK-NEXT:      memref.copy %[[SUBVIEW_0]], %[[SUBVIEW_2]] : memref<1xindex, strided<{{\[}}1], offset: 1>> to memref<1xindex, strided<{{\[}}1], offset: 2>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_7]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_5:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_6:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_7:.*]] = memref.load %[[ALLOC_10]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_8:.*]] = memref.load %[[ALLOC_10]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      %[[MULI_1:.*]] = arith.muli %[[LOAD_5]], %[[LOAD_6]] : index
// CHECK-NEXT:      %[[ADDI_0:.*]] = arith.addi %[[MULI_1]], %[[CONSTANT_4]] : index
// CHECK-NEXT:      %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_2:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_3:.*]] = arith.muli %[[LOAD_7]], %[[CONSTANT_3]] : index
// CHECK-NEXT:      %[[MULI_4:.*]] = arith.muli %[[MULI_3]], %[[LOAD_8]] : index
// CHECK-NEXT:      %[[ADDI_1:.*]] = arith.addi %[[MULI_4]], %[[CONSTANT_4]] : index
// CHECK-NEXT:      %[[DIVUI_1:.*]] = arith.divui %[[ADDI_1]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_5:.*]] = arith.muli %[[DIVUI_1]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[ADDI_2:.*]] = arith.addi %[[MULI_2]], %[[MULI_5]] : index
// CHECK-NEXT:      %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[ARG0]], %[[ADDI_2]]) {bufferization.manual_deallocation, domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_12]]]{{\[}}%[[LOAD_5]], %[[LOAD_6]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_1:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[MULI_2]]]{{\[}}%[[LOAD_7]], %[[LOAD_8]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_11:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[COLLAPSE_SHAPE_0:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}{{\[}}0, 1]] : memref<?x4096xf16, #hipsr.mem<device>> into memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.equal(%[[ARG0]]) ins(%[[ARG1]], %[[CONSTANT_9]] : memref<?x?xi64, #hipsr.mem<device>>, memref<i64, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:      %[[EXPAND_SHAPE_0:.*]] = memref.expand_shape %[[VIEW_0]] {{\[}}{{\[}}0], {{\[}}1, 2]] output_shape {{\[}}%[[LOAD_5]], %[[LOAD_6]], 1] : memref<?x?xi1, #hipsr.mem<device>> into memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.gather(%[[ARG0:[^:,]*]]) ins(%[[CONSTANT_10:.*]], %[[ARG1:[^:,]*]] : memref<248320x4096xf16, #hipsr.mem<device>>, memref<?x?xi64, #hipsr.mem<device>>) outs(%[[VIEW_1:.*]] : memref<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64}
// CHECK-NEXT:      %[[INDEX_CAST_0:.*]] = arith.index_cast %[[LOAD_7]] : index to i64
// CHECK-NEXT:      memref.store %[[INDEX_CAST_0]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_12]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_1:.*]] = arith.index_cast %[[LOAD_8]] : index to i64
// CHECK-NEXT:      memref.store %[[INDEX_CAST_1]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_11]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[CONSTANT_2]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_7]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_2]], %[[COLLAPSE_SHAPE_0]] : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_2]] : memref<1xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_5]], %[[VIEW_0]] : memref<2xindex>, memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_5]] : memref<2xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_6]], %[[EXPAND_SHAPE_0]] : memref<3xindex>, memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_6]] : memref<3xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_0]], %[[VIEW_1]] : memref<?xindex>, memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_10]] : memref<3xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_0]], %[[ALLOC_11]] : memref<1xindex>, memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_0]] : memref<1xindex>
// CHECK-NEXT:      %[[MEMREF_DIM_3:.*]] = memref.dim %[[EXPAND_SHAPE_0]], %[[CONSTANT_12]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MEMREF_DIM_4:.*]] = memref.dim %[[EXPAND_SHAPE_0]], %[[CONSTANT_11]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_12:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[MEMREF_DIM_3]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[MEMREF_DIM_4]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_11]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_9:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_12]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_2:.*]] = arith.index_cast %[[LOAD_9]] : i64 to index
// CHECK-NEXT:      %[[LOAD_10:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_11]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_3:.*]] = arith.index_cast %[[LOAD_10]] : i64 to index
// CHECK-NEXT:      %[[LOAD_11:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_7]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_4:.*]] = arith.index_cast %[[LOAD_11]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_13:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_2]], %[[ALLOC_13]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_3]], %[[ALLOC_13]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_4]], %[[ALLOC_13]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_14:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      scf.for %arg3 = %[[CONSTANT_12]] to %[[CONSTANT_13]] step %[[CONSTANT_11]] {
// CHECK-NEXT:      %[[LOAD_0:.*]] = arith.cmpi ult, %arg3, %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[MULI_0:.*]] = scf.if %[[LOAD_0]] -> (index) {
// CHECK-NEXT:      scf.yield %[[CONSTANT_11]] : index
// CHECK-NEXT:      } else {
// CHECK-NEXT:      %[[CMPI_0:.*]] = memref.load %[[ALLOC_12]]{{\[}}%arg3] : memref<3xindex>
// CHECK-NEXT:      %[[IF_0:.*]] = memref.load %[[ALLOC_13]]{{\[}}%arg3] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_1:.*]] = arith.cmpi eq, %[[IF_0]], %[[CONSTANT_11]] : index
// CHECK-NEXT:      %[[LOAD_2:.*]] = arith.select %[[LOAD_1]], %[[CMPI_0]], %[[IF_0]] : index
// CHECK-NEXT:      scf.yield %[[LOAD_2]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %[[MULI_0]], %[[ALLOC_14]]{{\[}}%arg3] : memref<3xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[ALLOC_13]] : memref<3xindex>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_12]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_12:.*]] = memref.load %[[ALLOC_14]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_13:.*]] = memref.load %[[ALLOC_14]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_14:.*]] = memref.load %[[ALLOC_14]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[MULI_6:.*]] = arith.muli %[[LOAD_12]], %[[LOAD_13]] : index
// CHECK-NEXT:      %[[MULI_7:.*]] = arith.muli %[[MULI_6]], %[[LOAD_14]] : index
// CHECK-NEXT:      %[[ADDI_3:.*]] = arith.addi %[[MULI_7]], %[[CONSTANT_4]] : index
// CHECK-NEXT:      %[[DIVUI_2:.*]] = arith.divui %[[ADDI_3]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_8:.*]] = arith.muli %[[DIVUI_2]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[GET_POOL_1:.*]] = hipsr.get_pool(%[[ARG0]], %[[MULI_8]]) {bufferization.manual_deallocation, domain_id = 1 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_2:.*]] = memref.view %[[GET_POOL_1]]{{\[}}%[[CONSTANT_12]]]{{\[}}%[[LOAD_12]], %[[LOAD_13]], %[[LOAD_14]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.expand(%[[ARG0]]) ins(%[[EXPAND_SHAPE_0]], %[[ALLOC_11]] : memref<?x?x1xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) outs(%[[VIEW_2]] : memref<?x?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_14]], %[[VIEW_2]] : memref<3xindex>, memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_14]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_15:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_11]], %[[ALLOC_15]]{{\[}}%[[CONSTANT_12]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_16:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_12]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_13]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_14]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_15:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_12]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_5:.*]] = arith.index_cast %[[LOAD_15]] : i64 to index
// CHECK-NEXT:      %[[LOAD_16:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_11]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_6:.*]] = arith.index_cast %[[LOAD_16]] : i64 to index
// CHECK-NEXT:      %[[LOAD_17:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_7]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_7:.*]] = arith.index_cast %[[LOAD_17]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_17:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_5]], %[[ALLOC_17]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_6]], %[[ALLOC_17]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_7]], %[[ALLOC_17]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_18:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      scf.for %arg3 = %[[CONSTANT_12]] to %[[CONSTANT_13]] step %[[CONSTANT_11]] {
// CHECK-NEXT:      %[[LOAD_0:.*]] = arith.cmpi ult, %arg3, %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[MULI_0:.*]] = scf.if %[[LOAD_0]] -> (index) {
// CHECK-NEXT:      scf.yield %[[CONSTANT_11]] : index
// CHECK-NEXT:      } else {
// CHECK-NEXT:      %[[CMPI_0:.*]] = memref.load %[[ALLOC_16]]{{\[}}%arg3] : memref<3xindex>
// CHECK-NEXT:      %[[IF_0:.*]] = memref.load %[[ALLOC_17]]{{\[}}%arg3] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_1:.*]] = arith.cmpi eq, %[[IF_0]], %[[CONSTANT_11]] : index
// CHECK-NEXT:      %[[LOAD_2:.*]] = arith.select %[[LOAD_1]], %[[CMPI_0]], %[[IF_0]] : index
// CHECK-NEXT:      scf.yield %[[LOAD_2]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %[[MULI_0]], %[[ALLOC_18]]{{\[}}%arg3] : memref<3xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[ALLOC_17]] : memref<3xindex>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_16]] : memref<3xindex>
// CHECK-NEXT:      %[[FOR_1:.*]] = scf.for %arg3 = %[[CONSTANT_12]] to %[[CONSTANT_13]] step %[[CONSTANT_11]] iter_args(%arg4 = %[[CONSTANT_11]]) -> (index) {
// CHECK-NEXT:      %[[LOAD_0:.*]] = memref.load %[[ALLOC_18]]{{\[}}%arg3] : memref<3xindex>
// CHECK-NEXT:      %[[MULI_0:.*]] = arith.muli %[[LOAD_0]], %arg4 : index
// CHECK-NEXT:      scf.yield %[[MULI_0]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[ALLOC_19:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_13]], %[[ALLOC_19]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[FOR_1]], %[[ALLOC_19]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_18:.*]] = memref.load %[[ALLOC_18]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_19:.*]] = memref.load %[[ALLOC_18]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_20:.*]] = memref.load %[[ALLOC_18]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_21:.*]] = memref.load %[[ALLOC_19]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[MULI_9:.*]] = arith.muli %[[LOAD_18]], %[[LOAD_19]] : index
// CHECK-NEXT:      %[[MULI_10:.*]] = arith.muli %[[MULI_9]], %[[LOAD_20]] : index
// CHECK-NEXT:      %[[ADDI_4:.*]] = arith.addi %[[MULI_10]], %[[CONSTANT_4]] : index
// CHECK-NEXT:      %[[DIVUI_3:.*]] = arith.divui %[[ADDI_4]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_11:.*]] = arith.muli %[[DIVUI_3]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_12:.*]] = arith.muli %[[LOAD_21]], %[[CONSTANT_1]] : index
// CHECK-NEXT:      %[[ADDI_5:.*]] = arith.addi %[[MULI_12]], %[[CONSTANT_4]] : index
// CHECK-NEXT:      %[[DIVUI_4:.*]] = arith.divui %[[ADDI_5]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_13:.*]] = arith.muli %[[DIVUI_4]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[ADDI_6:.*]] = arith.addi %[[MULI_11]], %[[MULI_13]] : index
// CHECK-NEXT:      %[[ADDI_7:.*]] = arith.addi %[[ADDI_6]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[GET_POOL_2:.*]] = hipsr.get_pool(%[[ARG0]], %[[ADDI_7]]) {bufferization.manual_deallocation, domain_id = 2 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_3:.*]] = memref.view %[[GET_POOL_2]]{{\[}}%[[CONSTANT_12]]]{{\[}}%[[LOAD_18]], %[[LOAD_19]], %[[LOAD_20]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_4:.*]] = memref.view %[[GET_POOL_2]]{{\[}}%[[MULI_11]]]{{\[}}%[[LOAD_21]]] : memref<?xi8, #hipsr.mem<device>> to memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_5:.*]] = memref.view %[[GET_POOL_2]]{{\[}}%[[ADDI_6]]]{{\[}}] : memref<?xi8, #hipsr.mem<device>> to memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_20:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.expand(%[[ARG0]]) ins(%[[VIEW_2]], %[[ALLOC_11]] : memref<?x?x?xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) outs(%[[VIEW_3]] : memref<?x?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.dealloc %[[ALLOC_11]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.nonzero(%[[ARG0]]) ins(%[[VIEW_3]] : memref<?x?x?xi1, #hipsr.mem<device>>) outs(%[[VIEW_4]], %[[VIEW_5]] : memref<3x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.copy_d2h(%[[ARG0]]) ins(%[[VIEW_5]] : memref<1xi64, #hipsr.mem<device>>) outs(%[[ALLOC_20]] : memref<1xi64, #hipsr.mem<host>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_18]], %[[VIEW_3]] : memref<3xindex>, memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_18]] : memref<3xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_19]], %[[VIEW_4]] : memref<2xindex>, memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_19]] : memref<2xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_15]], %[[VIEW_5]] : memref<1xindex>, memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_15]], %[[ALLOC_20]] : memref<1xindex>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_15]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_21:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_11]], %[[ALLOC_21]]{{\[}}%[[CONSTANT_12]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_22:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_7]], %[[ALLOC_22]]{{\[}}%[[CONSTANT_12]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_23:.*]] = memref.alloc() {alignment = 64 : i64} : memref<0xindex>
// CHECK-NEXT:      %[[LOAD_22:.*]] = memref.load %[[ALLOC_20]]{{\[}}%[[CONSTANT_12]]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_20]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_8:.*]] = arith.index_cast %[[LOAD_22]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_24:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_13]], %[[ALLOC_24]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_8]], %[[ALLOC_24]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_23:.*]] = memref.load %[[ALLOC_24]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_24:.*]] = memref.load %[[ALLOC_24]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_25:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[LOAD_23]], %[[ALLOC_25]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[LOAD_24]], %[[ALLOC_25]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_25:.*]] = memref.load %[[ALLOC_24]]{{\[}}%[[CONSTANT_11]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_26:.*]] = memref.load %[[ALLOC_25]]{{\[}}%[[CONSTANT_12]]] : memref<2xindex>
// CHECK-NEXT:      %[[MULI_14:.*]] = arith.muli %[[LOAD_25]], %[[CONSTANT_1]] : index
// CHECK-NEXT:      %[[MULI_15:.*]] = arith.muli %[[LOAD_26]], %[[CONSTANT_1]] : index
// CHECK-NEXT:      %[[MAXUI_0:.*]] = arith.maxui %[[MULI_14]], %[[MULI_15]] : index
// CHECK-NEXT:      %[[ADDI_8:.*]] = arith.addi %[[MAXUI_0]], %[[CONSTANT_4]] : index
// CHECK-NEXT:      %[[DIVUI_5:.*]] = arith.divui %[[ADDI_8]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_16:.*]] = arith.muli %[[DIVUI_5]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[GET_POOL_3:.*]] = hipsr.get_pool(%[[ARG0]], %[[MULI_16]]) {bufferization.manual_deallocation, domain_id = 3 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_6:.*]] = memref.view %[[GET_POOL_3]]{{\[}}%[[CONSTANT_12]]]{{\[}}%[[LOAD_26]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_26:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[ALLOC_27:.*]] = memref.alloc() {alignment = 64 : i64} : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[SUBVIEW_3:.*]] = memref.subview %[[VIEW_4]]{{\[}}0, 0] {{\[}}3, %[[LOAD_25]]] {{\[}}1, 1] : memref<3x?xi64, #hipsr.mem<device>> to memref<3x?xi64, strided<{{\[}}?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.transpose(%[[ARG0:[^:,]*]]) ins(%[[SUBVIEW_3:.*]] : memref<3x?xi64, strided<{{\[}}?, 1]>, #hipsr.mem<device>>) outs(%[[VIEW_6:.*]] : memref<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>}
// CHECK-NEXT:      %[[INDEX_CAST_9:.*]] = arith.index_cast %[[LOAD_26]] : index to i64
// CHECK-NEXT:      memref.store %[[INDEX_CAST_9]], %[[ALLOC_26]]{{\[}}%[[CONSTANT_12]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[CONSTANT_0]], %[[ALLOC_26]]{{\[}}%[[CONSTANT_11]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[LOAD_27:.*]] = memref.load %[[ALLOC_26]]{{\[}}%[[CONSTANT_12]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[LOAD_27]], %[[ALLOC_27]]{{\[}}] : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[EXPAND_SHAPE_1:.*]] = memref.expand_shape %[[ALLOC_27]] {{\[}}] output_shape {{\[}}1] : memref<i64, #hipsr.mem<host>> into memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_24]], %[[SUBVIEW_3]] : memref<2xindex>, memref<3x?xi64, strided<{{\[}}?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_24]] : memref<2xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_25]], %[[VIEW_6]] : memref<2xindex>, memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_25]] : memref<2xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_22]], %[[ALLOC_26]] : memref<1xindex>, memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_26]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_22]] : memref<1xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_23]], %[[ALLOC_27]] : memref<0xindex>, memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_23]] : memref<0xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_21]], %[[EXPAND_SHAPE_1]] : memref<1xindex>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_21]] : memref<1xindex>
// CHECK-NEXT:      %[[MEMREF_DIM_5:.*]] = memref.dim %[[COLLAPSE_SHAPE_0]], %[[CONSTANT_12]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[LOAD_28:.*]] = memref.load %[[EXPAND_SHAPE_1]]{{\[}}%[[CONSTANT_12]]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_10:.*]] = arith.index_cast %[[LOAD_28]] : i64 to index
// CHECK-NEXT:      %[[CMPI_2:.*]] = arith.cmpi slt, %[[INDEX_CAST_10]], %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[ADDI_9:.*]] = arith.addi %[[INDEX_CAST_10]], %[[MEMREF_DIM_5]] : index
// CHECK-NEXT:      %[[ARITH_SELECT_1:.*]] = arith.select %[[CMPI_2]], %[[ADDI_9]], %[[INDEX_CAST_10]] : index
// CHECK-NEXT:      %[[MINSI_0:.*]] = arith.minsi %[[MEMREF_DIM_5]], %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[MAXSI_0:.*]] = arith.maxsi %[[ARITH_SELECT_1]], %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[MINSI_1:.*]] = arith.minsi %[[MAXSI_0]], %[[MEMREF_DIM_5]] : index
// CHECK-NEXT:      %[[SUBI_0:.*]] = arith.subi %[[MINSI_1]], %[[MINSI_0]] : index
// CHECK-NEXT:      %[[MAXSI_1:.*]] = arith.maxsi %[[SUBI_0]], %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[ALLOC_28:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[MAXSI_1]], %[[ALLOC_28]]{{\[}}%[[CONSTANT_12]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_29:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_7]], %[[ALLOC_29]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_8]], %[[ALLOC_29]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_8]], %[[ALLOC_29]]{{\[}}%[[CONSTANT_7]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_29:.*]] = memref.load %[[ALLOC_28]]{{\[}}%[[CONSTANT_12]]] : memref<1xindex>
// CHECK-NEXT:      %[[MULI_17:.*]] = arith.muli %[[LOAD_29]], %[[CONSTANT_7]] : index
// CHECK-NEXT:      %[[ADDI_10:.*]] = arith.addi %[[MULI_17]], %[[CONSTANT_4]] : index
// CHECK-NEXT:      %[[DIVUI_6:.*]] = arith.divui %[[ADDI_10]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[MULI_18:.*]] = arith.muli %[[DIVUI_6]], %[[CONSTANT_5]] : index
// CHECK-NEXT:      %[[GET_POOL_4:.*]] = hipsr.get_pool(%[[ARG0]], %[[MULI_18]]) {bufferization.manual_deallocation, domain_id = 4 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_7:.*]] = memref.view %[[GET_POOL_4]]{{\[}}%[[CONSTANT_12]]]{{\[}}%[[LOAD_29]]] : memref<?xi8, #hipsr.mem<device>> to memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[LOAD_30:.*]] = memref.load %[[ALLOC_29]]{{\[}}%[[CONSTANT_12]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_31:.*]] = memref.load %[[ALLOC_29]]{{\[}}%[[CONSTANT_11]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_OUTPUT_0:.*]] = hipsr.alloc_output(%[[ARG0]], %[[LOAD_30]], %[[LOAD_31]]) {out_idx = 0 : i64} : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.slice(%[[ARG0:[^:,]*]]) ins(%[[COLLAPSE_SHAPE_0:.*]] : memref<?xf16, #hipsr.mem<device>>) ends(%[[EXPAND_SHAPE_1:.*]] : memref<1xi64, #hipsr.mem<host>>) outs(%[[VIEW_7:.*]] : memref<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>}
// CHECK-NEXT:      memref.dealloc %[[ALLOC_27]] : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.scatter_nd(%[[ARG0]]) ins(%[[VIEW_1]], %[[VIEW_6]], %[[VIEW_7]] : memref<?x?x4096xf16, #hipsr.mem<device>>, memref<?x3xi64, #hipsr.mem<device>>, memref<?xf16, #hipsr.mem<device>>) outs(%[[ALLOC_OUTPUT_0]] : memref<?x?x4096xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_28]], %[[VIEW_7]] : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_28]] : memref<1xindex>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_29]], %[[ALLOC_OUTPUT_0]] : memref<3xindex>, memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[ALLOC_29]] : memref<3xindex>
// CHECK-NEXT:      return %[[ALLOC_OUTPUT_0]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      }

module {
  func.func @main_graph(%arg0: tensor<?x?xi64> {onnx.name = "input_ids"}, %arg1: tensor<?x4096xf16> {onnx.name = "image_features"}) -> (tensor<?x?x4096xf16> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {node.outputs = ["embed_tokens.weight"], location = "embedding.onnx.data", offset = 0 : i64, size = 2034237440 : i64} : () -> tensor<248320x4096xf16>
    %2 = "onnx.Constant"() {node.outputs = ["/Constant_output_0"], value = dense<248056> : tensor<i64>} : () -> tensor<i64>
    %3 = "onnx.Constant"() {node.outputs = ["/Constant_1_output_0"], value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %4 = "onnx.Constant"() {node.outputs = ["/Constant_3_output_0"], value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %5 = "onnx.Constant"() {node.outputs = ["/Constant_4_output_0"], value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %6 = "onnx.Reshape"(%arg1, %3) {allowzero = 0 : si64, node.outputs = ["/Reshape_output_0"], onnx_node_name = "/Reshape"} : (tensor<?x4096xf16>, tensor<1xi64>) -> tensor<?xf16>
    %7 = "onnx.Equal"(%arg0, %2) {node.outputs = ["/Equal_output_0"], onnx_node_name = "/Equal"} : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi1>
    %8 = "onnx.Unsqueeze"(%7, %3) {node.outputs = ["/Unsqueeze_output_0"], onnx_node_name = "/Unsqueeze"} : (tensor<?x?xi1>, tensor<1xi64>) -> tensor<?x?x1xi1>
    %9 = "onnx.Gather"(%1, %arg0) {axis = 0 : si64, node.outputs = ["/embed_tokens/Gather_output_0"], onnx_node_name = "/embed_tokens/Gather"} : (tensor<248320x4096xf16>, tensor<?x?xi64>) -> tensor<?x?x4096xf16>
    %10 = "onnx.Shape"(%9) {node.outputs = ["/Shape_1_output_0"], onnx_node_name = "/Shape_1", start = 0 : si64} : (tensor<?x?x4096xf16>) -> tensor<3xi64>
    %11 = "onnx.Expand"(%8, %10) {node.outputs = ["/Expand_output_0"], onnx_node_name = "/Expand"} : (tensor<?x?x1xi1>, tensor<3xi64>) -> tensor<?x?x?xi1>
    %12 = "onnx.Expand"(%11, %10) {node.outputs = ["/Expand_1_output_0"], onnx_node_name = "/Expand_1"} : (tensor<?x?x?xi1>, tensor<3xi64>) -> tensor<?x?x?xi1>
    %13 = "onnx.NonZero"(%12) {node.outputs = ["/NonZero_output_0"], onnx_node_name = "/NonZero"} : (tensor<?x?x?xi1>) -> tensor<3x?xi64>
    %14 = "onnx.Transpose"(%13) {node.outputs = ["/Transpose_output_0"], onnx_node_name = "/Transpose", perm = [1, 0]} : (tensor<3x?xi64>) -> tensor<?x3xi64>
    %15 = "onnx.Shape"(%14) {node.outputs = ["/Shape_2_output_0"], onnx_node_name = "/Shape_2", start = 0 : si64} : (tensor<?x3xi64>) -> tensor<2xi64>
    %16 = "onnx.Gather"(%15, %4) {axis = 0 : si64, node.outputs = ["/Gather_output_0"], onnx_node_name = "/Gather"} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    %17 = "onnx.Unsqueeze"(%16, %5) {node.outputs = ["/Unsqueeze_1_output_0"], onnx_node_name = "/Unsqueeze_1"} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %18 = "onnx.Slice"(%6, %5, %17, %5, %0) {node.outputs = ["/Slice_output_0"], onnx_node_name = "/Slice"} : (tensor<?xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, none) -> tensor<?xf16>
    %19 = "onnx.ScatterND"(%9, %14, %18) {node.outputs = ["inputs_embeds"], onnx_node_name = "/ScatterND", reduction = "none"} : (tensor<?x?x4096xf16>, tensor<?x3xi64>, tensor<?xf16>) -> tensor<?x?x4096xf16>
    "onnx.Return"(%19) : (tensor<?x?x4096xf16>) -> ()
  }
}
