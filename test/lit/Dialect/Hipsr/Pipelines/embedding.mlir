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
// CHECK-SAME:      %[[ARG2:[^:,]*]]: memref<?x4096xf16, #hipsr.mem<device>> {onnx.name = "image_features"}) -> (memref<?x?x4096xf16,
// CHECK-SAME:      #hipsr.mem<device>> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:      %[[CONSTANT_0:.*]] = arith.constant 248320 : index
// CHECK-NEXT:      %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_2:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[CONSTANT_3:.*]] = hipsr.constant {value = dense<248056> : tensor<i64>} : memref<i64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_4:.*]] = hipsr.constant {value = dense_resource<"file|embedding.onnx.data|0"> : tensor<248320x4096xf16, #hipsr.mem<device>>} : memref<248320x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[CONSTANT_7:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[ALLOC_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_7]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_6]]] : memref<1xindex>
// CHECK-NEXT:      %[[DIM_0:.*]] = memref.dim %[[ARG2]], %[[CONSTANT_6]] : memref<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_2]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:      %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_1]] step %[[CONSTANT_5]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_5]]) -> (index) {
// CHECK-NEXT:        %[[LOAD_0:.*]] = memref.load %[[ALLOC_1]]{{\[}}%[[VAL_0]]] : memref<2xindex>
// CHECK-NEXT:        %[[MULI_0:.*]] = arith.muli %[[LOAD_0]], %[[VAL_1]] : index
// CHECK-NEXT:        scf.yield %[[MULI_0]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[FOR_0]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_6]]] : memref<1xindex>
// CHECK-NEXT:      %[[DIM_1:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_6]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_2:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_5]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_1]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_2]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<0xindex>
// CHECK-NEXT:      %[[ALLOC_5:.*]] = memref.alloc(%[[CONSTANT_1]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      scf.for %[[VAL_0:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_1]] step %[[CONSTANT_5]] {
// CHECK-NEXT:        %[[LOAD_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_6]] : index
// CHECK-NEXT:        %[[MULI_0:.*]] = scf.if %[[LOAD_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[CONSTANT_5]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_1:.*]] = memref.load %[[ALLOC_3]]{{\[}}%[[VAL_0]]] : memref<2xindex>
// CHECK-NEXT:          scf.yield %[[LOAD_1]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:        %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[MULI_0]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_1:.*]] = arith.subi %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:          %[[LOAD_2:.*]] = memref.load %[[ALLOC_4]]{{\[}}%[[LOAD_1]]] : memref<0xindex>
// CHECK-NEXT:          %[[CMPI_1:.*]] = arith.cmpi eq, %[[LOAD_2]], %[[CONSTANT_5]] : index
// CHECK-NEXT:          %[[SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[MULI_0]], %[[LOAD_2]] : index
// CHECK-NEXT:          scf.yield %[[SELECT_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %[[IF_0]], %[[ALLOC_5]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[CAST_0:.*]] = memref.cast %[[ALLOC_5]] : memref<?xindex> to memref<2xindex>
// CHECK-NEXT:      %[[LOAD_3:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_4:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK-NEXT:      %[[ALLOC_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_3]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_6]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[LOAD_4]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_5]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_5]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_1]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_0]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_2]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:      %[[DIM_3:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_6]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_4:.*]] = memref.dim %[[ARG1]], %[[CONSTANT_5]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_3]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[DIM_4]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:      %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_7]]{{\[}}%[[CONSTANT_6]]] {{\[}}%[[CONSTANT_6]]] {{\[}}%[[CONSTANT_5]]] : memref<2xindex> to memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:      %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_7]]{{\[}}%[[CONSTANT_5]]] {{\[}}%[[CONSTANT_5]]] {{\[}}%[[CONSTANT_5]]] : memref<2xindex> to memref<?xindex, strided<{{\[}}?], offset: ?>>
// CHECK-NEXT:      %[[ALLOC_9:.*]] = memref.alloc(%[[CONSTANT_1]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      %[[SUBVIEW_2:.*]] = memref.subview %[[ALLOC_9]]{{\[}}0] {{\[}}%[[CONSTANT_6]]] {{\[}}1] : memref<?xindex> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      memref.copy %[[SUBVIEW_0]], %[[SUBVIEW_2]] : memref<?xindex, strided<{{\[}}?], offset: ?>> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      %[[SUBVIEW_3:.*]] = memref.subview %[[ALLOC_9]]{{\[}}%[[CONSTANT_6]]] {{\[}}2] {{\[}}1] : memref<?xindex> to memref<2xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      memref.copy %[[ALLOC_8]], %[[SUBVIEW_3]] : memref<2xindex> to memref<2xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      %[[ALLOC_10:.*]] = memref.alloc(%[[CONSTANT_7]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      %[[SUBVIEW_4:.*]] = memref.subview %[[ALLOC_10]]{{\[}}0] {{\[}}%[[CONSTANT_1]]] {{\[}}1] : memref<?xindex> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      memref.copy %[[ALLOC_9]], %[[SUBVIEW_4]] : memref<?xindex> to memref<?xindex, strided<{{\[}}1]>>
// CHECK-NEXT:      %[[SUBVIEW_5:.*]] = memref.subview %[[ALLOC_10]]{{\[}}%[[CONSTANT_1]]] {{\[}}%[[CONSTANT_5]]] {{\[}}1] : memref<?xindex> to memref<?xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      memref.copy %[[SUBVIEW_1]], %[[SUBVIEW_5]] : memref<?xindex, strided<{{\[}}?], offset: ?>> to memref<?xindex, strided<{{\[}}1], offset: ?>>
// CHECK-NEXT:      %[[LOAD_5:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_6:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_7:.*]] = memref.load %[[ALLOC_10]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_8:.*]] = memref.load %[[ALLOC_10]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK-NEXT:      %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[MULI_1:.*]] = arith.muli %[[CONSTANT_8]], %[[LOAD_5]] : index
// CHECK-NEXT:      %[[MULI_2:.*]] = arith.muli %[[MULI_1]], %[[LOAD_6]] : index
// CHECK-NEXT:      %[[CONSTANT_9:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_10:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_0:.*]] = arith.addi %[[MULI_2]], %[[CONSTANT_10]] : index
// CHECK-NEXT:      %[[DIVUI_0:.*]] = arith.divui %[[ADDI_0]], %[[CONSTANT_9]] : index
// CHECK-NEXT:      %[[MULI_3:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_9]] : index
// CHECK-NEXT:      %[[CONSTANT_11:.*]] = arith.constant 8192 : index
// CHECK-NEXT:      %[[MULI_4:.*]] = arith.muli %[[CONSTANT_11]], %[[LOAD_7]] : index
// CHECK-NEXT:      %[[MULI_5:.*]] = arith.muli %[[MULI_4]], %[[LOAD_8]] : index
// CHECK-NEXT:      %[[CONSTANT_12:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_13:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_1:.*]] = arith.addi %[[MULI_5]], %[[CONSTANT_13]] : index
// CHECK-NEXT:      %[[DIVUI_1:.*]] = arith.divui %[[ADDI_1]], %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[MULI_6:.*]] = arith.muli %[[DIVUI_1]], %[[CONSTANT_12]] : index
// CHECK-NEXT:      %[[CONSTANT_14:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ADDI_2:.*]] = arith.addi %[[MULI_3]], %[[MULI_6]] : index
// CHECK-NEXT:      %[[GET_POOL_0:.*]] = hipsr.get_pool(%[[ARG0]], %[[ADDI_2]]) {domain_id = 0 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_0:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[CONSTANT_14]]]{{\[}}%[[LOAD_5]], %[[LOAD_6]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_1:.*]] = memref.view %[[GET_POOL_0]]{{\[}}%[[MULI_3]]]{{\[}}%[[LOAD_7]], %[[LOAD_8]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_11:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[COLLAPSE_SHAPE_0:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}{{\[}}0, 1]] : memref<?x4096xf16, #hipsr.mem<device>> into memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.equal(%[[ARG0]]) ins(%[[ARG1]], %[[CONSTANT_3]] : memref<?x?xi64, #hipsr.mem<device>>, memref<i64, #hipsr.mem<device>>) outs(%[[VIEW_0]] : memref<?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:      %[[CONSTANT_15:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_16:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM_5:.*]] = memref.dim %[[VIEW_0]], %[[CONSTANT_16]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_6:.*]] = memref.dim %[[VIEW_0]], %[[CONSTANT_15]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXPAND_SHAPE_0:.*]] = memref.expand_shape %[[VIEW_0]] {{\[}}{{\[}}0], {{\[}}1, 2]] output_shape {{\[}}%[[DIM_5]], %[[DIM_6]], 1] : memref<?x?xi1, #hipsr.mem<device>> into memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.gather(%[[ARG0]]) ins(%[[CONSTANT_4]], %[[ARG1]] : memref<248320x4096xf16, #hipsr.mem<device>>, memref<?x?xi64, #hipsr.mem<device>>) outs(%[[VIEW_1]] : memref<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64}
// CHECK-NEXT:      %[[CONSTANT_17:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_18:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:      %[[CONSTANT_19:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_20:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM_7:.*]] = memref.dim %[[VIEW_1]], %[[CONSTANT_20]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIM_7]] : index to i64
// CHECK-NEXT:      memref.store %[[INDEX_CAST_0]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_20]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[DIM_8:.*]] = memref.dim %[[VIEW_1]], %[[CONSTANT_19]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[INDEX_CAST_1:.*]] = arith.index_cast %[[DIM_8]] : index to i64
// CHECK-NEXT:      memref.store %[[INDEX_CAST_1]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_19]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[CONSTANT_18]], %[[ALLOC_11]]{{\[}}%[[CONSTANT_17]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_2]], %[[COLLAPSE_SHAPE_0]] : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_0]], %[[VIEW_0]] : memref<2xindex>, memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_6]], %[[EXPAND_SHAPE_0]] : memref<3xindex>, memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_10]], %[[VIEW_1]] : memref<?xindex>, memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_0]], %[[ALLOC_11]] : memref<1xindex>, memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[CONSTANT_21:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[CONSTANT_22:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_23:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_24:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM_9:.*]] = memref.dim %[[EXPAND_SHAPE_0]], %[[CONSTANT_24]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_10:.*]] = memref.dim %[[EXPAND_SHAPE_0]], %[[CONSTANT_23]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_12:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[DIM_9]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_24]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[DIM_10]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_23]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_23]], %[[ALLOC_12]]{{\[}}%[[CONSTANT_22]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_9:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_24]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_2:.*]] = arith.index_cast %[[LOAD_9]] : i64 to index
// CHECK-NEXT:      %[[LOAD_10:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_23]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_3:.*]] = arith.index_cast %[[LOAD_10]] : i64 to index
// CHECK-NEXT:      %[[LOAD_11:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_22]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_4:.*]] = arith.index_cast %[[LOAD_11]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_13:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_2]], %[[ALLOC_13]]{{\[}}%[[CONSTANT_24]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_3]], %[[ALLOC_13]]{{\[}}%[[CONSTANT_23]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_4]], %[[ALLOC_13]]{{\[}}%[[CONSTANT_22]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_14:.*]] = memref.alloc(%[[CONSTANT_21]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      scf.for %[[VAL_0:.*]] = %[[CONSTANT_24]] to %[[CONSTANT_21]] step %[[CONSTANT_23]] {
// CHECK-NEXT:        %[[LOAD_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_24]] : index
// CHECK-NEXT:        %[[MULI_0:.*]] = scf.if %[[LOAD_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[CONSTANT_23]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_1:.*]] = memref.load %[[ALLOC_12]]{{\[}}%[[VAL_0]]] : memref<3xindex>
// CHECK-NEXT:          scf.yield %[[LOAD_1]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_24]] : index
// CHECK-NEXT:        %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[MULI_0]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_1:.*]] = memref.load %[[ALLOC_13]]{{\[}}%[[VAL_0]]] : memref<3xindex>
// CHECK-NEXT:          %[[LOAD_2:.*]] = arith.cmpi eq, %[[LOAD_1]], %[[CONSTANT_23]] : index
// CHECK-NEXT:          %[[CMPI_1:.*]] = arith.select %[[LOAD_2]], %[[MULI_0]], %[[LOAD_1]] : index
// CHECK-NEXT:          scf.yield %[[CMPI_1]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %[[IF_0]], %[[ALLOC_14]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[CAST_1:.*]] = memref.cast %[[ALLOC_14]] : memref<?xindex> to memref<3xindex>
// CHECK-NEXT:      %[[LOAD_12:.*]] = memref.load %[[ALLOC_14]]{{\[}}%[[CONSTANT_24]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_13:.*]] = memref.load %[[ALLOC_14]]{{\[}}%[[CONSTANT_23]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_14:.*]] = memref.load %[[ALLOC_14]]{{\[}}%[[CONSTANT_22]]] : memref<?xindex>
// CHECK-NEXT:      %[[CONSTANT_25:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[MULI_7:.*]] = arith.muli %[[CONSTANT_25]], %[[LOAD_12]] : index
// CHECK-NEXT:      %[[MULI_8:.*]] = arith.muli %[[MULI_7]], %[[LOAD_13]] : index
// CHECK-NEXT:      %[[MULI_9:.*]] = arith.muli %[[MULI_8]], %[[LOAD_14]] : index
// CHECK-NEXT:      %[[CONSTANT_26:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_27:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_3:.*]] = arith.addi %[[MULI_9]], %[[CONSTANT_27]] : index
// CHECK-NEXT:      %[[DIVUI_2:.*]] = arith.divui %[[ADDI_3]], %[[CONSTANT_26]] : index
// CHECK-NEXT:      %[[MULI_10:.*]] = arith.muli %[[DIVUI_2]], %[[CONSTANT_26]] : index
// CHECK-NEXT:      %[[CONSTANT_28:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[GET_POOL_1:.*]] = hipsr.get_pool(%[[ARG0]], %[[MULI_10]]) {domain_id = 1 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_2:.*]] = memref.view %[[GET_POOL_1]]{{\[}}%[[CONSTANT_28]]]{{\[}}%[[LOAD_12]], %[[LOAD_13]], %[[LOAD_14]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.expand(%[[ARG0]]) ins(%[[EXPAND_SHAPE_0]], %[[ALLOC_11]] : memref<?x?x1xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) outs(%[[VIEW_2]] : memref<?x?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_1]], %[[VIEW_2]] : memref<3xindex>, memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_29:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[CONSTANT_30:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_31:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_32:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ALLOC_15:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_31]], %[[ALLOC_15]]{{\[}}%[[CONSTANT_32]]] : memref<1xindex>
// CHECK-NEXT:      %[[DIM_11:.*]] = memref.dim %[[VIEW_2]], %[[CONSTANT_32]] : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_12:.*]] = memref.dim %[[VIEW_2]], %[[CONSTANT_31]] : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_13:.*]] = memref.dim %[[VIEW_2]], %[[CONSTANT_30]] : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_16:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[DIM_11]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_32]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[DIM_12]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_31]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[DIM_13]], %[[ALLOC_16]]{{\[}}%[[CONSTANT_30]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_15:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_32]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_5:.*]] = arith.index_cast %[[LOAD_15]] : i64 to index
// CHECK-NEXT:      %[[LOAD_16:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_31]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_6:.*]] = arith.index_cast %[[LOAD_16]] : i64 to index
// CHECK-NEXT:      %[[LOAD_17:.*]] = memref.load %[[ALLOC_11]]{{\[}}%[[CONSTANT_30]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_7:.*]] = arith.index_cast %[[LOAD_17]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_17:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_5]], %[[ALLOC_17]]{{\[}}%[[CONSTANT_32]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_6]], %[[ALLOC_17]]{{\[}}%[[CONSTANT_31]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_7]], %[[ALLOC_17]]{{\[}}%[[CONSTANT_30]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_18:.*]] = memref.alloc(%[[CONSTANT_29]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:      scf.for %[[VAL_0:.*]] = %[[CONSTANT_32]] to %[[CONSTANT_29]] step %[[CONSTANT_31]] {
// CHECK-NEXT:        %[[LOAD_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_32]] : index
// CHECK-NEXT:        %[[MULI_0:.*]] = scf.if %[[LOAD_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[CONSTANT_31]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_1:.*]] = memref.load %[[ALLOC_16]]{{\[}}%[[VAL_0]]] : memref<3xindex>
// CHECK-NEXT:          scf.yield %[[LOAD_1]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[CONSTANT_32]] : index
// CHECK-NEXT:        %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:          scf.yield %[[MULI_0]] : index
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[LOAD_1:.*]] = memref.load %[[ALLOC_17]]{{\[}}%[[VAL_0]]] : memref<3xindex>
// CHECK-NEXT:          %[[LOAD_2:.*]] = arith.cmpi eq, %[[LOAD_1]], %[[CONSTANT_31]] : index
// CHECK-NEXT:          %[[CMPI_1:.*]] = arith.select %[[LOAD_2]], %[[MULI_0]], %[[LOAD_1]] : index
// CHECK-NEXT:          scf.yield %[[CMPI_1]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %[[IF_0]], %[[ALLOC_18]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[CAST_2:.*]] = memref.cast %[[ALLOC_18]] : memref<?xindex> to memref<3xindex>
// CHECK-NEXT:      %[[FOR_1:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_32]] to %[[CONSTANT_29]] step %[[CONSTANT_31]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_31]]) -> (index) {
// CHECK-NEXT:        %[[LOAD_0:.*]] = memref.load %[[ALLOC_18]]{{\[}}%[[VAL_0]]] : memref<?xindex>
// CHECK-NEXT:        %[[MULI_0:.*]] = arith.muli %[[LOAD_0]], %[[VAL_1]] : index
// CHECK-NEXT:        scf.yield %[[MULI_0]] : index
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[ALLOC_19:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_29]], %[[ALLOC_19]]{{\[}}%[[CONSTANT_32]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[FOR_1]], %[[ALLOC_19]]{{\[}}%[[CONSTANT_31]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_18:.*]] = memref.load %[[ALLOC_18]]{{\[}}%[[CONSTANT_32]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_19:.*]] = memref.load %[[ALLOC_18]]{{\[}}%[[CONSTANT_31]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_20:.*]] = memref.load %[[ALLOC_18]]{{\[}}%[[CONSTANT_30]]] : memref<?xindex>
// CHECK-NEXT:      %[[LOAD_21:.*]] = memref.load %[[ALLOC_19]]{{\[}}%[[CONSTANT_31]]] : memref<2xindex>
// CHECK-NEXT:      %[[CONSTANT_33:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[MULI_11:.*]] = arith.muli %[[CONSTANT_33]], %[[LOAD_18]] : index
// CHECK-NEXT:      %[[MULI_12:.*]] = arith.muli %[[MULI_11]], %[[LOAD_19]] : index
// CHECK-NEXT:      %[[MULI_13:.*]] = arith.muli %[[MULI_12]], %[[LOAD_20]] : index
// CHECK-NEXT:      %[[CONSTANT_34:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_35:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_4:.*]] = arith.addi %[[MULI_13]], %[[CONSTANT_35]] : index
// CHECK-NEXT:      %[[DIVUI_3:.*]] = arith.divui %[[ADDI_4]], %[[CONSTANT_34]] : index
// CHECK-NEXT:      %[[MULI_14:.*]] = arith.muli %[[DIVUI_3]], %[[CONSTANT_34]] : index
// CHECK-NEXT:      %[[CONSTANT_36:.*]] = arith.constant 24 : index
// CHECK-NEXT:      %[[MULI_15:.*]] = arith.muli %[[CONSTANT_36]], %[[LOAD_21]] : index
// CHECK-NEXT:      %[[CONSTANT_37:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_38:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_5:.*]] = arith.addi %[[MULI_15]], %[[CONSTANT_38]] : index
// CHECK-NEXT:      %[[DIVUI_4:.*]] = arith.divui %[[ADDI_5]], %[[CONSTANT_37]] : index
// CHECK-NEXT:      %[[MULI_16:.*]] = arith.muli %[[DIVUI_4]], %[[CONSTANT_37]] : index
// CHECK-NEXT:      %[[CONSTANT_39:.*]] = arith.constant 8 : index
// CHECK-NEXT:      %[[CONSTANT_40:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_41:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_6:.*]] = arith.addi %[[CONSTANT_39]], %[[CONSTANT_41]] : index
// CHECK-NEXT:      %[[DIVUI_5:.*]] = arith.divui %[[ADDI_6]], %[[CONSTANT_40]] : index
// CHECK-NEXT:      %[[MULI_17:.*]] = arith.muli %[[DIVUI_5]], %[[CONSTANT_40]] : index
// CHECK-NEXT:      %[[CONSTANT_42:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ADDI_7:.*]] = arith.addi %[[MULI_14]], %[[MULI_16]] : index
// CHECK-NEXT:      %[[ADDI_8:.*]] = arith.addi %[[ADDI_7]], %[[MULI_17]] : index
// CHECK-NEXT:      %[[GET_POOL_2:.*]] = hipsr.get_pool(%[[ARG0]], %[[ADDI_8]]) {domain_id = 2 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_3:.*]] = memref.view %[[GET_POOL_2]]{{\[}}%[[CONSTANT_42]]]{{\[}}%[[LOAD_18]], %[[LOAD_19]], %[[LOAD_20]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_4:.*]] = memref.view %[[GET_POOL_2]]{{\[}}%[[MULI_14]]]{{\[}}%[[LOAD_21]]] : memref<?xi8, #hipsr.mem<device>> to memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_5:.*]] = memref.view %[[GET_POOL_2]]{{\[}}%[[ADDI_7]]]{{\[}}] : memref<?xi8, #hipsr.mem<device>> to memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_20:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.expand(%[[ARG0]]) ins(%[[VIEW_2]], %[[ALLOC_11]] : memref<?x?x?xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) outs(%[[VIEW_3]] : memref<?x?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.nonzero(%[[ARG0]]) ins(%[[VIEW_3]] : memref<?x?x?xi1, #hipsr.mem<device>>) outs(%[[VIEW_4]], %[[VIEW_5]] : memref<3x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.copy_d2h(%[[ARG0]]) ins(%[[VIEW_5]] : memref<1xi64, #hipsr.mem<device>>) outs(%[[ALLOC_20]] : memref<1xi64, #hipsr.mem<host>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[CAST_2]], %[[VIEW_3]] : memref<3xindex>, memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_19]], %[[VIEW_4]] : memref<2xindex>, memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_15]], %[[VIEW_5]] : memref<1xindex>, memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_15]], %[[ALLOC_20]] : memref<1xindex>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[CONSTANT_43:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[CONSTANT_44:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_45:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[CONSTANT_46:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[ALLOC_21:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_46]], %[[ALLOC_21]]{{\[}}%[[CONSTANT_45]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_22:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_44]], %[[ALLOC_22]]{{\[}}%[[CONSTANT_45]]] : memref<1xindex>
// CHECK-NEXT:      %[[ALLOC_23:.*]] = memref.alloc() {alignment = 64 : i64} : memref<0xindex>
// CHECK-NEXT:      %[[LOAD_22:.*]] = memref.load %[[ALLOC_20]]{{\[}}%[[CONSTANT_45]]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_8:.*]] = arith.index_cast %[[LOAD_22]] : i64 to index
// CHECK-NEXT:      %[[ALLOC_24:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_43]], %[[ALLOC_24]]{{\[}}%[[CONSTANT_45]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[INDEX_CAST_8]], %[[ALLOC_24]]{{\[}}%[[CONSTANT_46]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_23:.*]] = memref.load %[[ALLOC_24]]{{\[}}%[[CONSTANT_46]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_24:.*]] = memref.load %[[ALLOC_24]]{{\[}}%[[CONSTANT_45]]] : memref<2xindex>
// CHECK-NEXT:      %[[ALLOC_25:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:      memref.store %[[LOAD_23]], %[[ALLOC_25]]{{\[}}%[[CONSTANT_45]]] : memref<2xindex>
// CHECK-NEXT:      memref.store %[[LOAD_24]], %[[ALLOC_25]]{{\[}}%[[CONSTANT_46]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_25:.*]] = memref.load %[[ALLOC_24]]{{\[}}%[[CONSTANT_46]]] : memref<2xindex>
// CHECK-NEXT:      %[[LOAD_26:.*]] = memref.load %[[ALLOC_25]]{{\[}}%[[CONSTANT_45]]] : memref<2xindex>
// CHECK-NEXT:      %[[CONSTANT_47:.*]] = arith.constant 24 : index
// CHECK-NEXT:      %[[MULI_18:.*]] = arith.muli %[[CONSTANT_47]], %[[LOAD_25]] : index
// CHECK-NEXT:      %[[CONSTANT_48:.*]] = arith.constant 24 : index
// CHECK-NEXT:      %[[MULI_19:.*]] = arith.muli %[[CONSTANT_48]], %[[LOAD_26]] : index
// CHECK-NEXT:      %[[MAXUI_0:.*]] = arith.maxui %[[MULI_18]], %[[MULI_19]] : index
// CHECK-NEXT:      %[[CONSTANT_49:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_50:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_9:.*]] = arith.addi %[[MAXUI_0]], %[[CONSTANT_50]] : index
// CHECK-NEXT:      %[[DIVUI_6:.*]] = arith.divui %[[ADDI_9]], %[[CONSTANT_49]] : index
// CHECK-NEXT:      %[[MULI_20:.*]] = arith.muli %[[DIVUI_6]], %[[CONSTANT_49]] : index
// CHECK-NEXT:      %[[CONSTANT_51:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[GET_POOL_3:.*]] = hipsr.get_pool(%[[ARG0]], %[[MULI_20]]) {domain_id = 3 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_6:.*]] = memref.view %[[GET_POOL_3]]{{\[}}%[[CONSTANT_51]]]{{\[}}%[[LOAD_25]]] : memref<?xi8, #hipsr.mem<device>> to memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_7:.*]] = memref.view %[[GET_POOL_3]]{{\[}}%[[CONSTANT_51]]]{{\[}}%[[LOAD_26]]] : memref<?xi8, #hipsr.mem<device>> to memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_26:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[ALLOC_27:.*]] = memref.alloc() {alignment = 64 : i64} : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[CONSTANT_52:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM_14:.*]] = memref.dim %[[VIEW_6]], %[[CONSTANT_52]] : memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[SUBVIEW_6:.*]] = memref.subview %[[VIEW_4]]{{\[}}0, 0] {{\[}}3, %[[DIM_14]]] {{\[}}1, 1] : memref<3x?xi64, #hipsr.mem<device>> to memref<3x?xi64, strided<{{\[}}?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.transpose(%[[ARG0]]) ins(%[[SUBVIEW_6]] : memref<3x?xi64, strided<{{\[}}?, 1]>, #hipsr.mem<device>>) outs(%[[VIEW_7]] : memref<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>}
// CHECK-NEXT:      %[[CONSTANT_53:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_54:.*]] = arith.constant 3 : i64
// CHECK-NEXT:      %[[CONSTANT_55:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM_15:.*]] = memref.dim %[[VIEW_7]], %[[CONSTANT_55]] : memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[INDEX_CAST_9:.*]] = arith.index_cast %[[DIM_15]] : index to i64
// CHECK-NEXT:      memref.store %[[INDEX_CAST_9]], %[[ALLOC_26]]{{\[}}%[[CONSTANT_55]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[CONSTANT_54]], %[[ALLOC_26]]{{\[}}%[[CONSTANT_53]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[CONSTANT_56:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LOAD_27:.*]] = memref.load %[[ALLOC_26]]{{\[}}%[[CONSTANT_56]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      memref.store %[[LOAD_27]], %[[ALLOC_27]]{{\[}}] : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[EXPAND_SHAPE_1:.*]] = memref.expand_shape %[[ALLOC_27]] {{\[}}] output_shape {{\[}}1] : memref<i64, #hipsr.mem<host>> into memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_24]], %[[SUBVIEW_6]] : memref<2xindex>, memref<3x?xi64, strided<{{\[}}?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_25]], %[[VIEW_7]] : memref<2xindex>, memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_22]], %[[ALLOC_26]] : memref<1xindex>, memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_23]], %[[ALLOC_27]] : memref<0xindex>, memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_21]], %[[EXPAND_SHAPE_1]] : memref<1xindex>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[CONSTANT_57:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[CONSTANT_58:.*]] = arith.constant 4096 : index
// CHECK-NEXT:      %[[CONSTANT_59:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[CONSTANT_60:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM_16:.*]] = memref.dim %[[COLLAPSE_SHAPE_0]], %[[CONSTANT_60]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[LOAD_28:.*]] = memref.load %[[EXPAND_SHAPE_1]]{{\[}}%[[CONSTANT_60]]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[INDEX_CAST_10:.*]] = arith.index_cast %[[LOAD_28]] : i64 to index
// CHECK-NEXT:      %[[CMPI_2:.*]] = arith.cmpi slt, %[[INDEX_CAST_10]], %[[CONSTANT_60]] : index
// CHECK-NEXT:      %[[ADDI_10:.*]] = arith.addi %[[INDEX_CAST_10]], %[[DIM_16]] : index
// CHECK-NEXT:      %[[SELECT_1:.*]] = arith.select %[[CMPI_2]], %[[ADDI_10]], %[[INDEX_CAST_10]] : index
// CHECK-NEXT:      %[[MINSI_0:.*]] = arith.minsi %[[DIM_16]], %[[CONSTANT_60]] : index
// CHECK-NEXT:      %[[MAXSI_0:.*]] = arith.maxsi %[[SELECT_1]], %[[CONSTANT_60]] : index
// CHECK-NEXT:      %[[MINSI_1:.*]] = arith.minsi %[[MAXSI_0]], %[[DIM_16]] : index
// CHECK-NEXT:      %[[SUBI_0:.*]] = arith.subi %[[MINSI_1]], %[[MINSI_0]] : index
// CHECK-NEXT:      %[[MAXSI_1:.*]] = arith.maxsi %[[SUBI_0]], %[[CONSTANT_60]] : index
// CHECK-NEXT:      %[[ALLOC_28:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:      memref.store %[[MAXSI_1]], %[[ALLOC_28]]{{\[}}%[[CONSTANT_60]]] : memref<1xindex>
// CHECK-NEXT:      %[[DIM_17:.*]] = memref.dim %[[VIEW_1]], %[[CONSTANT_60]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_18:.*]] = memref.dim %[[VIEW_1]], %[[CONSTANT_59]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[ALLOC_29:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:      memref.store %[[DIM_17]], %[[ALLOC_29]]{{\[}}%[[CONSTANT_60]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[DIM_18]], %[[ALLOC_29]]{{\[}}%[[CONSTANT_59]]] : memref<3xindex>
// CHECK-NEXT:      memref.store %[[CONSTANT_58]], %[[ALLOC_29]]{{\[}}%[[CONSTANT_57]]] : memref<3xindex>
// CHECK-NEXT:      %[[LOAD_29:.*]] = memref.load %[[ALLOC_28]]{{\[}}%[[CONSTANT_60]]] : memref<1xindex>
// CHECK-NEXT:      %[[DIM_19:.*]] = memref.dim %[[VIEW_1]], %[[CONSTANT_60]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[DIM_20:.*]] = memref.dim %[[VIEW_1]], %[[CONSTANT_59]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_61:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[MULI_21:.*]] = arith.muli %[[CONSTANT_61]], %[[LOAD_29]] : index
// CHECK-NEXT:      %[[CONSTANT_62:.*]] = arith.constant 256 : index
// CHECK-NEXT:      %[[CONSTANT_63:.*]] = arith.constant 255 : index
// CHECK-NEXT:      %[[ADDI_11:.*]] = arith.addi %[[MULI_21]], %[[CONSTANT_63]] : index
// CHECK-NEXT:      %[[DIVUI_7:.*]] = arith.divui %[[ADDI_11]], %[[CONSTANT_62]] : index
// CHECK-NEXT:      %[[MULI_22:.*]] = arith.muli %[[DIVUI_7]], %[[CONSTANT_62]] : index
// CHECK-NEXT:      %[[CONSTANT_64:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[GET_POOL_4:.*]] = hipsr.get_pool(%[[ARG0]], %[[MULI_22]]) {domain_id = 4 : i64} : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VIEW_8:.*]] = memref.view %[[GET_POOL_4]]{{\[}}%[[CONSTANT_64]]]{{\[}}%[[LOAD_29]]] : memref<?xi8, #hipsr.mem<device>> to memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CONSTANT_65:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LOAD_30:.*]] = memref.load %[[ALLOC_29]]{{\[}}%[[CONSTANT_65]]] : memref<3xindex>
// CHECK-NEXT:      %[[CONSTANT_66:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[LOAD_31:.*]] = memref.load %[[ALLOC_29]]{{\[}}%[[CONSTANT_66]]] : memref<3xindex>
// CHECK-NEXT:      %[[ALLOC_OUTPUT_0:.*]] = hipsr.alloc_output(%[[ARG0]], %[[LOAD_30]], %[[LOAD_31]]) {out_idx = 0 : i64} : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.slice(%[[ARG0]]) ins(%[[COLLAPSE_SHAPE_0]] : memref<?xf16, #hipsr.mem<device>>) ends(%[[EXPAND_SHAPE_1]] : memref<1xi64, #hipsr.mem<host>>) outs(%[[VIEW_8]] : memref<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>}
// CHECK-NEXT:      hipsr.scatter_nd(%[[ARG0]]) ins(%[[VIEW_1]], %[[VIEW_7]], %[[VIEW_8]] : memref<?x?x4096xf16, #hipsr.mem<device>>, memref<?x3xi64, #hipsr.mem<device>>, memref<?xf16, #hipsr.mem<device>>) outs(%[[ALLOC_OUTPUT_0]] : memref<?x?x4096xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_28]], %[[VIEW_8]] : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[ALLOC_29]], %[[ALLOC_OUTPUT_0]] : memref<3xindex>, memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      return %[[ALLOC_OUTPUT_0]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }


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
