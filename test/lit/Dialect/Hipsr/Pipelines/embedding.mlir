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
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: memref<?x?xi64, #hipsr.mem<device>> {onnx.name = "input_ids"},
// CHECK-SAME:      %[[ARG2:.*]]: memref<?x4096xf16, #hipsr.mem<device>> {onnx.name = "image_features"}) -> (memref<?x?x4096xf16, #hipsr.mem<device>> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]]:5 = hipsr.pool_domain(%[[ARG0]], %[[ARG2]], %[[ARG1]] : !hipsr.context, memref<?x4096xf16, #hipsr.mem<device>>, memref<?x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: memref<?x4096xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: memref<?x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[CONSTANT_0:.*]] = arith.constant 248320 : index
// CHECK-NEXT:        %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[CONSTANT_2:.*]] = arith.constant 4096 : index
// CHECK-NEXT:        %[[CONSTANT_3:.*]] = hipsr.constant {value = dense<248056> : tensor<i64>} : memref<i64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CONSTANT_4:.*]] = hipsr.constant {value = dense_resource<"file|embedding.onnx.data|0"> : tensor<248320x4096xf16, #hipsr.mem<device>>} : memref<248320x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[CONSTANT_7:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[ALLOC_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_7]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_6]]] : memref<1xindex>
// CHECK-NEXT:        %[[DIM_0:.*]] = memref.dim %[[VAL_1]], %[[CONSTANT_6]] : memref<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_0]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_2]], %[[ALLOC_1]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:        %[[FOR_0:.*]] = scf.for %[[VAL_3:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_1]] step %[[CONSTANT_5]] iter_args(%[[VAL_4:.*]] = %[[CONSTANT_5]]) -> (index) {
// CHECK-NEXT:          %[[LOAD_0:.*]] = memref.load %[[ALLOC_1]]{{\[}}%[[VAL_3]]] : memref<2xindex>
// CHECK-NEXT:          %[[MULI_0:.*]] = arith.muli %[[LOAD_0]], %[[VAL_4]] : index
// CHECK-NEXT:          scf.yield %[[MULI_0]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:        memref.store %[[FOR_0]], %[[ALLOC_2]]{{\[}}%[[CONSTANT_6]]] : memref<1xindex>
// CHECK-NEXT:        %[[DIM_1:.*]] = memref.dim %[[VAL_2]], %[[CONSTANT_6]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_2:.*]] = memref.dim %[[VAL_2]], %[[CONSTANT_5]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_1]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_2]], %[[ALLOC_3]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<0xindex>
// CHECK-NEXT:        %[[ALLOC_5:.*]] = memref.alloc(%[[CONSTANT_1]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        scf.for %[[VAL_5:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_1]] step %[[CONSTANT_5]] {
// CHECK-NEXT:          %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_5]], %[[CONSTANT_6]] : index
// CHECK-NEXT:          %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:            scf.yield %[[CONSTANT_5]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_1:.*]] = memref.load %[[ALLOC_3]]{{\[}}%[[VAL_5]]] : memref<2xindex>
// CHECK-NEXT:            scf.yield %[[LOAD_1]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_5]], %[[CONSTANT_1]] : index
// CHECK-NEXT:          %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (index) {
// CHECK-NEXT:            scf.yield %[[IF_0]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[SUBI_0:.*]] = arith.subi %[[VAL_5]], %[[CONSTANT_1]] : index
// CHECK-NEXT:            %[[LOAD_2:.*]] = memref.load %[[ALLOC_4]]{{\[}}%[[SUBI_0]]] : memref<0xindex>
// CHECK-NEXT:            %[[CMPI_2:.*]] = arith.cmpi eq, %[[LOAD_2]], %[[CONSTANT_5]] : index
// CHECK-NEXT:            %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[IF_0]], %[[LOAD_2]] : index
// CHECK-NEXT:            scf.yield %[[SELECT_0]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.store %[[IF_1]], %[[ALLOC_5]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CAST_0:.*]] = memref.cast %[[ALLOC_5]] : memref<?xindex> to memref<2xindex>
// CHECK-NEXT:        %[[LOAD_3:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK-NEXT:        %[[LOAD_4:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:        memref.store %[[LOAD_3]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_6]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[LOAD_4]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_5]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_5]], %[[ALLOC_6]]{{\[}}%[[CONSTANT_1]]] : memref<3xindex>
// CHECK-NEXT:        %[[ALLOC_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_0]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_2]], %[[ALLOC_7]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:        %[[DIM_3:.*]] = memref.dim %[[VAL_2]], %[[CONSTANT_6]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_4:.*]] = memref.dim %[[VAL_2]], %[[CONSTANT_5]] : memref<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_3]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_6]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[DIM_4]], %[[ALLOC_8]]{{\[}}%[[CONSTANT_5]]] : memref<2xindex>
// CHECK-NEXT:        %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_7]]{{\[}}%[[CONSTANT_6]]] [%[[CONSTANT_6]]] [%[[CONSTANT_5]]] : memref<2xindex> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:        %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_7]]{{\[}}%[[CONSTANT_5]]] [%[[CONSTANT_5]]] [%[[CONSTANT_5]]] : memref<2xindex> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-NEXT:        %[[ALLOC_9:.*]] = memref.alloc(%[[CONSTANT_1]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        %[[SUBVIEW_2:.*]] = memref.subview %[[ALLOC_9]]{{\[}}0] [%[[CONSTANT_6]]] [1] : memref<?xindex> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        memref.copy %[[SUBVIEW_0]], %[[SUBVIEW_2]] : memref<?xindex, strided<[?], offset: ?>> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        %[[SUBVIEW_3:.*]] = memref.subview %[[ALLOC_9]]{{\[}}%[[CONSTANT_6]]] [2] [1] : memref<?xindex> to memref<2xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        memref.copy %[[ALLOC_8]], %[[SUBVIEW_3]] : memref<2xindex> to memref<2xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[ALLOC_10:.*]] = memref.alloc(%[[CONSTANT_7]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        %[[SUBVIEW_4:.*]] = memref.subview %[[ALLOC_10]]{{\[}}0] [%[[CONSTANT_1]]] [1] : memref<?xindex> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        memref.copy %[[ALLOC_9]], %[[SUBVIEW_4]] : memref<?xindex> to memref<?xindex, strided<[1]>>
// CHECK-NEXT:        %[[SUBVIEW_5:.*]] = memref.subview %[[ALLOC_10]]{{\[}}%[[CONSTANT_1]]] [%[[CONSTANT_5]]] [1] : memref<?xindex> to memref<?xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        memref.copy %[[SUBVIEW_1]], %[[SUBVIEW_5]] : memref<?xindex, strided<[?], offset: ?>> to memref<?xindex, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[LOAD_5:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK-NEXT:        %[[LOAD_6:.*]] = memref.load %[[ALLOC_5]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_11:.*]] = memref.alloc(%[[LOAD_5]], %[[LOAD_6]]) {alignment = 64 : i64} : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        %[[LOAD_7:.*]] = memref.load %[[ALLOC_10]]{{\[}}%[[CONSTANT_6]]] : memref<?xindex>
// CHECK-NEXT:        %[[LOAD_8:.*]] = memref.load %[[ALLOC_10]]{{\[}}%[[CONSTANT_5]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_12:.*]] = memref.alloc(%[[LOAD_7]], %[[LOAD_8]]) {alignment = 64 : i64} : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_13:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[COMPUTE_0:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[VAL_1]] : memref<?x4096xf16, #hipsr.mem<device>>) outs(%[[VAL_1]] : memref<?x4096xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_6:.*]]: !hipsr.context, %[[VAL_7:.*]]: memref<?x4096xf16, #hipsr.mem<device>>, %[[VAL_8:.*]]: memref<?x4096xf16, #hipsr.mem<device>>):
// CHECK-NEXT:          %[[COLLAPSE_SHAPE_0:.*]] = memref.collapse_shape %[[VAL_7]] {{\[\[}}0, 1]] : memref<?x4096xf16, #hipsr.mem<device>> into memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:          hipsr.compute_yield %[[COLLAPSE_SHAPE_0]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:        } : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.equal(%[[VAL_0]]) ins(%[[VAL_2]], %[[CONSTANT_3]] : memref<?x?xi64, #hipsr.mem<device>>, memref<i64, #hipsr.mem<device>>) outs(%[[ALLOC_11]] : memref<?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:        %[[COMPUTE_1:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[ALLOC_11]] : memref<?x?xi1, #hipsr.mem<device>>) outs(%[[ALLOC_11]] : memref<?x?xi1, #hipsr.mem<device>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_9:.*]]: !hipsr.context, %[[VAL_10:.*]]: memref<?x?xi1, #hipsr.mem<device>>, %[[VAL_11:.*]]: memref<?x?xi1, #hipsr.mem<device>>):
// CHECK-NEXT:          %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[DIM_5:.*]] = memref.dim %[[VAL_10]], %[[CONSTANT_9]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:          %[[DIM_6:.*]] = memref.dim %[[VAL_10]], %[[CONSTANT_8]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:          %[[EXPAND_SHAPE_0:.*]] = memref.expand_shape %[[VAL_10]] {{\[\[}}0], [1, 2]] output_shape [%[[DIM_5]], %[[DIM_6]], 1] : memref<?x?xi1, #hipsr.mem<device>> into memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:          hipsr.compute_yield %[[EXPAND_SHAPE_0]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:        } : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.gather(%[[VAL_0]]) ins(%[[CONSTANT_4]], %[[VAL_2]] : memref<248320x4096xf16, #hipsr.mem<device>>, memref<?x?xi64, #hipsr.mem<device>>) outs(%[[ALLOC_12]] : memref<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64}
// CHECK-NEXT:        %[[COMPUTE_2:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[ALLOC_12]] : memref<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[ALLOC_13]] : memref<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_12:.*]]: !hipsr.context, %[[VAL_13:.*]]: memref<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_14:.*]]: memref<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:          %[[CONSTANT_10:.*]] = arith.constant 2 : index
// CHECK-NEXT:          %[[CONSTANT_11:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:          %[[CONSTANT_12:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[CONSTANT_13:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[DIM_7:.*]] = memref.dim %[[VAL_13]], %[[CONSTANT_13]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:          %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIM_7]] : index to i64
// CHECK-NEXT:          memref.store %[[INDEX_CAST_0]], %[[VAL_14]]{{\[}}%[[CONSTANT_13]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:          %[[DIM_8:.*]] = memref.dim %[[VAL_13]], %[[CONSTANT_12]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:          %[[INDEX_CAST_1:.*]] = arith.index_cast %[[DIM_8]] : index to i64
// CHECK-NEXT:          memref.store %[[INDEX_CAST_1]], %[[VAL_14]]{{\[}}%[[CONSTANT_12]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:          memref.store %[[CONSTANT_11]], %[[VAL_14]]{{\[}}%[[CONSTANT_10]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:          hipsr.compute_yield %[[VAL_14]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        } : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_2]], %[[COMPUTE_0]] : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[CAST_0]], %[[ALLOC_11]] : memref<2xindex>, memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_6]], %[[COMPUTE_1]] : memref<3xindex>, memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_10]], %[[ALLOC_12]] : memref<?xindex>, memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_0]], %[[COMPUTE_2]] : memref<1xindex>, memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[COMPUTE_0]], %[[COMPUTE_1]], %[[ALLOC_12]], %[[ALLOC_12]], %[[COMPUTE_2]] : memref<?xf16, #hipsr.mem<device>>, memref<?x?x1xi1, #hipsr.mem<device>>, memref<?x?x4096xf16, #hipsr.mem<device>>, memref<?x?x4096xf16, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } -> memref<?xf16, #hipsr.mem<device>>, memref<?x?x1xi1, #hipsr.mem<device>>, memref<?x?x4096xf16, #hipsr.mem<device>>, memref<?x?x4096xf16, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>> {domain_id = 0 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_1:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]]#1, %[[POOL_DOMAIN_0]]#4 : !hipsr.context, memref<?x?x1xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_15:.*]]: !hipsr.context, %[[VAL_16:.*]]: memref<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_17:.*]]: memref<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[CONSTANT_14:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[CONSTANT_15:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[CONSTANT_16:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[CONSTANT_17:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[DIM_9:.*]] = memref.dim %[[VAL_16]], %[[CONSTANT_17]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_10:.*]] = memref.dim %[[VAL_16]], %[[CONSTANT_16]] : memref<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_14:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:        memref.store %[[DIM_9]], %[[ALLOC_14]]{{\[}}%[[CONSTANT_17]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[DIM_10]], %[[ALLOC_14]]{{\[}}%[[CONSTANT_16]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_16]], %[[ALLOC_14]]{{\[}}%[[CONSTANT_15]]] : memref<3xindex>
// CHECK-NEXT:        %[[LOAD_9:.*]] = memref.load %[[VAL_17]]{{\[}}%[[CONSTANT_17]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_2:.*]] = arith.index_cast %[[LOAD_9]] : i64 to index
// CHECK-NEXT:        %[[LOAD_10:.*]] = memref.load %[[VAL_17]]{{\[}}%[[CONSTANT_16]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_3:.*]] = arith.index_cast %[[LOAD_10]] : i64 to index
// CHECK-NEXT:        %[[LOAD_11:.*]] = memref.load %[[VAL_17]]{{\[}}%[[CONSTANT_15]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_4:.*]] = arith.index_cast %[[LOAD_11]] : i64 to index
// CHECK-NEXT:        %[[ALLOC_15:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_2]], %[[ALLOC_15]]{{\[}}%[[CONSTANT_17]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_3]], %[[ALLOC_15]]{{\[}}%[[CONSTANT_16]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_4]], %[[ALLOC_15]]{{\[}}%[[CONSTANT_15]]] : memref<3xindex>
// CHECK-NEXT:        %[[ALLOC_16:.*]] = memref.alloc(%[[CONSTANT_14]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        scf.for %[[VAL_18:.*]] = %[[CONSTANT_17]] to %[[CONSTANT_14]] step %[[CONSTANT_16]] {
// CHECK-NEXT:          %[[CMPI_3:.*]] = arith.cmpi ult, %[[VAL_18]], %[[CONSTANT_17]] : index
// CHECK-NEXT:          %[[IF_2:.*]] = scf.if %[[CMPI_3]] -> (index) {
// CHECK-NEXT:            scf.yield %[[CONSTANT_16]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_12:.*]] = memref.load %[[ALLOC_14]]{{\[}}%[[VAL_18]]] : memref<3xindex>
// CHECK-NEXT:            scf.yield %[[LOAD_12]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[CMPI_4:.*]] = arith.cmpi ult, %[[VAL_18]], %[[CONSTANT_17]] : index
// CHECK-NEXT:          %[[IF_3:.*]] = scf.if %[[CMPI_4]] -> (index) {
// CHECK-NEXT:            scf.yield %[[IF_2]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_13:.*]] = memref.load %[[ALLOC_15]]{{\[}}%[[VAL_18]]] : memref<3xindex>
// CHECK-NEXT:            %[[CMPI_5:.*]] = arith.cmpi eq, %[[LOAD_13]], %[[CONSTANT_16]] : index
// CHECK-NEXT:            %[[SELECT_1:.*]] = arith.select %[[CMPI_5]], %[[IF_2]], %[[LOAD_13]] : index
// CHECK-NEXT:            scf.yield %[[SELECT_1]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.store %[[IF_3]], %[[ALLOC_16]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CAST_1:.*]] = memref.cast %[[ALLOC_16]] : memref<?xindex> to memref<3xindex>
// CHECK-NEXT:        %[[LOAD_14:.*]] = memref.load %[[ALLOC_16]]{{\[}}%[[CONSTANT_17]]] : memref<?xindex>
// CHECK-NEXT:        %[[LOAD_15:.*]] = memref.load %[[ALLOC_16]]{{\[}}%[[CONSTANT_16]]] : memref<?xindex>
// CHECK-NEXT:        %[[LOAD_16:.*]] = memref.load %[[ALLOC_16]]{{\[}}%[[CONSTANT_15]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_17:.*]] = memref.alloc(%[[LOAD_14]], %[[LOAD_15]], %[[LOAD_16]]) {alignment = 64 : i64} : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.expand(%[[VAL_15]]) ins(%[[VAL_16]], %[[VAL_17]] : memref<?x?x1xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) outs(%[[ALLOC_17]] : memref<?x?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:        hipsr.preserve_shape %[[CAST_1]], %[[ALLOC_17]] : memref<3xindex>, memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ALLOC_17]] : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:      } -> memref<?x?x?xi1, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_2:.*]]:2 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_1]], %[[POOL_DOMAIN_0]]#4 : !hipsr.context, memref<?x?x?xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_19:.*]]: !hipsr.context, %[[VAL_20:.*]]: memref<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_21:.*]]: memref<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[CONSTANT_18:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[CONSTANT_19:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[CONSTANT_20:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[CONSTANT_21:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[ALLOC_18:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_20]], %[[ALLOC_18]]{{\[}}%[[CONSTANT_21]]] : memref<1xindex>
// CHECK-NEXT:        %[[DIM_11:.*]] = memref.dim %[[VAL_20]], %[[CONSTANT_21]] : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_12:.*]] = memref.dim %[[VAL_20]], %[[CONSTANT_20]] : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_13:.*]] = memref.dim %[[VAL_20]], %[[CONSTANT_19]] : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_19:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:        memref.store %[[DIM_11]], %[[ALLOC_19]]{{\[}}%[[CONSTANT_21]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[DIM_12]], %[[ALLOC_19]]{{\[}}%[[CONSTANT_20]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[DIM_13]], %[[ALLOC_19]]{{\[}}%[[CONSTANT_19]]] : memref<3xindex>
// CHECK-NEXT:        %[[LOAD_17:.*]] = memref.load %[[VAL_21]]{{\[}}%[[CONSTANT_21]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_5:.*]] = arith.index_cast %[[LOAD_17]] : i64 to index
// CHECK-NEXT:        %[[LOAD_18:.*]] = memref.load %[[VAL_21]]{{\[}}%[[CONSTANT_20]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_6:.*]] = arith.index_cast %[[LOAD_18]] : i64 to index
// CHECK-NEXT:        %[[LOAD_19:.*]] = memref.load %[[VAL_21]]{{\[}}%[[CONSTANT_19]]] : memref<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_7:.*]] = arith.index_cast %[[LOAD_19]] : i64 to index
// CHECK-NEXT:        %[[ALLOC_20:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_5]], %[[ALLOC_20]]{{\[}}%[[CONSTANT_21]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_6]], %[[ALLOC_20]]{{\[}}%[[CONSTANT_20]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_7]], %[[ALLOC_20]]{{\[}}%[[CONSTANT_19]]] : memref<3xindex>
// CHECK-NEXT:        %[[ALLOC_21:.*]] = memref.alloc(%[[CONSTANT_18]]) {alignment = 64 : i64} : memref<?xindex>
// CHECK-NEXT:        scf.for %[[VAL_22:.*]] = %[[CONSTANT_21]] to %[[CONSTANT_18]] step %[[CONSTANT_20]] {
// CHECK-NEXT:          %[[CMPI_6:.*]] = arith.cmpi ult, %[[VAL_22]], %[[CONSTANT_21]] : index
// CHECK-NEXT:          %[[IF_4:.*]] = scf.if %[[CMPI_6]] -> (index) {
// CHECK-NEXT:            scf.yield %[[CONSTANT_20]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_20:.*]] = memref.load %[[ALLOC_19]]{{\[}}%[[VAL_22]]] : memref<3xindex>
// CHECK-NEXT:            scf.yield %[[LOAD_20]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[CMPI_7:.*]] = arith.cmpi ult, %[[VAL_22]], %[[CONSTANT_21]] : index
// CHECK-NEXT:          %[[IF_5:.*]] = scf.if %[[CMPI_7]] -> (index) {
// CHECK-NEXT:            scf.yield %[[IF_4]] : index
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %[[LOAD_21:.*]] = memref.load %[[ALLOC_20]]{{\[}}%[[VAL_22]]] : memref<3xindex>
// CHECK-NEXT:            %[[CMPI_8:.*]] = arith.cmpi eq, %[[LOAD_21]], %[[CONSTANT_20]] : index
// CHECK-NEXT:            %[[SELECT_2:.*]] = arith.select %[[CMPI_8]], %[[IF_4]], %[[LOAD_21]] : index
// CHECK-NEXT:            scf.yield %[[SELECT_2]] : index
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.store %[[IF_5]], %[[ALLOC_21]]{{\[}}%[[VAL_22]]] : memref<?xindex>
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[CAST_2:.*]] = memref.cast %[[ALLOC_21]] : memref<?xindex> to memref<3xindex>
// CHECK-NEXT:        %[[FOR_1:.*]] = scf.for %[[VAL_23:.*]] = %[[CONSTANT_21]] to %[[CONSTANT_18]] step %[[CONSTANT_20]] iter_args(%[[VAL_24:.*]] = %[[CONSTANT_20]]) -> (index) {
// CHECK-NEXT:          %[[LOAD_22:.*]] = memref.load %[[ALLOC_21]]{{\[}}%[[VAL_23]]] : memref<?xindex>
// CHECK-NEXT:          %[[MULI_1:.*]] = arith.muli %[[LOAD_22]], %[[VAL_24]] : index
// CHECK-NEXT:          scf.yield %[[MULI_1]] : index
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[ALLOC_22:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_18]], %[[ALLOC_22]]{{\[}}%[[CONSTANT_21]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[FOR_1]], %[[ALLOC_22]]{{\[}}%[[CONSTANT_20]]] : memref<2xindex>
// CHECK-NEXT:        %[[LOAD_23:.*]] = memref.load %[[ALLOC_21]]{{\[}}%[[CONSTANT_21]]] : memref<?xindex>
// CHECK-NEXT:        %[[LOAD_24:.*]] = memref.load %[[ALLOC_21]]{{\[}}%[[CONSTANT_20]]] : memref<?xindex>
// CHECK-NEXT:        %[[LOAD_25:.*]] = memref.load %[[ALLOC_21]]{{\[}}%[[CONSTANT_19]]] : memref<?xindex>
// CHECK-NEXT:        %[[ALLOC_23:.*]] = memref.alloc(%[[LOAD_23]], %[[LOAD_24]], %[[LOAD_25]]) {alignment = 64 : i64} : memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        %[[LOAD_26:.*]] = memref.load %[[ALLOC_22]]{{\[}}%[[CONSTANT_20]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_24:.*]] = memref.alloc(%[[LOAD_26]]) {alignment = 64 : i64} : memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_25:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_26:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.expand(%[[VAL_19]]) ins(%[[VAL_20]], %[[VAL_21]] : memref<?x?x?xi1, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<host>>) outs(%[[ALLOC_23]] : memref<?x?x?xi1, #hipsr.mem<device>>)
// CHECK-NEXT:        hipsr.nonzero(%[[VAL_19]]) ins(%[[ALLOC_23]] : memref<?x?x?xi1, #hipsr.mem<device>>) outs(%[[ALLOC_24]], %[[ALLOC_25]] : memref<3x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>)
// CHECK-NEXT:        hipsr.copy_d2h(%[[VAL_19]]) ins(%[[ALLOC_25]] : memref<1xi64, #hipsr.mem<device>>) outs(%[[ALLOC_26]] : memref<1xi64, #hipsr.mem<host>>)
// CHECK-NEXT:        hipsr.preserve_shape %[[CAST_2]], %[[ALLOC_23]] : memref<3xindex>, memref<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_22]], %[[ALLOC_24]] : memref<2xindex>, memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_18]], %[[ALLOC_25]] : memref<1xindex>, memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_18]], %[[ALLOC_26]] : memref<1xindex>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ALLOC_24]], %[[ALLOC_26]] : memref<3x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } -> memref<3x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<host>> {domain_id = 2 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_3:.*]]:3 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_2]]#1, %[[POOL_DOMAIN_2]]#0 : !hipsr.context, memref<1xi64, #hipsr.mem<host>>, memref<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_25:.*]]: !hipsr.context, %[[VAL_26:.*]]: memref<1xi64, #hipsr.mem<host>>, %[[VAL_27:.*]]: memref<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[CONSTANT_22:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[CONSTANT_23:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[CONSTANT_24:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[CONSTANT_25:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[ALLOC_27:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_25]], %[[ALLOC_27]]{{\[}}%[[CONSTANT_24]]] : memref<1xindex>
// CHECK-NEXT:        %[[ALLOC_28:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_23]], %[[ALLOC_28]]{{\[}}%[[CONSTANT_24]]] : memref<1xindex>
// CHECK-NEXT:        %[[ALLOC_29:.*]] = memref.alloc() {alignment = 64 : i64} : memref<0xindex>
// CHECK-NEXT:        %[[LOAD_27:.*]] = memref.load %[[VAL_26]]{{\[}}%[[CONSTANT_24]]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_8:.*]] = arith.index_cast %[[LOAD_27]] : i64 to index
// CHECK-NEXT:        %[[ALLOC_30:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_22]], %[[ALLOC_30]]{{\[}}%[[CONSTANT_24]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[INDEX_CAST_8]], %[[ALLOC_30]]{{\[}}%[[CONSTANT_25]]] : memref<2xindex>
// CHECK-NEXT:        %[[LOAD_28:.*]] = memref.load %[[ALLOC_30]]{{\[}}%[[CONSTANT_25]]] : memref<2xindex>
// CHECK-NEXT:        %[[LOAD_29:.*]] = memref.load %[[ALLOC_30]]{{\[}}%[[CONSTANT_24]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_31:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xindex>
// CHECK-NEXT:        memref.store %[[LOAD_28]], %[[ALLOC_31]]{{\[}}%[[CONSTANT_24]]] : memref<2xindex>
// CHECK-NEXT:        memref.store %[[LOAD_29]], %[[ALLOC_31]]{{\[}}%[[CONSTANT_25]]] : memref<2xindex>
// CHECK-NEXT:        %[[LOAD_30:.*]] = memref.load %[[ALLOC_30]]{{\[}}%[[CONSTANT_25]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_32:.*]] = memref.alloc(%[[LOAD_30]]) {alignment = 64 : i64} : memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[LOAD_31:.*]] = memref.load %[[ALLOC_31]]{{\[}}%[[CONSTANT_24]]] : memref<2xindex>
// CHECK-NEXT:        %[[ALLOC_33:.*]] = memref.alloc(%[[LOAD_31]]) {alignment = 64 : i64} : memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_34:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[ALLOC_35:.*]] = memref.alloc() {alignment = 64 : i64} : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[COMPUTE_3:.*]] = hipsr.compute(%[[VAL_25]]) ins(%[[VAL_26]], %[[VAL_27]] : memref<1xi64, #hipsr.mem<host>>, memref<3x?xi64, #hipsr.mem<device>>) outs(%[[ALLOC_32]] : memref<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_28:.*]]: !hipsr.context, %[[VAL_29:.*]]: memref<1xi64, #hipsr.mem<host>>, %[[VAL_30:.*]]: memref<3x?xi64, #hipsr.mem<device>>, %[[VAL_31:.*]]: memref<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:          %[[CONSTANT_26:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[DIM_14:.*]] = memref.dim %[[VAL_31]], %[[CONSTANT_26]] : memref<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:          %[[SUBVIEW_6:.*]] = memref.subview %[[VAL_30]]{{\[}}0, 0] [3, %[[DIM_14]]] [1, 1] : memref<3x?xi64, #hipsr.mem<device>> to memref<3x?xi64, strided<[?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:          hipsr.compute_yield %[[SUBVIEW_6]] : memref<3x?xi64, strided<[?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:        } : memref<3x?xi64, strided<[?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.transpose(%[[VAL_25]]) ins(%[[COMPUTE_3]] : memref<3x?xi64, strided<[?, 1]>, #hipsr.mem<device>>) outs(%[[ALLOC_33]] : memref<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>}
// CHECK-NEXT:        %[[COMPUTE_4:.*]] = hipsr.compute(%[[VAL_25]]) ins(%[[ALLOC_33]] : memref<?x3xi64, #hipsr.mem<device>>) outs(%[[ALLOC_34]] : memref<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_32:.*]]: !hipsr.context, %[[VAL_33:.*]]: memref<?x3xi64, #hipsr.mem<device>>, %[[VAL_34:.*]]: memref<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:          %[[CONSTANT_27:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[CONSTANT_28:.*]] = arith.constant 3 : i64
// CHECK-NEXT:          %[[CONSTANT_29:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[DIM_15:.*]] = memref.dim %[[VAL_33]], %[[CONSTANT_29]] : memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:          %[[INDEX_CAST_9:.*]] = arith.index_cast %[[DIM_15]] : index to i64
// CHECK-NEXT:          memref.store %[[INDEX_CAST_9]], %[[VAL_34]]{{\[}}%[[CONSTANT_29]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:          memref.store %[[CONSTANT_28]], %[[VAL_34]]{{\[}}%[[CONSTANT_27]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:          hipsr.compute_yield %[[VAL_34]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        } : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[COMPUTE_5:.*]] = hipsr.compute(%[[VAL_25]]) ins(%[[COMPUTE_4]] : memref<2xi64, #hipsr.mem<host>>) outs(%[[ALLOC_35]] : memref<i64, #hipsr.mem<host>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_35:.*]]: !hipsr.context, %[[VAL_36:.*]]: memref<2xi64, #hipsr.mem<host>>, %[[VAL_37:.*]]: memref<i64, #hipsr.mem<host>>):
// CHECK-NEXT:          %[[CONSTANT_30:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[LOAD_32:.*]] = memref.load %[[VAL_36]]{{\[}}%[[CONSTANT_30]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:          memref.store %[[LOAD_32]], %[[VAL_37]]{{\[}}] : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:          hipsr.compute_yield %[[VAL_37]] : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:        } : memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[COMPUTE_6:.*]] = hipsr.compute(%[[VAL_25]]) ins(%[[COMPUTE_5]] : memref<i64, #hipsr.mem<host>>) outs(%[[COMPUTE_5]] : memref<i64, #hipsr.mem<host>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_38:.*]]: !hipsr.context, %[[VAL_39:.*]]: memref<i64, #hipsr.mem<host>>, %[[VAL_40:.*]]: memref<i64, #hipsr.mem<host>>):
// CHECK-NEXT:          %[[EXPAND_SHAPE_1:.*]] = memref.expand_shape %[[VAL_39]] [] output_shape [1] : memref<i64, #hipsr.mem<host>> into memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:          hipsr.compute_yield %[[EXPAND_SHAPE_1]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        } : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_30]], %[[COMPUTE_3]] : memref<2xindex>, memref<3x?xi64, strided<[?, 1]>, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_31]], %[[ALLOC_33]] : memref<2xindex>, memref<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_28]], %[[COMPUTE_4]] : memref<1xindex>, memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_29]], %[[COMPUTE_5]] : memref<0xindex>, memref<i64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_27]], %[[COMPUTE_6]] : memref<1xindex>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ALLOC_33]], %[[ALLOC_33]], %[[COMPUTE_6]] : memref<?x3xi64, #hipsr.mem<device>>, memref<?x3xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } -> memref<?x3xi64, #hipsr.mem<device>>, memref<?x3xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<host>> {domain_id = 3 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_4:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]]#0, %[[POOL_DOMAIN_3]]#2, %[[POOL_DOMAIN_0]]#2, %[[POOL_DOMAIN_3]]#0, %[[POOL_DOMAIN_0]]#3, %[[POOL_DOMAIN_3]]#1 : !hipsr.context, memref<?xf16, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<host>>, memref<?x?x4096xf16, #hipsr.mem<device>>, memref<?x3xi64, #hipsr.mem<device>>, memref<?x?x4096xf16, #hipsr.mem<device>>, memref<?x3xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_41:.*]]: !hipsr.context, %[[VAL_42:.*]]: memref<?xf16, #hipsr.mem<device>>, %[[VAL_43:.*]]: memref<1xi64, #hipsr.mem<host>>, %[[VAL_44:.*]]: memref<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_45:.*]]: memref<?x3xi64, #hipsr.mem<device>>, %[[VAL_46:.*]]: memref<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_47:.*]]: memref<?x3xi64, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[CONSTANT_31:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[CONSTANT_32:.*]] = arith.constant 4096 : index
// CHECK-NEXT:        %[[CONSTANT_33:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[CONSTANT_34:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[DIM_16:.*]] = memref.dim %[[VAL_42]], %[[CONSTANT_34]] : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[LOAD_33:.*]] = memref.load %[[VAL_43]]{{\[}}%[[CONSTANT_34]]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:        %[[INDEX_CAST_10:.*]] = arith.index_cast %[[LOAD_33]] : i64 to index
// CHECK-NEXT:        %[[CMPI_9:.*]] = arith.cmpi slt, %[[INDEX_CAST_10]], %[[CONSTANT_34]] : index
// CHECK-NEXT:        %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_10]], %[[DIM_16]] : index
// CHECK-NEXT:        %[[SELECT_3:.*]] = arith.select %[[CMPI_9]], %[[ADDI_0]], %[[INDEX_CAST_10]] : index
// CHECK-NEXT:        %[[MINSI_0:.*]] = arith.minsi %[[DIM_16]], %[[CONSTANT_34]] : index
// CHECK-NEXT:        %[[MAXSI_0:.*]] = arith.maxsi %[[SELECT_3]], %[[CONSTANT_34]] : index
// CHECK-NEXT:        %[[MINSI_1:.*]] = arith.minsi %[[MAXSI_0]], %[[DIM_16]] : index
// CHECK-NEXT:        %[[SUBI_1:.*]] = arith.subi %[[MINSI_1]], %[[MINSI_0]] : index
// CHECK-NEXT:        %[[MAXSI_1:.*]] = arith.maxsi %[[SUBI_1]], %[[CONSTANT_34]] : index
// CHECK-NEXT:        %[[ALLOC_36:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1xindex>
// CHECK-NEXT:        memref.store %[[MAXSI_1]], %[[ALLOC_36]]{{\[}}%[[CONSTANT_34]]] : memref<1xindex>
// CHECK-NEXT:        %[[DIM_17:.*]] = memref.dim %[[VAL_44]], %[[CONSTANT_34]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_18:.*]] = memref.dim %[[VAL_44]], %[[CONSTANT_33]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ALLOC_37:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xindex>
// CHECK-NEXT:        memref.store %[[DIM_17]], %[[ALLOC_37]]{{\[}}%[[CONSTANT_34]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[DIM_18]], %[[ALLOC_37]]{{\[}}%[[CONSTANT_33]]] : memref<3xindex>
// CHECK-NEXT:        memref.store %[[CONSTANT_32]], %[[ALLOC_37]]{{\[}}%[[CONSTANT_31]]] : memref<3xindex>
// CHECK-NEXT:        %[[LOAD_34:.*]] = memref.load %[[ALLOC_36]]{{\[}}%[[CONSTANT_34]]] : memref<1xindex>
// CHECK-NEXT:        %[[ALLOC_38:.*]] = memref.alloc(%[[LOAD_34]]) {alignment = 64 : i64} : memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_19:.*]] = memref.dim %[[VAL_44]], %[[CONSTANT_34]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[DIM_20:.*]] = memref.dim %[[VAL_44]], %[[CONSTANT_33]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CONSTANT_35:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[LOAD_35:.*]] = memref.load %[[ALLOC_37]]{{\[}}%[[CONSTANT_35]]] : memref<3xindex>
// CHECK-NEXT:        %[[CONSTANT_36:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[LOAD_36:.*]] = memref.load %[[ALLOC_37]]{{\[}}%[[CONSTANT_36]]] : memref<3xindex>
// CHECK-NEXT:        %[[ALLOC_OUTPUT_0:.*]] = hipsr.alloc_output(%[[VAL_41]], %[[LOAD_35]], %[[LOAD_36]]) {out_idx = 0 : i64} : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.slice(%[[VAL_41]]) ins(%[[VAL_42]] : memref<?xf16, #hipsr.mem<device>>) ends(%[[VAL_43]] : memref<1xi64, #hipsr.mem<host>>) outs(%[[ALLOC_38]] : memref<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>}
// CHECK-NEXT:        hipsr.scatter_nd(%[[VAL_41]]) ins(%[[VAL_46]], %[[VAL_47]], %[[ALLOC_38]] : memref<?x?x4096xf16, #hipsr.mem<device>>, memref<?x3xi64, #hipsr.mem<device>>, memref<?xf16, #hipsr.mem<device>>) outs(%[[ALLOC_OUTPUT_0]] : memref<?x?x4096xf16, #hipsr.mem<device>>)
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_36]], %[[ALLOC_38]] : memref<1xindex>, memref<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.preserve_shape %[[ALLOC_37]], %[[ALLOC_OUTPUT_0]] : memref<3xindex>, memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ALLOC_OUTPUT_0]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> memref<?x?x4096xf16, #hipsr.mem<device>> {domain_id = 4 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_4]] : memref<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }
// CHECK-NEXT:  }



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
