// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: hipsr pipeline on a dynamically shaped embedding graph.
//
// The input is the ONNX-MLIR form of a text-embedding graph that scatters image
// features into token embeddings. Every batch and sequence extent stays
// symbolic, and NonZero makes the scatter index count data-dependent, so the
// pipeline must never freeze an extent to a constant.
//
// The graph is importer output with one edit: the omitted `steps` operand of
// onnx.Slice is spelled onnx.NoValue. The embedding table stays as imported, an
// external byte range naming the model's two-gigabyte weight file, so the RUN
// line creates a file of exactly that size for the conversion to map. Only its
// length matters: the compiler records the path and offset and never reads the
// weights, so the contents are left undefined.
//
// The whole output is checked. The pipeline decides how many pool domains to
// cut, where each cut sits, and what every shape computation does, so a partial
// check would let those move unnoticed.
//
// Pool domains and what cuts them
// -------------------------------
//
// A domain is one pool allocation, so every buffer inside it must be sized
// before it runs. The pipeline therefore starts a new domain wherever a shape
// depends on a value the host cannot know until the previous domain has
// finished. Each cut below reads the host tensor named on the arrow.
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
// Everything in domain 0 is shaped by the argument types alone. Domains 1
// through 4 each open with a shape computation that extracts extents from a
// host buffer an earlier domain filled, which is what a cut looks like below.
//
// The three zones of a domain
// ---------------------------
//
// hipsr-materialize-init-tensors leaves each domain body in order: the
// scf.execute_region shape computations, then the tensor.empty allocations that
// read their dynamic extents back out of those shapes, then the data ops. A
// constant a shape computation reads moves up with it; the rest stay where
// conversion left them.
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../Inputs/make_external_data.py %t/embedding.onnx.data 2034237440 && cd %t && hip-mlir-opt --onnx-dialect=modeled --hipsr-pipeline --mlir-elide-resource-strings-if-larger=32 %s | FileCheck %s

module {

// The context becomes argument 0 and every symbolic extent survives.
// CHECK-LABEL:   func.func @main_graph(
// CHECK-SAME:      %[[ARG0:[^,]*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:[^,]*]]: tensor<?x?xi64, #hipsr.mem<device>> {onnx.name = "input_ids"},
// CHECK-SAME:      %[[ARG2:[^,]*]]: tensor<?x4096xf16, #hipsr.mem<device>> {onnx.name = "image_features"}) -> (tensor<?x?x4096xf16, #hipsr.mem<device>> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {

// Domain 0 takes everything whose shape follows from the graph inputs alone:
// the embedding lookup, the bool mask, and the shape vector the broadcasts
// downstream will read. Its six constants come first, in conversion order, so
// each stays ahead of the two shape computations that read one, then the five
// computations follow.
// CHECK-NEXT:           %[[POOL_DOMAIN_0:.*]]:8 = hipsr.pool_domain(%[[ARG0]], %[[ARG2]], %[[ARG1]] : !hipsr.context, tensor<?x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: tensor<?x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = hipsr.constant {value = dense_resource<"file|embedding.onnx.data|0"> : tensor<248320x4096xf16, #hipsr.mem<device>>} : tensor<248320x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_7:.*]] = arith.constant dense<248056> : tensor<i64>
// CHECK-NEXT:             %[[CONSTANT_8:.*]] = hipsr.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_9:.*]] = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:             %[[CONSTANT_10:.*]] = hipsr.constant {value = dense<0> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = hipsr.constant {value = dense<248056> : tensor<i64>} : tensor<i64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[SHAPE_OF_0:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x4096xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:               %[[NUM_ELEMENTS_0:.*]] = shape.num_elements %[[SHAPE_OF_0]] : !shape.shape -> !shape.size
// CHECK-NEXT:               %[[SIZE_TO_INDEX_0:.*]] = shape.size_to_index %[[NUM_ELEMENTS_0]] : !shape.size
// CHECK-NEXT:               %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIVUI_0:.*]] = arith.divui %[[SIZE_TO_INDEX_0]], %[[CONSTANT_0]] : index
// CHECK-NEXT:               %[[FROM_EXTENTS_0:.*]] = shape.from_extents %[[DIVUI_0]] : index
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_0]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_1:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[SHAPE_OF_1:.*]] = shape.shape_of %[[VAL_2]] : tensor<?x?xi64, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:               %[[SHAPE_OF_2:.*]] = shape.shape_of %[[CONSTANT_1]] : tensor<i64, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:               %[[BROADCAST_0:.*]] = shape.broadcast %[[SHAPE_OF_1]], %[[SHAPE_OF_2]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:               scf.yield %[[BROADCAST_0]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_2:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONST_SIZE_0:.*]] = shape.const_size 0
// CHECK-NEXT:               %[[GET_EXTENT_0:.*]] = shape.get_extent %[[EXECUTE_REGION_1]], %[[CONST_SIZE_0]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[SIZE_TO_INDEX_1:.*]] = shape.size_to_index %[[GET_EXTENT_0]] : !shape.size
// CHECK-NEXT:               %[[CONST_SIZE_1:.*]] = shape.const_size 1
// CHECK-NEXT:               %[[GET_EXTENT_1:.*]] = shape.get_extent %[[EXECUTE_REGION_1]], %[[CONST_SIZE_1]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[SIZE_TO_INDEX_2:.*]] = shape.size_to_index %[[GET_EXTENT_1]] : !shape.size
// CHECK-NEXT:               %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_1:.*]] = shape.from_extents %[[SIZE_TO_INDEX_1]], %[[SIZE_TO_INDEX_2]], %[[CONSTANT_4]] : index, index, index
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_3:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[SHAPE_OF_3:.*]] = shape.shape_of %[[CONSTANT_5]] : tensor<248320x4096xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:               %[[SHAPE_OF_4:.*]] = shape.shape_of %[[VAL_2]] : tensor<?x?xi64, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:               %[[CONST_SIZE_2:.*]] = shape.const_size 0
// CHECK-NEXT:               %[[VAL_3:.*]], %[[VAL_4:.*]] = "shape.split_at"(%[[SHAPE_OF_3]], %[[CONST_SIZE_2]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:               %[[CONST_SIZE_3:.*]] = shape.const_size 1
// CHECK-NEXT:               %[[VAL_5:.*]], %[[VAL_6:.*]] = "shape.split_at"(%[[SHAPE_OF_3]], %[[CONST_SIZE_3]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:               %[[CONCAT_0:.*]] = shape.concat %[[VAL_3]], %[[SHAPE_OF_4]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:               %[[CONCAT_1:.*]] = shape.concat %[[CONCAT_0]], %[[VAL_6]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:               scf.yield %[[CONCAT_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_4:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONSTANT_6:.*]] = arith.constant 3 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_2:.*]] = shape.from_extents %[[CONSTANT_6]] : index
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_2]] : !shape.shape
// CHECK-NEXT:             }

// The allocation zone. Each tensor.empty takes its dynamic extents from the
// shape yielded above; a fully static result type needs none.
// CHECK-NEXT:             %[[CONST_SIZE_4:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_2:.*]] = shape.get_extent %[[EXECUTE_REGION_0]], %[[CONST_SIZE_4]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_3:.*]] = shape.size_to_index %[[GET_EXTENT_2]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_0:.*]] = tensor.empty(%[[SIZE_TO_INDEX_3]]) : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONST_SIZE_5:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_3:.*]] = shape.get_extent %[[EXECUTE_REGION_1]], %[[CONST_SIZE_5]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_4:.*]] = shape.size_to_index %[[GET_EXTENT_3]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_6:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_4:.*]] = shape.get_extent %[[EXECUTE_REGION_1]], %[[CONST_SIZE_6]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_5:.*]] = shape.size_to_index %[[GET_EXTENT_4]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_1:.*]] = tensor.empty(%[[SIZE_TO_INDEX_4]], %[[SIZE_TO_INDEX_5]]) : tensor<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONST_SIZE_7:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_5:.*]] = shape.get_extent %[[EXECUTE_REGION_2]], %[[CONST_SIZE_7]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_6:.*]] = shape.size_to_index %[[GET_EXTENT_5]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_8:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_6:.*]] = shape.get_extent %[[EXECUTE_REGION_2]], %[[CONST_SIZE_8]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_7:.*]] = shape.size_to_index %[[GET_EXTENT_6]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_2:.*]] = tensor.empty(%[[SIZE_TO_INDEX_6]], %[[SIZE_TO_INDEX_7]]) : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONST_SIZE_9:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_7:.*]] = shape.get_extent %[[EXECUTE_REGION_3]], %[[CONST_SIZE_9]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_8:.*]] = shape.size_to_index %[[GET_EXTENT_7]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_10:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_8:.*]] = shape.get_extent %[[EXECUTE_REGION_3]], %[[CONST_SIZE_10]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_9:.*]] = shape.size_to_index %[[GET_EXTENT_8]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_3:.*]] = tensor.empty(%[[SIZE_TO_INDEX_8]], %[[SIZE_TO_INDEX_9]]) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_4:.*]] = tensor.empty() : tensor<3xi64, #hipsr.mem<host>>

// The data zone, in conversion order, each op now initialized by a tensor.empty.
// CHECK-NEXT:             %[[COMPUTE_0:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<?x4096xf16, #hipsr.mem<device>>) outs(%[[EMPTY_0]] : tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_7:.*]]: !hipsr.context, %[[VAL_8:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %[[VAL_9:.*]]: tensor<?xf16, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[VAL_8]] {{\[\[}}0, 1]] : tensor<?x4096xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[COLLAPSE_SHAPE_0]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EQUAL_0:.*]] = hipsr.equal(%[[VAL_0]]) ins(%[[VAL_2]], %[[CONSTANT_1]] : tensor<?x?xi64, #hipsr.mem<device>>, tensor<i64, #hipsr.mem<device>>) outs(%[[EMPTY_1]] : tensor<?x?xi1, #hipsr.mem<device>>) : tensor<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[COMPUTE_1:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[EQUAL_0]] : tensor<?x?xi1, #hipsr.mem<device>>) outs(%[[EMPTY_2]] : tensor<?x?x1xi1, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_10:.*]]: !hipsr.context, %[[VAL_11:.*]]: tensor<?x?xi1, #hipsr.mem<device>>, %[[VAL_12:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[CONSTANT_11:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_0:.*]] = tensor.dim %[[VAL_12]], %[[CONSTANT_11]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               %[[CONSTANT_12:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIM_1:.*]] = tensor.dim %[[VAL_12]], %[[CONSTANT_12]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[VAL_11]] {{\[\[}}0], [1, 2]] output_shape {{\[}}%[[DIM_0]], %[[DIM_1]], 1] : tensor<?x?xi1, #hipsr.mem<device>> into tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXPAND_SHAPE_0]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[GATHER_0:.*]] = hipsr.gather(%[[VAL_0]]) ins(%[[CONSTANT_5]], %[[VAL_2]] : tensor<248320x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) outs(%[[EMPTY_3]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64} : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[COMPUTE_2:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[GATHER_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[EMPTY_4]] : tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_13:.*]]: !hipsr.context, %[[VAL_14:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_15:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_13:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_2:.*]] = tensor.dim %[[VAL_14]], %[[CONSTANT_13]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK-NEXT:               %[[CONSTANT_14:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIM_3:.*]] = tensor.dim %[[VAL_14]], %[[CONSTANT_14]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK-NEXT:               %[[CONSTANT_15:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:               %[[FROM_ELEMENTS_0:.*]] = tensor.from_elements %[[INDEX_CAST_0]], %[[INDEX_CAST_1]], %[[CONSTANT_15]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_0]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_0]], %[[COMPUTE_0]], %[[EMPTY_2]], %[[COMPUTE_1]], %[[EMPTY_3]], %[[GATHER_0]], %[[EMPTY_4]], %[[COMPUTE_2]] : tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<3xi64, #hipsr.mem<host>> {domain_id = 0 : i64}

// Each broadcast reads its extents from a host shape vector rather than from
// the type, so each one opens a domain whose shape computation extracts them
// with tensor.extract and rebuilds the destination shape.
// CHECK-NEXT:           %[[POOL_DOMAIN_1:.*]]:2 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]]#2, %[[POOL_DOMAIN_0]]#6, %[[POOL_DOMAIN_0]]#3, %[[POOL_DOMAIN_0]]#7 : !hipsr.context, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_17:.*]]: !hipsr.context, %[[VAL_18:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_19:.*]]: tensor<3xi64, #hipsr.mem<host>>, %[[VAL_20:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_21:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:             %[[EXECUTE_REGION_5:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[SHAPE_OF_5:.*]] = shape.shape_of %[[VAL_18]] : tensor<?x?x1xi1, #hipsr.mem<device>> -> tensor<3xindex>
// CHECK-NEXT:               %[[CONSTANT_16:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_0:.*]] = tensor.extract %[[VAL_19]]{{\[}}%[[CONSTANT_16]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_2:.*]] = arith.index_cast %[[EXTRACT_0]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_17:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[EXTRACT_1:.*]] = tensor.extract %[[VAL_19]]{{\[}}%[[CONSTANT_17]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_3:.*]] = arith.index_cast %[[EXTRACT_1]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_18:.*]] = arith.constant 2 : index
// CHECK-NEXT:               %[[EXTRACT_2:.*]] = tensor.extract %[[VAL_19]]{{\[}}%[[CONSTANT_18]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_4:.*]] = arith.index_cast %[[EXTRACT_2]] : i64 to index
// CHECK-NEXT:               %[[FROM_EXTENTS_3:.*]] = shape.from_extents %[[INDEX_CAST_2]], %[[INDEX_CAST_3]], %[[INDEX_CAST_4]] : index, index, index
// CHECK-NEXT:               %[[CSTR_BROADCASTABLE_0:.*]] = shape.cstr_broadcastable %[[SHAPE_OF_5]], %[[FROM_EXTENTS_3]] : tensor<3xindex>, !shape.shape
// CHECK-NEXT:               %[[ASSUMING_0:.*]] = shape.assuming %[[CSTR_BROADCASTABLE_0]] -> (!shape.shape) {
// CHECK-NEXT:                 %[[BROADCAST_1:.*]] = shape.broadcast %[[SHAPE_OF_5]], %[[FROM_EXTENTS_3]] : tensor<3xindex>, !shape.shape -> !shape.shape
// CHECK-NEXT:                 shape.assuming_yield %[[BROADCAST_1]] : !shape.shape
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %[[ASSUMING_0]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[CONST_SIZE_11:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_9:.*]] = shape.get_extent %[[EXECUTE_REGION_5]], %[[CONST_SIZE_11]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_10:.*]] = shape.size_to_index %[[GET_EXTENT_9]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_12:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_10:.*]] = shape.get_extent %[[EXECUTE_REGION_5]], %[[CONST_SIZE_12]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_11:.*]] = shape.size_to_index %[[GET_EXTENT_10]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_13:.*]] = shape.const_size 2
// CHECK-NEXT:             %[[GET_EXTENT_11:.*]] = shape.get_extent %[[EXECUTE_REGION_5]], %[[CONST_SIZE_13]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_12:.*]] = shape.size_to_index %[[GET_EXTENT_11]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_5:.*]] = tensor.empty(%[[SIZE_TO_INDEX_10]], %[[SIZE_TO_INDEX_11]], %[[SIZE_TO_INDEX_12]]) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EXPAND_0:.*]] = hipsr.expand(%[[VAL_17]]) ins(%[[VAL_20]], %[[VAL_21]] : tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) outs(%[[EMPTY_5]] : tensor<?x?x?xi1, #hipsr.mem<device>>) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_5]], %[[EXPAND_0]] : tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:           } -> tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<?x?x?xi1, #hipsr.mem<device>> {domain_id = 1 : i64}

// The second broadcast feeds the search. Its destination holds one row per
// input axis and, in the worst case, a column per input element; the count of
// the columns it filled sits beside it and is read back to the host.
// CHECK-NEXT:           %[[POOL_DOMAIN_2:.*]]:4 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_1]]#0, %[[POOL_DOMAIN_0]]#6, %[[POOL_DOMAIN_1]]#1, %[[POOL_DOMAIN_0]]#7 : !hipsr.context, tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_24:.*]]: !hipsr.context, %[[VAL_25:.*]]: tensor<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_26:.*]]: tensor<3xi64, #hipsr.mem<host>>, %[[VAL_27:.*]]: tensor<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_28:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:             %[[EXECUTE_REGION_6:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[SHAPE_OF_6:.*]] = shape.shape_of %[[VAL_25]] : tensor<?x?x?xi1, #hipsr.mem<device>> -> tensor<3xindex>
// CHECK-NEXT:               %[[CONSTANT_19:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_3:.*]] = tensor.extract %[[VAL_26]]{{\[}}%[[CONSTANT_19]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_5:.*]] = arith.index_cast %[[EXTRACT_3]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_20:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[EXTRACT_4:.*]] = tensor.extract %[[VAL_26]]{{\[}}%[[CONSTANT_20]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_6:.*]] = arith.index_cast %[[EXTRACT_4]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_21:.*]] = arith.constant 2 : index
// CHECK-NEXT:               %[[EXTRACT_5:.*]] = tensor.extract %[[VAL_26]]{{\[}}%[[CONSTANT_21]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_7:.*]] = arith.index_cast %[[EXTRACT_5]] : i64 to index
// CHECK-NEXT:               %[[FROM_EXTENTS_4:.*]] = shape.from_extents %[[INDEX_CAST_5]], %[[INDEX_CAST_6]], %[[INDEX_CAST_7]] : index, index, index
// CHECK-NEXT:               %[[CSTR_BROADCASTABLE_1:.*]] = shape.cstr_broadcastable %[[SHAPE_OF_6]], %[[FROM_EXTENTS_4]] : tensor<3xindex>, !shape.shape
// CHECK-NEXT:               %[[ASSUMING_1:.*]] = shape.assuming %[[CSTR_BROADCASTABLE_1]] -> (!shape.shape) {
// CHECK-NEXT:                 %[[BROADCAST_2:.*]] = shape.broadcast %[[SHAPE_OF_6]], %[[FROM_EXTENTS_4]] : tensor<3xindex>, !shape.shape -> !shape.shape
// CHECK-NEXT:                 shape.assuming_yield %[[BROADCAST_2]] : !shape.shape
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %[[ASSUMING_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_7:.*]]:2 = scf.execute_region -> (!shape.shape, !shape.shape) {
// CHECK-NEXT:               %[[CONST_SIZE_14:.*]] = shape.const_size 3
// CHECK-NEXT:               %[[NUM_ELEMENTS_1:.*]] = shape.num_elements %[[EXECUTE_REGION_6]] : !shape.shape -> !shape.size
// CHECK-NEXT:               %[[FROM_EXTENTS_5:.*]] = shape.from_extents %[[CONST_SIZE_14]], %[[NUM_ELEMENTS_1]] : !shape.size, !shape.size
// CHECK-NEXT:               %[[CONST_SHAPE_0:.*]] = shape.const_shape [1] : !shape.shape
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_5]], %[[CONST_SHAPE_0]] : !shape.shape, !shape.shape
// CHECK-NEXT:             }

// The host count is a 1xi64 buffer, so its allocation needs no extent and this
// shape goes unread.
// CHECK-NEXT:             %[[EXECUTE_REGION_8:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               scf.yield %[[EXECUTE_REGION_7]]#1 : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[CONST_SIZE_15:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_12:.*]] = shape.get_extent %[[EXECUTE_REGION_6]], %[[CONST_SIZE_15]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_13:.*]] = shape.size_to_index %[[GET_EXTENT_12]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_16:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_13:.*]] = shape.get_extent %[[EXECUTE_REGION_6]], %[[CONST_SIZE_16]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_14:.*]] = shape.size_to_index %[[GET_EXTENT_13]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_17:.*]] = shape.const_size 2
// CHECK-NEXT:             %[[GET_EXTENT_14:.*]] = shape.get_extent %[[EXECUTE_REGION_6]], %[[CONST_SIZE_17]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_15:.*]] = shape.size_to_index %[[GET_EXTENT_14]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_6:.*]] = tensor.empty(%[[SIZE_TO_INDEX_13]], %[[SIZE_TO_INDEX_14]], %[[SIZE_TO_INDEX_15]]) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONST_SIZE_18:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_15:.*]] = shape.get_extent %[[EXECUTE_REGION_7]]#0, %[[CONST_SIZE_18]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_16:.*]] = shape.size_to_index %[[GET_EXTENT_15]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_7:.*]] = tensor.empty(%[[SIZE_TO_INDEX_16]]) : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_8:.*]] = tensor.empty() : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_9:.*]] = tensor.empty() : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[EXPAND_1:.*]] = hipsr.expand(%[[VAL_24]]) ins(%[[VAL_27]], %[[VAL_28]] : tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) outs(%[[EMPTY_6]] : tensor<?x?x?xi1, #hipsr.mem<device>>) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[NONZERO_0:.*]]:2 = hipsr.nonzero(%[[VAL_24]]) ins(%[[EXPAND_1]] : tensor<?x?x?xi1, #hipsr.mem<device>>) outs(%[[EMPTY_7]], %[[EMPTY_8]] : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[VAL_31:.*]] = hipsr.copy_d2h(%[[VAL_24]]) ins(%[[NONZERO_0]]#1 : tensor<1xi64, #hipsr.mem<device>>) outs(%[[EMPTY_9]] : tensor<1xi64, #hipsr.mem<host>>) : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_7]], %[[NONZERO_0]]#0, %[[EMPTY_9]], %[[VAL_31]] : tensor<3x?xi64, #hipsr.mem<device>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<3x?xi64, #hipsr.mem<device>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>> {domain_id = 2 : i64}

// That host count opens the next domain, where the first shape computation
// turns it into the trailing extent so the search destination narrows to the
// columns that hold a position. The domain takes both buffers twice, and the
// shape computation reads only the count while the data op reads only the
// coordinates.
// CHECK-NEXT:           %[[POOL_DOMAIN_3:.*]]:4 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_2]]#2, %[[POOL_DOMAIN_2]]#0, %[[POOL_DOMAIN_2]]#3, %[[POOL_DOMAIN_2]]#1 : !hipsr.context, tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_33:.*]]: !hipsr.context, %[[VAL_34:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_35:.*]]: tensor<3x?xi64, #hipsr.mem<device>>, %[[VAL_36:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_37:.*]]: tensor<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[EXECUTE_REGION_9:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONSTANT_22:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_6:.*]] = tensor.extract %[[VAL_34]]{{\[}}%[[CONSTANT_22]]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_8:.*]] = arith.index_cast %[[EXTRACT_6]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_23:.*]] = arith.constant 3 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_6:.*]] = shape.from_extents %[[CONSTANT_23]], %[[INDEX_CAST_8]] : index, index
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_6]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_10:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONST_SIZE_19:.*]] = shape.const_size 1
// CHECK-NEXT:               %[[GET_EXTENT_16:.*]] = shape.get_extent %[[EXECUTE_REGION_9]], %[[CONST_SIZE_19]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[CONST_SIZE_20:.*]] = shape.const_size 0
// CHECK-NEXT:               %[[GET_EXTENT_17:.*]] = shape.get_extent %[[EXECUTE_REGION_9]], %[[CONST_SIZE_20]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[FROM_EXTENTS_7:.*]] = shape.from_extents %[[GET_EXTENT_16]], %[[GET_EXTENT_17]] : !shape.size, !shape.size
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_7]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_11:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONSTANT_24:.*]] = arith.constant 2 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_8:.*]] = shape.from_extents %[[CONSTANT_24]] : index
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_8]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_12:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONST_SHAPE_1:.*]] = shape.const_shape [] : !shape.shape
// CHECK-NEXT:               scf.yield %[[CONST_SHAPE_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_13:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONSTANT_25:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_9:.*]] = shape.from_extents %[[CONSTANT_25]] : index
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_9]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[CONST_SIZE_21:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_18:.*]] = shape.get_extent %[[EXECUTE_REGION_9]], %[[CONST_SIZE_21]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_17:.*]] = shape.size_to_index %[[GET_EXTENT_18]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_10:.*]] = tensor.empty(%[[SIZE_TO_INDEX_17]]) : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONST_SIZE_22:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_19:.*]] = shape.get_extent %[[EXECUTE_REGION_10]], %[[CONST_SIZE_22]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_18:.*]] = shape.size_to_index %[[GET_EXTENT_19]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_11:.*]] = tensor.empty(%[[SIZE_TO_INDEX_18]]) : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_12:.*]] = tensor.empty() : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[EMPTY_13:.*]] = tensor.empty() : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[EMPTY_14:.*]] = tensor.empty() : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[COMPUTE_3:.*]] = hipsr.compute(%[[VAL_33]]) ins(%[[VAL_36]], %[[VAL_37]] : tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[EMPTY_10]] : tensor<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_38:.*]]: !hipsr.context, %[[VAL_39:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_40:.*]]: tensor<3x?xi64, #hipsr.mem<device>>, %[[VAL_41:.*]]: tensor<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[CONSTANT_26:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIM_4:.*]] = tensor.dim %[[VAL_41]], %[[CONSTANT_26]] : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[VAL_40]][0, 0] [3, %[[DIM_4]]] [1, 1] : tensor<3x?xi64, #hipsr.mem<device>> to tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXTRACT_SLICE_0]] : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[TRANSPOSE_0:.*]] = hipsr.transpose(%[[VAL_33]]) ins(%[[COMPUTE_3]] : tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[EMPTY_11]] : tensor<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>} : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[COMPUTE_4:.*]] = hipsr.compute(%[[VAL_33]]) ins(%[[TRANSPOSE_0]] : tensor<?x3xi64, #hipsr.mem<device>>) outs(%[[EMPTY_12]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_42:.*]]: !hipsr.context, %[[VAL_43:.*]]: tensor<?x3xi64, #hipsr.mem<device>>, %[[VAL_44:.*]]: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_27:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_5:.*]] = tensor.dim %[[VAL_43]], %[[CONSTANT_27]] : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_9:.*]] = arith.index_cast %[[DIM_5]] : index to i64
// CHECK-NEXT:               %[[CONSTANT_28:.*]] = arith.constant 3 : i64
// CHECK-NEXT:               %[[FROM_ELEMENTS_1:.*]] = tensor.from_elements %[[INDEX_CAST_9]], %[[CONSTANT_28]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[COMPUTE_5:.*]] = hipsr.compute(%[[VAL_33]]) ins(%[[COMPUTE_4]] : tensor<2xi64, #hipsr.mem<host>>) outs(%[[EMPTY_13]] : tensor<i64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_45:.*]]: !hipsr.context, %[[VAL_46:.*]]: tensor<2xi64, #hipsr.mem<host>>, %[[VAL_47:.*]]: tensor<i64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_29:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_7:.*]] = tensor.extract %[[VAL_46]]{{\[}}%[[CONSTANT_29]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[FROM_ELEMENTS_2:.*]] = tensor.from_elements %[[EXTRACT_7]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_2]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[COMPUTE_6:.*]] = hipsr.compute(%[[VAL_33]]) ins(%[[COMPUTE_5]] : tensor<i64, #hipsr.mem<host>>) outs(%[[EMPTY_14]] : tensor<1xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_48:.*]]: !hipsr.context, %[[VAL_49:.*]]: tensor<i64, #hipsr.mem<host>>, %[[VAL_50:.*]]: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[VAL_49]] [] output_shape [1] : tensor<i64, #hipsr.mem<host>> into tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXPAND_SHAPE_1]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_11]], %[[TRANSPOSE_0]], %[[EMPTY_14]], %[[COMPUTE_6]] : tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>> {domain_id = 3 : i64}

// The update window ends at that same count, and the scatter writes it into
// the embeddings.
// CHECK-NEXT:           %[[POOL_DOMAIN_4:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]]#0, %[[POOL_DOMAIN_3]]#2, %[[POOL_DOMAIN_0]]#1, %[[POOL_DOMAIN_3]]#3, %[[POOL_DOMAIN_0]]#4, %[[POOL_DOMAIN_3]]#0, %[[POOL_DOMAIN_0]]#5, %[[POOL_DOMAIN_3]]#1 : !hipsr.context, tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_53:.*]]: !hipsr.context, %[[VAL_54:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[VAL_55:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_56:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[VAL_57:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_58:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_59:.*]]: tensor<?x3xi64, #hipsr.mem<device>>, %[[VAL_60:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_61:.*]]: tensor<?x3xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[EXECUTE_REGION_14:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[CONSTANT_30:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_6:.*]] = tensor.dim %[[VAL_54]], %[[CONSTANT_30]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[CONSTANT_31:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_8:.*]] = tensor.extract %[[VAL_55]]{{\[}}%[[CONSTANT_31]]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_10:.*]] = arith.index_cast %[[EXTRACT_8]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_32:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[INDEX_CAST_10]], %[[CONSTANT_32]] : index
// CHECK-NEXT:               %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_10]], %[[DIM_6]] : index
// CHECK-NEXT:               %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_0]], %[[INDEX_CAST_10]] : index
// CHECK-NEXT:               %[[CONSTANT_33:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[MINSI_0:.*]] = arith.minsi %[[DIM_6]], %[[CONSTANT_33]] : index
// CHECK-NEXT:               %[[MAXSI_0:.*]] = arith.maxsi %[[SELECT_0]], %[[CONSTANT_32]] : index
// CHECK-NEXT:               %[[MINSI_1:.*]] = arith.minsi %[[MAXSI_0]], %[[DIM_6]] : index
// CHECK-NEXT:               %[[SUBI_0:.*]] = arith.subi %[[MINSI_1]], %[[MINSI_0]] : index
// CHECK-NEXT:               %[[MAXSI_1:.*]] = arith.maxsi %[[SUBI_0]], %[[CONSTANT_32]] : index
// CHECK-NEXT:               %[[FROM_EXTENTS_10:.*]] = shape.from_extents %[[MAXSI_1]] : index
// CHECK-NEXT:               scf.yield %[[FROM_EXTENTS_10]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXECUTE_REGION_15:.*]] = scf.execute_region -> !shape.shape {
// CHECK-NEXT:               %[[SHAPE_OF_7:.*]] = shape.shape_of %[[VAL_58]] : tensor<?x?x4096xf16, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:               %[[SHAPE_OF_8:.*]] = shape.shape_of %[[VAL_59]] : tensor<?x3xi64, #hipsr.mem<device>> -> !shape.shape
// CHECK-NEXT:               scf.yield %[[SHAPE_OF_7]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[CONST_SIZE_23:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_20:.*]] = shape.get_extent %[[EXECUTE_REGION_14]], %[[CONST_SIZE_23]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_19:.*]] = shape.size_to_index %[[GET_EXTENT_20]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_15:.*]] = tensor.empty(%[[SIZE_TO_INDEX_19]]) : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONST_SIZE_24:.*]] = shape.const_size 0
// CHECK-NEXT:             %[[GET_EXTENT_21:.*]] = shape.get_extent %[[EXECUTE_REGION_15]], %[[CONST_SIZE_24]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_20:.*]] = shape.size_to_index %[[GET_EXTENT_21]] : !shape.size
// CHECK-NEXT:             %[[CONST_SIZE_25:.*]] = shape.const_size 1
// CHECK-NEXT:             %[[GET_EXTENT_22:.*]] = shape.get_extent %[[EXECUTE_REGION_15]], %[[CONST_SIZE_25]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:             %[[SIZE_TO_INDEX_21:.*]] = shape.size_to_index %[[GET_EXTENT_22]] : !shape.size
// CHECK-NEXT:             %[[EMPTY_16:.*]] = tensor.empty(%[[SIZE_TO_INDEX_20]], %[[SIZE_TO_INDEX_21]]) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[SLICE_0:.*]] = hipsr.slice(%[[VAL_53]]) ins(%[[VAL_56]] : tensor<?xf16, #hipsr.mem<device>>) ends(%[[VAL_57]] : tensor<1xi64, #hipsr.mem<host>>) outs(%[[EMPTY_15]] : tensor<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>} : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[SCATTER_ND_0:.*]] = hipsr.scatter_nd(%[[VAL_53]]) ins(%[[VAL_60]], %[[VAL_61]], %[[SLICE_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) outs(%[[EMPTY_16]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[SCATTER_ND_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:           } -> tensor<?x?x4096xf16, #hipsr.mem<device>> {domain_id = 4 : i64}
// CHECK-NEXT:           return %[[POOL_DOMAIN_4]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:         }

// The table's bytes stay outside the IR, so the resource section prints empty
// once the RUN line's elision drops the blob string.
// CHECK:              {-#
// CHECK-EMPTY:
// CHECK-NEXT:         #-}
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
