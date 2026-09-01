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
// cut, where each barrier sits, and what every shape region computes, so a
// partial check would let those move unnoticed.
//
// Pool domains and what cuts them
// -------------------------------
//
// A domain is one pool allocation, so every buffer inside it must be sized
// before it runs. The pipeline therefore starts a new domain wherever a shape
// depends on a value the host cannot know until the previous domain has
// finished. Each cut below is a barrier placeholder whose shape region reads
// the host tensor named on the arrow.
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
// Only domain 0 opens with an ordinary placeholder, because everything in it is
// shaped by the argument types alone. Domains 1 through 4 each open with a
// barrier placeholder, which is what a cut looks like in the IR below.
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
// downstream will read.
// CHECK-NEXT:           %[[POOL_DOMAIN_0:.*]]:8 = hipsr.pool_domain(%[[ARG0]], %[[ARG2]], %[[ARG1]] : !hipsr.context, tensor<?x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: tensor<?x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = hipsr.constant {value = dense_resource<"file|embedding.onnx.data|0"> : tensor<248320x4096xf16, #hipsr.mem<device>>} : tensor<248320x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant dense<248056> : tensor<i64>
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = hipsr.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = hipsr.constant {value = dense<0> : tensor<1xi64>} : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_0:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<?x4096xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_3:.*]]: !shape.shape):
// CHECK-NEXT:               %[[NUM_ELEMENTS_0:.*]] = shape.num_elements %[[VAL_3]] : !shape.shape -> !shape.size
// CHECK-NEXT:               %[[SIZE_TO_INDEX_0:.*]] = shape.size_to_index %[[NUM_ELEMENTS_0]] : !shape.size
// CHECK-NEXT:               %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIVUI_0:.*]] = arith.divui %[[SIZE_TO_INDEX_0]], %[[CONSTANT_5]] : index
// CHECK-NEXT:               %[[FROM_EXTENTS_0:.*]] = shape.from_extents %[[DIVUI_0]] : index
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_0]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[COMPUTE_0:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<?x4096xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_0]] : tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_4:.*]]: !hipsr.context, %[[VAL_5:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %[[VAL_6:.*]]: tensor<?xf16, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[VAL_5]] {{\[\[}}0, 1]] : tensor<?x4096xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[COLLAPSE_SHAPE_0]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_6:.*]] = hipsr.constant {value = dense<248056> : tensor<i64>} : tensor<i64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_1:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_2]], %[[CONSTANT_6]] : tensor<?x?xi64, #hipsr.mem<device>>, tensor<i64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?xi1, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_7:.*]]: !shape.shape, %[[VAL_8:.*]]: !shape.shape):
// CHECK-NEXT:               %[[BROADCAST_0:.*]] = shape.broadcast %[[VAL_7]], %[[VAL_8]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:               hipsr.shape_yield %[[BROADCAST_0]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EQUAL_0:.*]] = hipsr.equal(%[[VAL_0]]) ins(%[[VAL_2]], %[[CONSTANT_6]] : tensor<?x?xi64, #hipsr.mem<device>>, tensor<i64, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_1]] : tensor<?x?xi1, #hipsr.mem<device>>) : tensor<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_2:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[PLACEHOLDER_1]] : tensor<?x?xi1, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x1xi1, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_9:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONST_SIZE_0:.*]] = shape.const_size 0
// CHECK-NEXT:               %[[GET_EXTENT_0:.*]] = shape.get_extent %[[VAL_9]], %[[CONST_SIZE_0]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[SIZE_TO_INDEX_1:.*]] = shape.size_to_index %[[GET_EXTENT_0]] : !shape.size
// CHECK-NEXT:               %[[CONST_SIZE_1:.*]] = shape.const_size 1
// CHECK-NEXT:               %[[GET_EXTENT_1:.*]] = shape.get_extent %[[VAL_9]], %[[CONST_SIZE_1]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[SIZE_TO_INDEX_2:.*]] = shape.size_to_index %[[GET_EXTENT_1]] : !shape.size
// CHECK-NEXT:               %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_1:.*]] = shape.from_extents %[[SIZE_TO_INDEX_1]], %[[SIZE_TO_INDEX_2]], %[[CONSTANT_9]] : index, index, index
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[COMPUTE_1:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[EQUAL_0]] : tensor<?x?xi1, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_2]] : tensor<?x?x1xi1, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_10:.*]]: !hipsr.context, %[[VAL_11:.*]]: tensor<?x?xi1, #hipsr.mem<device>>, %[[VAL_12:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[CONSTANT_10:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_0:.*]] = tensor.dim %[[VAL_12]], %[[CONSTANT_10]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIM_1:.*]] = tensor.dim %[[VAL_12]], %[[CONSTANT_11]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[VAL_11]] {{\[\[}}0], [1, 2]] output_shape {{\[}}%[[DIM_0]], %[[DIM_1]], 1] : tensor<?x?xi1, #hipsr.mem<device>> into tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXPAND_SHAPE_0]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_3:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[CONSTANT_0]], %[[VAL_2]] : tensor<248320x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_13:.*]]: !shape.shape, %[[VAL_14:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONST_SIZE_2:.*]] = shape.const_size 0
// CHECK-NEXT:               %[[VAL_15:.*]], %[[VAL_16:.*]] = "shape.split_at"(%[[VAL_13]], %[[CONST_SIZE_2]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:               %[[CONST_SIZE_3:.*]] = shape.const_size 1
// CHECK-NEXT:               %[[VAL_17:.*]], %[[VAL_18:.*]] = "shape.split_at"(%[[VAL_13]], %[[CONST_SIZE_3]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT:               %[[CONCAT_0:.*]] = shape.concat %[[VAL_15]], %[[VAL_14]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:               %[[CONCAT_1:.*]] = shape.concat %[[CONCAT_0]], %[[VAL_18]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT:               hipsr.shape_yield %[[CONCAT_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[GATHER_0:.*]] = hipsr.gather(%[[VAL_0]]) ins(%[[CONSTANT_0]], %[[VAL_2]] : tensor<248320x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_3]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64} : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_4:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[PLACEHOLDER_3]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_19:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONSTANT_12:.*]] = arith.constant 3 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_2:.*]] = shape.from_extents %[[CONSTANT_12]] : index
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_2]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[COMPUTE_2:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[GATHER_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_4]] : tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_20:.*]]: !hipsr.context, %[[VAL_21:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_22:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_13:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_2:.*]] = tensor.dim %[[VAL_21]], %[[CONSTANT_13]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK-NEXT:               %[[CONSTANT_14:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIM_3:.*]] = tensor.dim %[[VAL_21]], %[[CONSTANT_14]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK-NEXT:               %[[CONSTANT_15:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:               %[[FROM_ELEMENTS_0:.*]] = tensor.from_elements %[[INDEX_CAST_0]], %[[INDEX_CAST_1]], %[[CONSTANT_15]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_0]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[PLACEHOLDER_0]], %[[COMPUTE_0]], %[[PLACEHOLDER_2]], %[[COMPUTE_1]], %[[PLACEHOLDER_3]], %[[GATHER_0]], %[[PLACEHOLDER_4]], %[[COMPUTE_2]] : tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<3xi64, #hipsr.mem<host>> {domain_id = 0 : i64}

// Each broadcast reads its extents from a host shape vector rather than from
// the type, so each one opens a domain behind a barrier placeholder.
// CHECK-NEXT:           %[[POOL_DOMAIN_1:.*]]:2 = hipsr.pool_domain(%[[ARG0]], %[[VAL_23:.*]]#2, %[[VAL_23]]#6, %[[VAL_23]]#3, %[[VAL_23]]#7 : !hipsr.context, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_24:.*]]: !hipsr.context, %[[VAL_25:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_26:.*]]: tensor<3xi64, #hipsr.mem<host>>, %[[VAL_27:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_28:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:             %[[PLACEHOLDER_5:.*]] = hipsr.placeholder(%[[VAL_24]]) ins(%[[VAL_25]], %[[VAL_26]] : tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x?x?xi1, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_29:.*]]: !hipsr.context, %[[VAL_30:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_31:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[SHAPE_OF_0:.*]] = shape.shape_of %[[VAL_30]] : tensor<?x?x1xi1, #hipsr.mem<device>> -> tensor<3xindex>
// CHECK-NEXT:               %[[CONSTANT_16:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_0:.*]] = tensor.extract %[[VAL_31]]{{\[}}%[[CONSTANT_16]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_2:.*]] = arith.index_cast %[[EXTRACT_0]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_17:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[EXTRACT_1:.*]] = tensor.extract %[[VAL_31]]{{\[}}%[[CONSTANT_17]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_3:.*]] = arith.index_cast %[[EXTRACT_1]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_18:.*]] = arith.constant 2 : index
// CHECK-NEXT:               %[[EXTRACT_2:.*]] = tensor.extract %[[VAL_31]]{{\[}}%[[CONSTANT_18]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_4:.*]] = arith.index_cast %[[EXTRACT_2]] : i64 to index
// CHECK-NEXT:               %[[FROM_EXTENTS_3:.*]] = shape.from_extents %[[INDEX_CAST_2]], %[[INDEX_CAST_3]], %[[INDEX_CAST_4]] : index, index, index
// CHECK-NEXT:               %[[CSTR_BROADCASTABLE_0:.*]] = shape.cstr_broadcastable %[[SHAPE_OF_0]], %[[FROM_EXTENTS_3]] : tensor<3xindex>, !shape.shape
// CHECK-NEXT:               %[[ASSUMING_0:.*]] = shape.assuming %[[CSTR_BROADCASTABLE_0]] -> (!shape.shape) {
// CHECK-NEXT:                 %[[BROADCAST_1:.*]] = shape.broadcast %[[SHAPE_OF_0]], %[[FROM_EXTENTS_3]] : tensor<3xindex>, !shape.shape -> !shape.shape
// CHECK-NEXT:                 shape.assuming_yield %[[BROADCAST_1]] : !shape.shape
// CHECK-NEXT:               }
// CHECK-NEXT:               hipsr.shape_yield %[[ASSUMING_0]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXPAND_0:.*]] = hipsr.expand(%[[VAL_24]]) ins(%[[VAL_27]], %[[VAL_28]] : tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) outs(%[[PLACEHOLDER_5]] : tensor<?x?x?xi1, #hipsr.mem<device>>) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[PLACEHOLDER_5]], %[[EXPAND_0]] : tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:           } -> tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<?x?x?xi1, #hipsr.mem<device>> {domain_id = 1 : i64}

// The second broadcast feeds the search. Its destination holds one row per
// input axis and, in the worst case, a column per input element; the count of
// the columns it filled sits beside it and is read back to the host.
// CHECK-NEXT:           %[[POOL_DOMAIN_2:.*]]:4 = hipsr.pool_domain(%[[ARG0]], %[[VAL_32:.*]]#0, %[[VAL_33:.*]]#6, %[[VAL_32]]#1, %[[VAL_33]]#7 : !hipsr.context, tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_34:.*]]: !hipsr.context, %[[VAL_35:.*]]: tensor<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_36:.*]]: tensor<3xi64, #hipsr.mem<host>>, %[[VAL_37:.*]]: tensor<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_38:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:             %[[PLACEHOLDER_6:.*]] = hipsr.placeholder(%[[VAL_34]]) ins(%[[VAL_35]], %[[VAL_36]] : tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x?x?xi1, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_39:.*]]: !hipsr.context, %[[VAL_40:.*]]: tensor<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_41:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[SHAPE_OF_1:.*]] = shape.shape_of %[[VAL_40]] : tensor<?x?x?xi1, #hipsr.mem<device>> -> tensor<3xindex>
// CHECK-NEXT:               %[[CONSTANT_19:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_3:.*]] = tensor.extract %[[VAL_41]]{{\[}}%[[CONSTANT_19]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_5:.*]] = arith.index_cast %[[EXTRACT_3]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_20:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[EXTRACT_4:.*]] = tensor.extract %[[VAL_41]]{{\[}}%[[CONSTANT_20]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_6:.*]] = arith.index_cast %[[EXTRACT_4]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_21:.*]] = arith.constant 2 : index
// CHECK-NEXT:               %[[EXTRACT_5:.*]] = tensor.extract %[[VAL_41]]{{\[}}%[[CONSTANT_21]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_7:.*]] = arith.index_cast %[[EXTRACT_5]] : i64 to index
// CHECK-NEXT:               %[[FROM_EXTENTS_4:.*]] = shape.from_extents %[[INDEX_CAST_5]], %[[INDEX_CAST_6]], %[[INDEX_CAST_7]] : index, index, index
// CHECK-NEXT:               %[[CSTR_BROADCASTABLE_1:.*]] = shape.cstr_broadcastable %[[SHAPE_OF_1]], %[[FROM_EXTENTS_4]] : tensor<3xindex>, !shape.shape
// CHECK-NEXT:               %[[ASSUMING_1:.*]] = shape.assuming %[[CSTR_BROADCASTABLE_1]] -> (!shape.shape) {
// CHECK-NEXT:                 %[[BROADCAST_2:.*]] = shape.broadcast %[[SHAPE_OF_1]], %[[FROM_EXTENTS_4]] : tensor<3xindex>, !shape.shape -> !shape.shape
// CHECK-NEXT:                 shape.assuming_yield %[[BROADCAST_2]] : !shape.shape
// CHECK-NEXT:               }
// CHECK-NEXT:               hipsr.shape_yield %[[ASSUMING_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[EXPAND_1:.*]] = hipsr.expand(%[[VAL_34]]) ins(%[[VAL_37]], %[[VAL_38]] : tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) outs(%[[PLACEHOLDER_6]] : tensor<?x?x?xi1, #hipsr.mem<device>>) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_7:.*]]:2 = hipsr.placeholder(%[[VAL_34]]) ins(%[[PLACEHOLDER_6]] : tensor<?x?x?xi1, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_42:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONST_SIZE_4:.*]] = shape.const_size 3
// CHECK-NEXT:               %[[NUM_ELEMENTS_1:.*]] = shape.num_elements %[[VAL_42]] : !shape.shape -> !shape.size
// CHECK-NEXT:               %[[FROM_EXTENTS_5:.*]] = shape.from_extents %[[CONST_SIZE_4]], %[[NUM_ELEMENTS_1]] : !shape.size, !shape.size
// CHECK-NEXT:               %[[CONST_SHAPE_0:.*]] = shape.const_shape [1] : !shape.shape
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_5]], %[[CONST_SHAPE_0]] : !shape.shape, !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[NONZERO_0:.*]]:2 = hipsr.nonzero(%[[VAL_34]]) ins(%[[EXPAND_1]] : tensor<?x?x?xi1, #hipsr.mem<device>>) outs(%[[VAL_43:.*]]#0, %[[VAL_43]]#1 : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_8:.*]] = hipsr.placeholder(%[[VAL_34]]) ins(%[[VAL_43]]#1 : tensor<1xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_44:.*]]: !shape.shape):
// CHECK-NEXT:               hipsr.shape_yield %[[VAL_44]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[VAL_45:.*]] = hipsr.copy_d2h(%[[VAL_34]]) ins(%[[NONZERO_0]]#1 : tensor<1xi64, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_8]] : tensor<1xi64, #hipsr.mem<host>>) : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[VAL_43]]#0, %[[NONZERO_0]]#0, %[[PLACEHOLDER_8]], %[[VAL_45]] : tensor<3x?xi64, #hipsr.mem<device>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<3x?xi64, #hipsr.mem<device>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>> {domain_id = 2 : i64}

// That host count opens the next domain. A barrier turns it into the trailing
// extent so the search destination narrows to the columns that hold a position.
// The barrier and the compute list the same two values, so they share a domain
// and each reads only the one it needs.
// CHECK-NEXT:           %[[POOL_DOMAIN_3:.*]]:4 = hipsr.pool_domain(%[[ARG0]], %[[VAL_46:.*]]#2, %[[VAL_46]]#0, %[[VAL_46]]#3, %[[VAL_46]]#1 : !hipsr.context, tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_47:.*]]: !hipsr.context, %[[VAL_48:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_49:.*]]: tensor<3x?xi64, #hipsr.mem<device>>, %[[VAL_50:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_51:.*]]: tensor<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[PLACEHOLDER_9:.*]] = hipsr.placeholder(%[[VAL_47]]) ins(%[[VAL_48]], %[[VAL_49]] : tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<3x?xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_52:.*]]: !hipsr.context, %[[VAL_53:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_54:.*]]: tensor<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[CONSTANT_22:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_6:.*]] = tensor.extract %[[VAL_53]]{{\[}}%[[CONSTANT_22]]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[INDEX_CAST_8:.*]] = arith.index_cast %[[EXTRACT_6]] : i64 to index
// CHECK-NEXT:               %[[CONSTANT_23:.*]] = arith.constant 3 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_6:.*]] = shape.from_extents %[[CONSTANT_23]], %[[INDEX_CAST_8]] : index, index
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_6]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[COMPUTE_3:.*]] = hipsr.compute(%[[VAL_47]]) ins(%[[VAL_50]], %[[VAL_51]] : tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_9]] : tensor<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_55:.*]]: !hipsr.context, %[[VAL_56:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_57:.*]]: tensor<3x?xi64, #hipsr.mem<device>>, %[[VAL_58:.*]]: tensor<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[CONSTANT_24:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIM_4:.*]] = tensor.dim %[[VAL_58]], %[[CONSTANT_24]] : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[VAL_57]][0, 0] [3, %[[DIM_4]]] [1, 1] : tensor<3x?xi64, #hipsr.mem<device>> to tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXTRACT_SLICE_0]] : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_10:.*]] = hipsr.placeholder(%[[VAL_47]]) ins(%[[PLACEHOLDER_9]] : tensor<3x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x3xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_59:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONST_SIZE_5:.*]] = shape.const_size 1
// CHECK-NEXT:               %[[GET_EXTENT_2:.*]] = shape.get_extent %[[VAL_59]], %[[CONST_SIZE_5]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[CONST_SIZE_6:.*]] = shape.const_size 0
// CHECK-NEXT:               %[[GET_EXTENT_3:.*]] = shape.get_extent %[[VAL_59]], %[[CONST_SIZE_6]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:               %[[FROM_EXTENTS_7:.*]] = shape.from_extents %[[GET_EXTENT_2]], %[[GET_EXTENT_3]] : !shape.size, !shape.size
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_7]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[TRANSPOSE_0:.*]] = hipsr.transpose(%[[VAL_47]]) ins(%[[COMPUTE_3]] : tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_10]] : tensor<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>} : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_11:.*]] = hipsr.placeholder(%[[VAL_47]]) ins(%[[PLACEHOLDER_10]] : tensor<?x3xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_60:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONSTANT_25:.*]] = arith.constant 2 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_8:.*]] = shape.from_extents %[[CONSTANT_25]] : index
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_8]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[COMPUTE_4:.*]] = hipsr.compute(%[[VAL_47]]) ins(%[[TRANSPOSE_0]] : tensor<?x3xi64, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_11]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_61:.*]]: !hipsr.context, %[[VAL_62:.*]]: tensor<?x3xi64, #hipsr.mem<device>>, %[[VAL_63:.*]]: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_26:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_5:.*]] = tensor.dim %[[VAL_62]], %[[CONSTANT_26]] : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_9:.*]] = arith.index_cast %[[DIM_5]] : index to i64
// CHECK-NEXT:               %[[CONSTANT_27:.*]] = arith.constant 3 : i64
// CHECK-NEXT:               %[[FROM_ELEMENTS_1:.*]] = tensor.from_elements %[[INDEX_CAST_9]], %[[CONSTANT_27]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_1]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[PLACEHOLDER_12:.*]] = hipsr.placeholder(%[[VAL_47]]) ins(%[[PLACEHOLDER_11]] : tensor<2xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<i64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_64:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONST_SHAPE_1:.*]] = shape.const_shape [] : !shape.shape
// CHECK-NEXT:               hipsr.shape_yield %[[CONST_SHAPE_1]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[COMPUTE_5:.*]] = hipsr.compute(%[[VAL_47]]) ins(%[[COMPUTE_4]] : tensor<2xi64, #hipsr.mem<host>>) outs(%[[PLACEHOLDER_12]] : tensor<i64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_65:.*]]: !hipsr.context, %[[VAL_66:.*]]: tensor<2xi64, #hipsr.mem<host>>, %[[VAL_67:.*]]: tensor<i64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_28:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_7:.*]] = tensor.extract %[[VAL_66]]{{\[}}%[[CONSTANT_28]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[FROM_ELEMENTS_2:.*]] = tensor.from_elements %[[EXTRACT_7]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_2]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[PLACEHOLDER_13:.*]] = hipsr.placeholder(%[[VAL_47]]) ins(%[[PLACEHOLDER_12]] : tensor<i64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_68:.*]]: !shape.shape):
// CHECK-NEXT:               %[[CONSTANT_29:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[FROM_EXTENTS_9:.*]] = shape.from_extents %[[CONSTANT_29]] : index
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_9]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[COMPUTE_6:.*]] = hipsr.compute(%[[VAL_47]]) ins(%[[COMPUTE_5]] : tensor<i64, #hipsr.mem<host>>) outs(%[[PLACEHOLDER_13]] : tensor<1xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_69:.*]]: !hipsr.context, %[[VAL_70:.*]]: tensor<i64, #hipsr.mem<host>>, %[[VAL_71:.*]]: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[VAL_70]] [] output_shape [1] : tensor<i64, #hipsr.mem<host>> into tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXPAND_SHAPE_1]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[PLACEHOLDER_10]], %[[TRANSPOSE_0]], %[[PLACEHOLDER_13]], %[[COMPUTE_6]] : tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>> {domain_id = 3 : i64}

// The update window ends at that same count, and the scatter writes it into
// the embeddings.
// CHECK-NEXT:           %[[POOL_DOMAIN_4:.*]] = hipsr.pool_domain(%[[ARG0]], %[[VAL_72:.*]]#0, %[[VAL_73:.*]]#2, %[[VAL_72]]#1, %[[VAL_73]]#3, %[[VAL_72]]#4, %[[VAL_73]]#0, %[[VAL_72]]#5, %[[VAL_73]]#1 : !hipsr.context, tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_74:.*]]: !hipsr.context, %[[VAL_75:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[VAL_76:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_77:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[VAL_78:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_79:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_80:.*]]: tensor<?x3xi64, #hipsr.mem<device>>, %[[VAL_81:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_82:.*]]: tensor<?x3xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[PLACEHOLDER_14:.*]] = hipsr.placeholder(%[[VAL_74]]) ins(%[[VAL_75]], %[[VAL_76]] : tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_83:.*]]: !hipsr.context, %[[VAL_84:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[VAL_85:.*]]: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_30:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_6:.*]] = tensor.dim %[[VAL_84]], %[[CONSTANT_30]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[CONSTANT_31:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_8:.*]] = tensor.extract %[[VAL_85]]{{\[}}%[[CONSTANT_31]]] : tensor<1xi64, #hipsr.mem<host>>
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
// CHECK-NEXT:               hipsr.shape_yield %[[FROM_EXTENTS_10]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[SLICE_0:.*]] = hipsr.slice(%[[VAL_74]]) ins(%[[VAL_77]] : tensor<?xf16, #hipsr.mem<device>>) ends(%[[VAL_78]] : tensor<1xi64, #hipsr.mem<host>>) outs(%[[PLACEHOLDER_14]] : tensor<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>} : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[PLACEHOLDER_15:.*]] = hipsr.placeholder(%[[VAL_74]]) ins(%[[VAL_79]], %[[VAL_80]], %[[PLACEHOLDER_14]] : tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:             ^bb0(%[[VAL_86:.*]]: !shape.shape, %[[VAL_87:.*]]: !shape.shape, %[[VAL_88:.*]]: !shape.shape):
// CHECK-NEXT:               hipsr.shape_yield %[[VAL_86]] : !shape.shape
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[SCATTER_ND_0:.*]] = hipsr.scatter_nd(%[[VAL_74]]) ins(%[[VAL_81]], %[[VAL_82]], %[[SLICE_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_15]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) : tensor<?x?x4096xf16, #hipsr.mem<device>>
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
