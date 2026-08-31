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
// The zones of a domain
// ---------------------
//
// hipsr-materialize-init-tensors leaves each domain body in this order:
//
//   1. the constants;
//   2. the shape computations;
//   3. the tensor.empty allocations that read their dynamic extents from those
//      shapes;
//   4. the data ops;
//   5. one hipsr.preserve_shape per result, tying its shape to the value that
//      filled the buffer it sized.
//
// A shape computation starts as an scf.execute_region yielding one extent
// tensor per result. What reaches the checks below:
//
//   - upstream lowers the region to arith, scf and tensor arithmetic;
//   - canonicalization inlines it, leaving straight-line code;
//   - everything constant-like is hoisted to the top of the domain;
//   - preserve_shape names every result shape, so a shape stays alive even
//     when no allocation reads its extents;
//   - nothing of the shape dialect survives, which checking every line of the
//     output shows: such an op would have to appear on a line of its own.
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../Inputs/make_external_data.py %t/embedding.onnx.data 2034237440 && cd %t && hip-mlir-opt --onnx-dialect=modeled --hipsr-pipeline --mlir-elide-resource-strings-if-larger=32 %s | FileCheck %s

module {


// The context becomes argument 0 and every symbolic extent survives.
// CHECK-LABEL:   func.func @main_graph(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<?x?xi64, #hipsr.mem<device>> {onnx.name = "input_ids"},
// CHECK-SAME:      %[[ARG2:.*]]: tensor<?x4096xf16, #hipsr.mem<device>> {onnx.name = "image_features"}) -> (tensor<?x?x4096xf16, #hipsr.mem<device>> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {

// Domain 0 takes everything shaped by the graph inputs alone: the embedding
// lookup, the bool mask, and the shape vector the broadcasts read.
//
//   - Constant-like values lead: index constants, static shapes, weights.
//   - The flat reshape's extent is a product over the input shape, so an
//     scf.for opens the shape graph.
// CHECK-NEXT:           %[[POOL_DOMAIN_0:.*]]:8 = hipsr.pool_domain(%[[ARG0]], %[[ARG2]], %[[ARG1]] : !hipsr.context, tensor<?x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %[[VAL_2:.*]]: tensor<?x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_0:.*]] = arith.constant dense<> : tensor<0xindex>
// CHECK-NEXT:             %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:             %[[CONSTANT_2:.*]] = arith.constant 4096 : index
// CHECK-NEXT:             %[[CONSTANT_3:.*]] = arith.constant dense<3> : tensor<1xindex>
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_6:.*]] = hipsr.constant {value = dense_resource<"file|embedding.onnx.data|0"> : tensor<248320x4096xf16, #hipsr.mem<device>>} : tensor<248320x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[CONSTANT_7:.*]] = hipsr.constant {value = dense<248056> : tensor<i64>} : tensor<i64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_0:.*]] = tensor.dim %[[VAL_1]], %[[CONSTANT_4]] : tensor<?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[FROM_ELEMENTS_0:.*]] = tensor.from_elements %[[DIM_0]], %[[CONSTANT_2]] : tensor<2xindex>
// CHECK-NEXT:             %[[FOR_0:.*]] = scf.for %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_1]] step %[[CONSTANT_5]] iter_args(%[[VAL_4:.*]] = %[[CONSTANT_5]]) -> (index) {
// CHECK-NEXT:               %[[EXTRACT_0:.*]] = tensor.extract %[[FROM_ELEMENTS_0]]{{\[}}%[[VAL_3]]] : tensor<2xindex>
// CHECK-NEXT:               %[[MULI_0:.*]] = arith.muli %[[EXTRACT_0]], %[[VAL_4]] : index
// CHECK-NEXT:               scf.yield %[[MULI_0]] : index
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[FROM_ELEMENTS_1:.*]] = tensor.from_elements %[[FOR_0]] : tensor<1xindex>
// CHECK-NEXT:             %[[DIM_1:.*]] = tensor.dim %[[VAL_2]], %[[CONSTANT_4]] : tensor<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_2:.*]] = tensor.dim %[[VAL_2]], %[[CONSTANT_5]] : tensor<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[FROM_ELEMENTS_2:.*]] = tensor.from_elements %[[DIM_1]], %[[DIM_2]] : tensor<2xindex>
// CHECK-NEXT:             %[[GENERATE_0:.*]] = tensor.generate  {
// CHECK-NEXT:             ^bb0(%[[VAL_5:.*]]: index):
// CHECK-NEXT:               %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_5]], %[[CONSTANT_4]] : index
// CHECK-NEXT:               %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[CONSTANT_5]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[EXTRACT_1:.*]] = tensor.extract %[[FROM_ELEMENTS_2]]{{\[}}%[[VAL_5]]] : tensor<2xindex>
// CHECK-NEXT:                 scf.yield %[[EXTRACT_1]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_5]], %[[CONSTANT_1]] : index
// CHECK-NEXT:               %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[IF_0]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[SUBI_0:.*]] = arith.subi %[[VAL_5]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 %[[EXTRACT_2:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[SUBI_0]]] : tensor<0xindex>
// CHECK-NEXT:                 %[[CMPI_2:.*]] = arith.cmpi eq, %[[EXTRACT_2]], %[[CONSTANT_5]] : index
// CHECK-NEXT:                 %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[IF_0]], %[[EXTRACT_2]] : index
// CHECK-NEXT:                 scf.yield %[[SELECT_0]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               tensor.yield %[[IF_1]] : index
// CHECK-NEXT:             } : tensor<2xindex>
// CHECK-NEXT:             %[[FROM_ELEMENTS_3:.*]] = tensor.from_elements %[[DIM_1]], %[[DIM_2]], %[[CONSTANT_5]] : tensor<3xindex>
// CHECK-NEXT:             %[[DIM_3:.*]] = tensor.dim %[[VAL_2]], %[[CONSTANT_4]] : tensor<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_4:.*]] = tensor.dim %[[VAL_2]], %[[CONSTANT_5]] : tensor<?x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[FROM_ELEMENTS_4:.*]] = tensor.from_elements %[[DIM_3]], %[[DIM_4]], %[[CONSTANT_2]] : tensor<3xindex>

// The allocation zone. Each tensor.empty takes its dynamic extents from the
// shape graph above; a fully static result type needs none. Assembling an
// extent tensor and reading one back out fold, so each allocation names it.
// CHECK-NEXT:             %[[EMPTY_0:.*]] = tensor.empty(%[[FOR_0]]) : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_1:.*]] = tensor.empty(%[[DIM_1]], %[[DIM_2]]) : tensor<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_2:.*]] = tensor.empty(%[[DIM_1]], %[[DIM_2]]) : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_3:.*]] = tensor.empty(%[[DIM_3]], %[[DIM_4]]) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_4:.*]] = tensor.empty() : tensor<3xi64, #hipsr.mem<host>>

// The data zone, in conversion order, each op now initialized by a tensor.empty.
// CHECK-NEXT:             %[[COMPUTE_0:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<?x4096xf16, #hipsr.mem<device>>) outs(%[[EMPTY_0]] : tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_6:.*]]: !hipsr.context, %[[VAL_7:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>, %[[VAL_8:.*]]: tensor<?xf16, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[VAL_7]] {{\[\[}}0, 1]] : tensor<?x4096xf16, #hipsr.mem<device>> into tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[COLLAPSE_SHAPE_0]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EQUAL_0:.*]] = hipsr.equal(%[[VAL_0]]) ins(%[[VAL_2]], %[[CONSTANT_7]] : tensor<?x?xi64, #hipsr.mem<device>>, tensor<i64, #hipsr.mem<device>>) outs(%[[EMPTY_1]] : tensor<?x?xi1, #hipsr.mem<device>>) : tensor<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[COMPUTE_1:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[EQUAL_0]] : tensor<?x?xi1, #hipsr.mem<device>>) outs(%[[EMPTY_2]] : tensor<?x?x1xi1, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_9:.*]]: !hipsr.context, %[[VAL_10:.*]]: tensor<?x?xi1, #hipsr.mem<device>>, %[[VAL_11:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_5:.*]] = tensor.dim %[[VAL_11]], %[[CONSTANT_9]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               %[[DIM_6:.*]] = tensor.dim %[[VAL_11]], %[[CONSTANT_8]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[VAL_10]] {{\[\[}}0], [1, 2]] output_shape [%[[DIM_5]], %[[DIM_6]], 1] : tensor<?x?xi1, #hipsr.mem<device>> into tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXPAND_SHAPE_0]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[GATHER_0:.*]] = hipsr.gather(%[[VAL_0]]) ins(%[[CONSTANT_6]], %[[VAL_2]] : tensor<248320x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) outs(%[[EMPTY_3]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64} : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[COMPUTE_2:.*]] = hipsr.compute(%[[VAL_0]]) ins(%[[GATHER_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) outs(%[[EMPTY_4]] : tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_12:.*]]: !hipsr.context, %[[VAL_13:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_14:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_10:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:               %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_7:.*]] = tensor.dim %[[VAL_13]], %[[CONSTANT_12]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIM_7]] : index to i64
// CHECK-NEXT:               %[[DIM_8:.*]] = tensor.dim %[[VAL_13]], %[[CONSTANT_11]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[DIM_8]] : index to i64
// CHECK-NEXT:               %[[FROM_ELEMENTS_5:.*]] = tensor.from_elements %[[INDEX_CAST_0]], %[[INDEX_CAST_1]], %[[CONSTANT_10]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_5]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<3xi64, #hipsr.mem<host>>

// The preserve_shape zone closes the domain. Each one names the shape that
// sized the buffer its value filled. For a shape no allocation reads, this is
// the only use left, and it is what keeps the shape alive.
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_1]], %[[COMPUTE_0]] : tensor<1xindex>, tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[GENERATE_0]], %[[EQUAL_0]] : tensor<2xindex>, tensor<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_3]], %[[COMPUTE_1]] : tensor<3xindex>, tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_4]], %[[GATHER_0]] : tensor<3xindex>, tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[CONSTANT_3]], %[[COMPUTE_2]] : tensor<1xindex>, tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_0]], %[[COMPUTE_0]], %[[EMPTY_2]], %[[COMPUTE_1]], %[[EMPTY_3]], %[[GATHER_0]], %[[EMPTY_4]], %[[COMPUTE_2]] : tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<?xf16, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<3xi64, #hipsr.mem<host>> {domain_id = 0 : i64}

// Each broadcast reads its extents from a host shape vector, not from the type,
// so each one opens a domain. Its shape graph extracts them with tensor.extract
// and rebuilds the destination shape, which folds to one select per axis.
// CHECK-NEXT:           %[[POOL_DOMAIN_1:.*]]:2 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]]#2, %[[POOL_DOMAIN_0]]#6, %[[POOL_DOMAIN_0]]#3, %[[POOL_DOMAIN_0]]#7 : !hipsr.context, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_15:.*]]: !hipsr.context, %[[VAL_16:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_17:.*]]: tensor<3xi64, #hipsr.mem<host>>, %[[VAL_18:.*]]: tensor<?x?x1xi1, #hipsr.mem<device>>, %[[VAL_19:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:             %[[CONSTANT_13:.*]] = arith.constant 2 : index
// CHECK-NEXT:             %[[CONSTANT_14:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_15:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[DIM_9:.*]] = tensor.dim %[[VAL_16]], %[[CONSTANT_15]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_10:.*]] = tensor.dim %[[VAL_16]], %[[CONSTANT_14]] : tensor<?x?x1xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[FROM_ELEMENTS_6:.*]] = tensor.from_elements %[[DIM_9]], %[[DIM_10]], %[[CONSTANT_14]] : tensor<3xindex>
// CHECK-NEXT:             %[[EXTRACT_3:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[CONSTANT_15]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[EXTRACT_3]] : i64 to index
// CHECK-NEXT:             %[[EXTRACT_4:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[CONSTANT_14]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_3:.*]] = arith.index_cast %[[EXTRACT_4]] : i64 to index
// CHECK-NEXT:             %[[EXTRACT_5:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[CONSTANT_13]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_4:.*]] = arith.index_cast %[[EXTRACT_5]] : i64 to index
// CHECK-NEXT:             %[[FROM_ELEMENTS_7:.*]] = tensor.from_elements %[[INDEX_CAST_2]], %[[INDEX_CAST_3]], %[[INDEX_CAST_4]] : tensor<3xindex>
// CHECK-NEXT:             %[[GENERATE_1:.*]] = tensor.generate  {
// CHECK-NEXT:             ^bb0(%[[VAL_20:.*]]: index):
// CHECK-NEXT:               %[[CMPI_3:.*]] = arith.cmpi ult, %[[VAL_20]], %[[CONSTANT_15]] : index
// CHECK-NEXT:               %[[IF_2:.*]] = scf.if %[[CMPI_3]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[CONSTANT_14]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[EXTRACT_6:.*]] = tensor.extract %[[FROM_ELEMENTS_6]]{{\[}}%[[VAL_20]]] : tensor<3xindex>
// CHECK-NEXT:                 scf.yield %[[EXTRACT_6]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               %[[CMPI_4:.*]] = arith.cmpi ult, %[[VAL_20]], %[[CONSTANT_15]] : index
// CHECK-NEXT:               %[[IF_3:.*]] = scf.if %[[CMPI_4]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[IF_2]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[EXTRACT_7:.*]] = tensor.extract %[[FROM_ELEMENTS_7]]{{\[}}%[[VAL_20]]] : tensor<3xindex>
// CHECK-NEXT:                 %[[CMPI_5:.*]] = arith.cmpi eq, %[[EXTRACT_7]], %[[CONSTANT_14]] : index
// CHECK-NEXT:                 %[[SELECT_1:.*]] = arith.select %[[CMPI_5]], %[[IF_2]], %[[EXTRACT_7]] : index
// CHECK-NEXT:                 scf.yield %[[SELECT_1]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               tensor.yield %[[IF_3]] : index
// CHECK-NEXT:             } : tensor<3xindex>
// CHECK-NEXT:             %[[CMPI_6:.*]] = arith.cmpi eq, %[[INDEX_CAST_2]], %[[CONSTANT_14]] : index
// CHECK-NEXT:             %[[SELECT_2:.*]] = arith.select %[[CMPI_6]], %[[DIM_9]], %[[INDEX_CAST_2]] : index
// CHECK-NEXT:             %[[CMPI_7:.*]] = arith.cmpi eq, %[[INDEX_CAST_3]], %[[CONSTANT_14]] : index
// CHECK-NEXT:             %[[SELECT_3:.*]] = arith.select %[[CMPI_7]], %[[DIM_10]], %[[INDEX_CAST_3]] : index
// CHECK-NEXT:             %[[EMPTY_5:.*]] = tensor.empty(%[[SELECT_2]], %[[SELECT_3]], %[[INDEX_CAST_4]]) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EXPAND_0:.*]] = hipsr.expand(%[[VAL_15]]) ins(%[[VAL_18]], %[[VAL_19]] : tensor<?x?x1xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) outs(%[[EMPTY_5]] : tensor<?x?x?xi1, #hipsr.mem<device>>) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[GENERATE_1]], %[[EXPAND_0]] : tensor<3xindex>, tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_5]], %[[EXPAND_0]] : tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:           } -> tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<?x?x?xi1, #hipsr.mem<device>> {domain_id = 1 : i64}

// The second broadcast feeds the search:
//
//   - its destination holds one row per input axis, one column per element;
//   - the count of filled columns sits beside it and travels to the host;
//   - that count is a product over the destination, so the scf.for from
//     shape.num_elements survives.
// CHECK-NEXT:           %[[POOL_DOMAIN_2:.*]]:4 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_1]]#0, %[[POOL_DOMAIN_0]]#6, %[[POOL_DOMAIN_1]]#1, %[[POOL_DOMAIN_0]]#7 : !hipsr.context, tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>, tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_21:.*]]: !hipsr.context, %[[VAL_22:.*]]: tensor<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_23:.*]]: tensor<3xi64, #hipsr.mem<host>>, %[[VAL_24:.*]]: tensor<?x?x?xi1, #hipsr.mem<device>>, %[[VAL_25:.*]]: tensor<3xi64, #hipsr.mem<host>>):
// CHECK-NEXT:             %[[CONSTANT_16:.*]] = arith.constant dense<1> : tensor<1xindex>
// CHECK-NEXT:             %[[CONSTANT_17:.*]] = arith.constant 3 : index
// CHECK-NEXT:             %[[CONSTANT_18:.*]] = arith.constant 2 : index
// CHECK-NEXT:             %[[CONSTANT_19:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_20:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[DIM_11:.*]] = tensor.dim %[[VAL_22]], %[[CONSTANT_20]] : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_12:.*]] = tensor.dim %[[VAL_22]], %[[CONSTANT_19]] : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_13:.*]] = tensor.dim %[[VAL_22]], %[[CONSTANT_18]] : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[FROM_ELEMENTS_8:.*]] = tensor.from_elements %[[DIM_11]], %[[DIM_12]], %[[DIM_13]] : tensor<3xindex>
// CHECK-NEXT:             %[[EXTRACT_8:.*]] = tensor.extract %[[VAL_23]]{{\[}}%[[CONSTANT_20]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_5:.*]] = arith.index_cast %[[EXTRACT_8]] : i64 to index
// CHECK-NEXT:             %[[EXTRACT_9:.*]] = tensor.extract %[[VAL_23]]{{\[}}%[[CONSTANT_19]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_6:.*]] = arith.index_cast %[[EXTRACT_9]] : i64 to index
// CHECK-NEXT:             %[[EXTRACT_10:.*]] = tensor.extract %[[VAL_23]]{{\[}}%[[CONSTANT_18]]] : tensor<3xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_7:.*]] = arith.index_cast %[[EXTRACT_10]] : i64 to index
// CHECK-NEXT:             %[[FROM_ELEMENTS_9:.*]] = tensor.from_elements %[[INDEX_CAST_5]], %[[INDEX_CAST_6]], %[[INDEX_CAST_7]] : tensor<3xindex>
// CHECK-NEXT:             %[[GENERATE_2:.*]] = tensor.generate  {
// CHECK-NEXT:             ^bb0(%[[VAL_26:.*]]: index):
// CHECK-NEXT:               %[[CMPI_8:.*]] = arith.cmpi ult, %[[VAL_26]], %[[CONSTANT_20]] : index
// CHECK-NEXT:               %[[IF_4:.*]] = scf.if %[[CMPI_8]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[CONSTANT_19]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[EXTRACT_11:.*]] = tensor.extract %[[FROM_ELEMENTS_8]]{{\[}}%[[VAL_26]]] : tensor<3xindex>
// CHECK-NEXT:                 scf.yield %[[EXTRACT_11]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               %[[CMPI_9:.*]] = arith.cmpi ult, %[[VAL_26]], %[[CONSTANT_20]] : index
// CHECK-NEXT:               %[[IF_5:.*]] = scf.if %[[CMPI_9]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[IF_4]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[EXTRACT_12:.*]] = tensor.extract %[[FROM_ELEMENTS_9]]{{\[}}%[[VAL_26]]] : tensor<3xindex>
// CHECK-NEXT:                 %[[CMPI_10:.*]] = arith.cmpi eq, %[[EXTRACT_12]], %[[CONSTANT_19]] : index
// CHECK-NEXT:                 %[[SELECT_4:.*]] = arith.select %[[CMPI_10]], %[[IF_4]], %[[EXTRACT_12]] : index
// CHECK-NEXT:                 scf.yield %[[SELECT_4]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               tensor.yield %[[IF_5]] : index
// CHECK-NEXT:             } : tensor<3xindex>
// CHECK-NEXT:             %[[FOR_1:.*]] = scf.for %[[VAL_27:.*]] = %[[CONSTANT_20]] to %[[CONSTANT_17]] step %[[CONSTANT_19]] iter_args(%[[VAL_28:.*]] = %[[CONSTANT_19]]) -> (index) {
// CHECK-NEXT:               %[[CMPI_11:.*]] = arith.cmpi ult, %[[VAL_27]], %[[CONSTANT_20]] : index
// CHECK-NEXT:               %[[IF_6:.*]] = scf.if %[[CMPI_11]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[CONSTANT_19]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[EXTRACT_13:.*]] = tensor.extract %[[FROM_ELEMENTS_8]]{{\[}}%[[VAL_27]]] : tensor<3xindex>
// CHECK-NEXT:                 scf.yield %[[EXTRACT_13]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               %[[CMPI_12:.*]] = arith.cmpi ult, %[[VAL_27]], %[[CONSTANT_20]] : index
// CHECK-NEXT:               %[[IF_7:.*]] = scf.if %[[CMPI_12]] -> (index) {
// CHECK-NEXT:                 scf.yield %[[IF_6]] : index
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[EXTRACT_14:.*]] = tensor.extract %[[FROM_ELEMENTS_9]]{{\[}}%[[VAL_27]]] : tensor<3xindex>
// CHECK-NEXT:                 %[[CMPI_13:.*]] = arith.cmpi eq, %[[EXTRACT_14]], %[[CONSTANT_19]] : index
// CHECK-NEXT:                 %[[SELECT_5:.*]] = arith.select %[[CMPI_13]], %[[IF_6]], %[[EXTRACT_14]] : index
// CHECK-NEXT:                 scf.yield %[[SELECT_5]] : index
// CHECK-NEXT:               }
// CHECK-NEXT:               %[[MULI_1:.*]] = arith.muli %[[IF_7]], %[[VAL_28]] : index
// CHECK-NEXT:               scf.yield %[[MULI_1]] : index
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[FROM_ELEMENTS_10:.*]] = tensor.from_elements %[[CONSTANT_17]], %[[FOR_1]] : tensor<2xindex>
// CHECK-NEXT:             %[[CMPI_14:.*]] = arith.cmpi eq, %[[INDEX_CAST_5]], %[[CONSTANT_19]] : index
// CHECK-NEXT:             %[[SELECT_6:.*]] = arith.select %[[CMPI_14]], %[[DIM_11]], %[[INDEX_CAST_5]] : index
// CHECK-NEXT:             %[[CMPI_15:.*]] = arith.cmpi eq, %[[INDEX_CAST_6]], %[[CONSTANT_19]] : index
// CHECK-NEXT:             %[[SELECT_7:.*]] = arith.select %[[CMPI_15]], %[[DIM_12]], %[[INDEX_CAST_6]] : index
// CHECK-NEXT:             %[[CMPI_16:.*]] = arith.cmpi eq, %[[INDEX_CAST_7]], %[[CONSTANT_19]] : index
// CHECK-NEXT:             %[[SELECT_8:.*]] = arith.select %[[CMPI_16]], %[[DIM_13]], %[[INDEX_CAST_7]] : index
// CHECK-NEXT:             %[[EMPTY_6:.*]] = tensor.empty(%[[SELECT_6]], %[[SELECT_7]], %[[SELECT_8]]) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_7:.*]] = tensor.empty(%[[FOR_1]]) : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_8:.*]] = tensor.empty() : tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_9:.*]] = tensor.empty() : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[EXPAND_1:.*]] = hipsr.expand(%[[VAL_21]]) ins(%[[VAL_24]], %[[VAL_25]] : tensor<?x?x?xi1, #hipsr.mem<device>>, tensor<3xi64, #hipsr.mem<host>>) outs(%[[EMPTY_6]] : tensor<?x?x?xi1, #hipsr.mem<device>>) : tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             %[[NONZERO_0:.*]]:2 = hipsr.nonzero(%[[VAL_21]]) ins(%[[EXPAND_1]] : tensor<?x?x?xi1, #hipsr.mem<device>>) outs(%[[EMPTY_7]], %[[EMPTY_8]] : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) : tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[COPY_D2H_0:.*]] = hipsr.copy_d2h(%[[VAL_21]]) ins(%[[NONZERO_0]]#1 : tensor<1xi64, #hipsr.mem<device>>) outs(%[[EMPTY_9]] : tensor<1xi64, #hipsr.mem<host>>) : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.preserve_shape %[[GENERATE_2]], %[[EXPAND_1]] : tensor<3xindex>, tensor<?x?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_10]], %[[NONZERO_0]]#0 : tensor<2xindex>, tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[CONSTANT_16]], %[[NONZERO_0]]#1 : tensor<1xindex>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[CONSTANT_16]], %[[COPY_D2H_0]] : tensor<1xindex>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_7]], %[[NONZERO_0]]#0, %[[EMPTY_9]], %[[COPY_D2H_0]] : tensor<3x?xi64, #hipsr.mem<device>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<3x?xi64, #hipsr.mem<device>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>> {domain_id = 2 : i64}

// That host count opens the next domain. Its shape graph turns the count into
// the trailing extent, narrowing the search to the columns holding a position.
// The domain takes both buffers twice, one pair per graph.
// CHECK-NEXT:           %[[POOL_DOMAIN_3:.*]]:4 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_2]]#2, %[[POOL_DOMAIN_2]]#0, %[[POOL_DOMAIN_2]]#3, %[[POOL_DOMAIN_2]]#1 : !hipsr.context, tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_29:.*]]: !hipsr.context, %[[VAL_30:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_31:.*]]: tensor<3x?xi64, #hipsr.mem<device>>, %[[VAL_32:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_33:.*]]: tensor<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_21:.*]] = arith.constant dense<1> : tensor<1xindex>
// CHECK-NEXT:             %[[CONSTANT_22:.*]] = arith.constant dense<2> : tensor<1xindex>
// CHECK-NEXT:             %[[CONSTANT_23:.*]] = arith.constant dense<> : tensor<0xindex>
// CHECK-NEXT:             %[[CONSTANT_24:.*]] = arith.constant 3 : index
// CHECK-NEXT:             %[[CONSTANT_25:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[EXTRACT_15:.*]] = tensor.extract %[[VAL_30]]{{\[}}%[[CONSTANT_25]]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_8:.*]] = arith.index_cast %[[EXTRACT_15]] : i64 to index
// CHECK-NEXT:             %[[FROM_ELEMENTS_11:.*]] = tensor.from_elements %[[CONSTANT_24]], %[[INDEX_CAST_8]] : tensor<2xindex>
// CHECK-NEXT:             %[[FROM_ELEMENTS_12:.*]] = tensor.from_elements %[[INDEX_CAST_8]], %[[CONSTANT_24]] : tensor<2xindex>
// CHECK-NEXT:             %[[EMPTY_10:.*]] = tensor.empty(%[[INDEX_CAST_8]]) : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_11:.*]] = tensor.empty(%[[INDEX_CAST_8]]) : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_12:.*]] = tensor.empty() : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[EMPTY_13:.*]] = tensor.empty() : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[EMPTY_14:.*]] = tensor.empty() : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[COMPUTE_3:.*]] = hipsr.compute(%[[VAL_29]]) ins(%[[VAL_32]], %[[VAL_33]] : tensor<1xi64, #hipsr.mem<host>>, tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[EMPTY_10]] : tensor<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_34:.*]]: !hipsr.context, %[[VAL_35:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_36:.*]]: tensor<3x?xi64, #hipsr.mem<device>>, %[[VAL_37:.*]]: tensor<3x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:               %[[CONSTANT_26:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[DIM_14:.*]] = tensor.dim %[[VAL_37]], %[[CONSTANT_26]] : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[VAL_36]]{{\[}}0, 0] [3, %[[DIM_14]]] [1, 1] : tensor<3x?xi64, #hipsr.mem<device>> to tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXTRACT_SLICE_0]] : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             } : tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[TRANSPOSE_0:.*]] = hipsr.transpose(%[[VAL_29]]) ins(%[[COMPUTE_3]] : tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[EMPTY_11]] : tensor<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>} : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:             %[[COMPUTE_4:.*]] = hipsr.compute(%[[VAL_29]]) ins(%[[TRANSPOSE_0]] : tensor<?x3xi64, #hipsr.mem<device>>) outs(%[[EMPTY_12]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_38:.*]]: !hipsr.context, %[[VAL_39:.*]]: tensor<?x3xi64, #hipsr.mem<device>>, %[[VAL_40:.*]]: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_27:.*]] = arith.constant 3 : i64
// CHECK-NEXT:               %[[CONSTANT_28:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[DIM_15:.*]] = tensor.dim %[[VAL_39]], %[[CONSTANT_28]] : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:               %[[INDEX_CAST_9:.*]] = arith.index_cast %[[DIM_15]] : index to i64
// CHECK-NEXT:               %[[FROM_ELEMENTS_13:.*]] = tensor.from_elements %[[INDEX_CAST_9]], %[[CONSTANT_27]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_13]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[COMPUTE_5:.*]] = hipsr.compute(%[[VAL_29]]) ins(%[[COMPUTE_4]] : tensor<2xi64, #hipsr.mem<host>>) outs(%[[EMPTY_13]] : tensor<i64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_41:.*]]: !hipsr.context, %[[VAL_42:.*]]: tensor<2xi64, #hipsr.mem<host>>, %[[VAL_43:.*]]: tensor<i64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[CONSTANT_29:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[EXTRACT_16:.*]] = tensor.extract %[[VAL_42]]{{\[}}%[[CONSTANT_29]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:               %[[FROM_ELEMENTS_14:.*]] = tensor.from_elements %[[EXTRACT_16]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[FROM_ELEMENTS_14]] : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[COMPUTE_6:.*]] = hipsr.compute(%[[VAL_29]]) ins(%[[COMPUTE_5]] : tensor<i64, #hipsr.mem<host>>) outs(%[[EMPTY_14]] : tensor<1xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:             ^bb0(%[[VAL_44:.*]]: !hipsr.context, %[[VAL_45:.*]]: tensor<i64, #hipsr.mem<host>>, %[[VAL_46:.*]]: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:               %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[VAL_45]] [] output_shape [1] : tensor<i64, #hipsr.mem<host>> into tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:               hipsr.compute_yield %[[EXPAND_SHAPE_1]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             } : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_11]], %[[COMPUTE_3]] : tensor<2xindex>, tensor<3x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_12]], %[[TRANSPOSE_0]] : tensor<2xindex>, tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[CONSTANT_22]], %[[COMPUTE_4]] : tensor<1xindex>, tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.preserve_shape %[[CONSTANT_23]], %[[COMPUTE_5]] : tensor<0xindex>, tensor<i64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.preserve_shape %[[CONSTANT_21]], %[[COMPUTE_6]] : tensor<1xindex>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[EMPTY_11]], %[[TRANSPOSE_0]], %[[EMPTY_14]], %[[COMPUTE_6]] : tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:           } -> tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<1xi64, #hipsr.mem<host>> {domain_id = 3 : i64}

// The update window ends at that same count, and the scatter writes it into
// the embeddings.
// CHECK-NEXT:           %[[POOL_DOMAIN_4:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]]#0, %[[POOL_DOMAIN_3]]#2, %[[POOL_DOMAIN_0]]#1, %[[POOL_DOMAIN_3]]#3, %[[POOL_DOMAIN_0]]#4, %[[POOL_DOMAIN_3]]#0, %[[POOL_DOMAIN_0]]#5, %[[POOL_DOMAIN_3]]#1 : !hipsr.context, tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<?xf16, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<host>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:           ^bb0(%[[VAL_47:.*]]: !hipsr.context, %[[VAL_48:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[VAL_49:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_50:.*]]: tensor<?xf16, #hipsr.mem<device>>, %[[VAL_51:.*]]: tensor<1xi64, #hipsr.mem<host>>, %[[VAL_52:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_53:.*]]: tensor<?x3xi64, #hipsr.mem<device>>, %[[VAL_54:.*]]: tensor<?x?x4096xf16, #hipsr.mem<device>>, %[[VAL_55:.*]]: tensor<?x3xi64, #hipsr.mem<device>>):
// CHECK-NEXT:             %[[CONSTANT_30:.*]] = arith.constant 4096 : index
// CHECK-NEXT:             %[[CONSTANT_31:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_32:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[DIM_16:.*]] = tensor.dim %[[VAL_48]], %[[CONSTANT_32]] : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EXTRACT_17:.*]] = tensor.extract %[[VAL_49]]{{\[}}%[[CONSTANT_32]]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:             %[[INDEX_CAST_10:.*]] = arith.index_cast %[[EXTRACT_17]] : i64 to index
// CHECK-NEXT:             %[[CMPI_17:.*]] = arith.cmpi slt, %[[INDEX_CAST_10]], %[[CONSTANT_32]] : index
// CHECK-NEXT:             %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_10]], %[[DIM_16]] : index
// CHECK-NEXT:             %[[SELECT_9:.*]] = arith.select %[[CMPI_17]], %[[ADDI_0]], %[[INDEX_CAST_10]] : index
// CHECK-NEXT:             %[[MINSI_0:.*]] = arith.minsi %[[DIM_16]], %[[CONSTANT_32]] : index
// CHECK-NEXT:             %[[MAXSI_0:.*]] = arith.maxsi %[[SELECT_9]], %[[CONSTANT_32]] : index
// CHECK-NEXT:             %[[MINSI_1:.*]] = arith.minsi %[[MAXSI_0]], %[[DIM_16]] : index
// CHECK-NEXT:             %[[SUBI_1:.*]] = arith.subi %[[MINSI_1]], %[[MINSI_0]] : index
// CHECK-NEXT:             %[[MAXSI_1:.*]] = arith.maxsi %[[SUBI_1]], %[[CONSTANT_32]] : index
// CHECK-NEXT:             %[[FROM_ELEMENTS_15:.*]] = tensor.from_elements %[[MAXSI_1]] : tensor<1xindex>
// CHECK-NEXT:             %[[DIM_17:.*]] = tensor.dim %[[VAL_52]], %[[CONSTANT_32]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[DIM_18:.*]] = tensor.dim %[[VAL_52]], %[[CONSTANT_31]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[FROM_ELEMENTS_16:.*]] = tensor.from_elements %[[DIM_17]], %[[DIM_18]], %[[CONSTANT_30]] : tensor<3xindex>
// CHECK-NEXT:             %[[EMPTY_15:.*]] = tensor.empty(%[[MAXSI_1]]) : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[EMPTY_16:.*]] = tensor.empty(%[[DIM_17]], %[[DIM_18]]) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[SLICE_0:.*]] = hipsr.slice(%[[VAL_47]]) ins(%[[VAL_50]] : tensor<?xf16, #hipsr.mem<device>>) ends(%[[VAL_51]] : tensor<1xi64, #hipsr.mem<host>>) outs(%[[EMPTY_15]] : tensor<?xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 0>, steps_attr = array<i64: 1>} : tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             %[[SCATTER_ND_0:.*]] = hipsr.scatter_nd(%[[VAL_47]]) ins(%[[VAL_54]], %[[VAL_55]], %[[SLICE_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) outs(%[[EMPTY_16]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_15]], %[[SLICE_0]] : tensor<1xindex>, tensor<?xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.preserve_shape %[[FROM_ELEMENTS_16]], %[[SCATTER_ND_0]] : tensor<3xindex>, tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:             hipsr.pool_domain_yield %[[SCATTER_ND_0]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:           } -> tensor<?x?x4096xf16, #hipsr.mem<device>> {domain_id = 4 : i64}
// CHECK-NEXT:           return %[[POOL_DOMAIN_4]] : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:         }
// CHECK-NEXT:       }

// The table's bytes stay outside the IR, so the resource section prints empty
// once the RUN line's elision drops the blob string.
// CHECK-EMPTY:
// CHECK-NEXT:       {-#
// CHECK-EMPTY:
// CHECK-NEXT:       #-}
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
