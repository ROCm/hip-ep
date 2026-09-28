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
//              copy_d2h(count)                       -> count    host 1xi32
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
// The checks cover the full LLVM IR after --hipsr-pipeline.
//
// The embedding table lives in an external file, so the RUN line creates a
// file of the right length to map. Only the length matters; nothing reads the
// weights.

// RUN: %python %S/../../../Inputs/make_external_data.py %t/embedding.onnx.data 2034237440 && cd %t && hip-mlir-opt --onnx-dialect=modeled --hipsr-pipeline --mlir-elide-resource-strings-if-larger=32 %s | FileCheck %s

// generate-interface reads these constant-layout module attributes.
// This graph has no matmul, so the pipeline assigns no op-state slots.
// Rank-3 output: batch, sequence, hidden.

// CHECK-LABEL: module attributes {
// CHECK-SAME: hip.constants_file = "constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64: 0, 64>
// CHECK-SAME: hipdnn.constant_sizes = array<i64: 8, 2034237440>
// CHECK-SAME: hipdnn.input_count = 2 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 8, 2>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: -1, -1>, array<i64: -1, 4096>]
// CHECK-SAME: hipdnn.output_count = 1 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 2>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: -1, -1, 4096>]
// Blob and JSON payload bytes are incidental; only the symbols matter.
// CHECK: llvm.mlir.global internal constant @__metadata_json
// CHECK: llvm.mlir.global internal constant @__metadata_blob
// CHECK:       llvm.func @wrap_scatter_nd(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_slice(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @wrap_transpose(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_copy_d2h(!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_nonzero(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_expand(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_gather(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_equal(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:  llvm.func @free(!llvm.ptr)
// CHECK-NEXT:  llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr<1>
// The ABI wrapper unpacks packed memref descriptors for the graph body.
// CHECK-LABEL: llvm.func private @main_graph(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr) -> i32 attributes {passthrough = ["noinline"]} {
// CHECK-NEXT:    %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[ARG1]]{{\[}}%[[MLIR_0]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_0:.*]] = llvm.load %[[GETELEMENTPTR_0]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[LOAD_1:.*]] = llvm.load %[[LOAD_0]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[LOAD_1]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[LOAD_1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_2:.*]] = llvm.extractvalue %[[LOAD_1]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_3:.*]] = llvm.extractvalue %[[LOAD_1]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_4:.*]] = llvm.extractvalue %[[LOAD_1]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_5:.*]] = llvm.extractvalue %[[LOAD_1]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_6:.*]] = llvm.extractvalue %[[LOAD_1]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[MLIR_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[ARG1]]{{\[}}%[[MLIR_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_2:.*]] = llvm.load %[[GETELEMENTPTR_1]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[LOAD_3:.*]] = llvm.load %[[LOAD_2]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_7:.*]] = llvm.extractvalue %[[LOAD_3]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_8:.*]] = llvm.extractvalue %[[LOAD_3]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_9:.*]] = llvm.extractvalue %[[LOAD_3]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_10:.*]] = llvm.extractvalue %[[LOAD_3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_11:.*]] = llvm.extractvalue %[[LOAD_3]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_12:.*]] = llvm.extractvalue %[[LOAD_3]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[EXTRACTVALUE_13:.*]] = llvm.extractvalue %[[LOAD_3]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[CALL_0:.*]] = llvm.call @main_graph_internal(%[[ARG0]], %[[EXTRACTVALUE_0]], %[[EXTRACTVALUE_1]], %[[EXTRACTVALUE_2]], %[[EXTRACTVALUE_3]], %[[EXTRACTVALUE_4]], %[[EXTRACTVALUE_5]], %[[EXTRACTVALUE_6]], %[[EXTRACTVALUE_7]], %[[EXTRACTVALUE_8]], %[[EXTRACTVALUE_9]], %[[EXTRACTVALUE_10]], %[[EXTRACTVALUE_11]], %[[EXTRACTVALUE_12]], %[[EXTRACTVALUE_13]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[MLIR_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[MLIR_2]] : i32
// CHECK-NEXT:  }
// CHECK-LABEL: llvm.func private @main_graph_internal(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr<1>, %[[ARG2:[^,]*]]: !llvm.ptr<1>, %[[ARG3:[^,]*]]: i64, %[[ARG4:[^,]*]]: i64, %[[ARG5:[^,]*]]: i64, %[[ARG6:[^,]*]]: i64, %[[ARG7:[^,]*]]: i64, %[[ARG8:[^,]*]]: !llvm.ptr<1>, %[[ARG9:[^,]*]]: !llvm.ptr<1>, %[[ARG10:[^,]*]]: i64, %[[ARG11:[^,]*]]: i64, %[[ARG12:[^,]*]]: i64, %[[ARG13:[^,]*]]: i64, %[[ARG14:[^,]*]]: i64) -> (!llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1:.*]] = llvm.insertvalue %[[ARG1]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[ARG2]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[ARG3]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[ARG4]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[ARG6]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.insertvalue %[[ARG5]], %[[V5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[ARG7]], %[[V6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V9:.*]] = llvm.insertvalue %[[ARG8]], %[[V8]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V10:.*]] = llvm.insertvalue %[[ARG9]], %[[V9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V11:.*]] = llvm.insertvalue %[[ARG10]], %[[V10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V12:.*]] = llvm.insertvalue %[[ARG11]], %[[V11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V13:.*]] = llvm.insertvalue %[[ARG13]], %[[V12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V14:.*]] = llvm.insertvalue %[[ARG12]], %[[V13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.*]] = llvm.insertvalue %[[ARG14]], %[[V14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V17:.*]] = llvm.mlir.constant(24 : index) : i64
// CHECK-NEXT:    %[[V18:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V19:.*]] = llvm.mlir.constant(8192 : index) : i64
// CHECK-NEXT:    %[[V20:.*]] = llvm.mlir.constant(255 : index) : i64
// CHECK-NEXT:    %[[V21:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT:    %[[V22:.*]] = llvm.mlir.constant(248320 : index) : i64
// CHECK-NEXT:    %[[V23:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V24:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V25:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V26:.*]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V25]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V27:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V28:.*]] = llvm.insertvalue %[[V26]], %[[V27]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V29:.*]] = llvm.insertvalue %[[V26]], %[[V28]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V30:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V31:.*]] = llvm.insertvalue %[[V30]], %[[V29]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V32:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V33:.*]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V32]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V34:.*]] = llvm.mlir.constant(248320 : i64) : i64
// CHECK-NEXT:    %[[V35:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V36:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V37:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V38:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V39:.*]] = llvm.insertvalue %[[V33]], %[[V38]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V40:.*]] = llvm.insertvalue %[[V33]], %[[V39]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V41:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V42:.*]] = llvm.insertvalue %[[V41]], %[[V40]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V43:.*]] = llvm.insertvalue %[[V34]], %[[V42]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V44:.*]] = llvm.insertvalue %[[V35]], %[[V43]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V45:.*]] = llvm.insertvalue %[[V37]], %[[V44]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V46:.*]] = llvm.insertvalue %[[V36]], %[[V45]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V47:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V48:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V49:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V50:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V51:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V52:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V53:.*]] = llvm.getelementptr %[[V52]][%[[V50]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V54:.*]] = llvm.ptrtoint %[[V53]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V55:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V56:.*]] = llvm.add %[[V54]], %[[V55]] : i64
// CHECK-NEXT:    %[[V57:.*]] = llvm.call @malloc(%[[V56]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V58:.*]] = llvm.ptrtoint %[[V57]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V59:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V60:.*]] = llvm.sub %[[V55]], %[[V59]] : i64
// CHECK-NEXT:    %[[V61:.*]] = llvm.add %[[V58]], %[[V60]] : i64
// CHECK-NEXT:    %[[V62:.*]] = llvm.urem %[[V61]], %[[V55]] : i64
// CHECK-NEXT:    %[[V63:.*]] = llvm.sub %[[V61]], %[[V62]] : i64
// CHECK-NEXT:    %[[V64:.*]] = llvm.inttoptr %[[V63]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V65:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V66:.*]] = llvm.insertvalue %[[V57]], %[[V65]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V67:.*]] = llvm.insertvalue %[[V64]], %[[V66]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V68:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V69:.*]] = llvm.insertvalue %[[V68]], %[[V67]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V70:.*]] = llvm.insertvalue %[[V50]], %[[V69]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V71:.*]] = llvm.insertvalue %[[V51]], %[[V70]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V72:.*]] = llvm.extractvalue %[[V71]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V73:.*]] = llvm.getelementptr inbounds|nuw %[[V72]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V49]], %[[V73]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V74:.*]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V75:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V76:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V77:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V78:.*]] = llvm.getelementptr %[[V77]][%[[V75]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V79:.*]] = llvm.ptrtoint %[[V78]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V80:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V81:.*]] = llvm.add %[[V79]], %[[V80]] : i64
// CHECK-NEXT:    %[[V82:.*]] = llvm.call @malloc(%[[V81]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V83:.*]] = llvm.ptrtoint %[[V82]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V84:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V85:.*]] = llvm.sub %[[V80]], %[[V84]] : i64
// CHECK-NEXT:    %[[V86:.*]] = llvm.add %[[V83]], %[[V85]] : i64
// CHECK-NEXT:    %[[V87:.*]] = llvm.urem %[[V86]], %[[V80]] : i64
// CHECK-NEXT:    %[[V88:.*]] = llvm.sub %[[V86]], %[[V87]] : i64
// CHECK-NEXT:    %[[V89:.*]] = llvm.inttoptr %[[V88]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V90:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V91:.*]] = llvm.insertvalue %[[V82]], %[[V90]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V92:.*]] = llvm.insertvalue %[[V89]], %[[V91]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V93:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V94:.*]] = llvm.insertvalue %[[V93]], %[[V92]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V95:.*]] = llvm.insertvalue %[[V75]], %[[V94]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V96:.*]] = llvm.insertvalue %[[V76]], %[[V95]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V97:.*]] = llvm.extractvalue %[[V96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V98:.*]] = llvm.getelementptr inbounds|nuw %[[V97]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V74]], %[[V98]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V99:.*]] = llvm.extractvalue %[[V96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V100:.*]] = llvm.getelementptr inbounds|nuw %[[V99]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V24]], %[[V100]] : i64, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1(%[[V48]], %[[V47]] : i64, i64)
// CHECK-NEXT:    ^bb1(%[[V101:.*]]: i64, %[[V102:.*]]: i64):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:    %[[V103:.*]] = llvm.icmp "slt" %[[V101]], %[[V23]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V103]], ^bb2, ^bb3
// CHECK-NEXT:    ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %[[V104:.*]] = llvm.extractvalue %[[V96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V105:.*]] = llvm.getelementptr inbounds|nuw %[[V104]][%[[V101]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V106:.*]] = llvm.load %[[V105]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V107:.*]] = llvm.mul %[[V106]], %[[V102]] : i64
// CHECK-NEXT:    %[[V108:.*]] = llvm.add %[[V101]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb1(%[[V108]], %[[V107]] : i64, i64)
// CHECK-NEXT:    ^bb3:  // pred: ^bb1
// CHECK-NEXT:    %[[V109:.*]] = llvm.extractvalue %[[V96]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V109]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V110:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V111:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V112:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V113:.*]] = llvm.getelementptr %[[V112]][%[[V110]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V114:.*]] = llvm.ptrtoint %[[V113]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V115:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V116:.*]] = llvm.add %[[V114]], %[[V115]] : i64
// CHECK-NEXT:    %[[V117:.*]] = llvm.call @malloc(%[[V116]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V118:.*]] = llvm.ptrtoint %[[V117]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V119:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V120:.*]] = llvm.sub %[[V115]], %[[V119]] : i64
// CHECK-NEXT:    %[[V121:.*]] = llvm.add %[[V118]], %[[V120]] : i64
// CHECK-NEXT:    %[[V122:.*]] = llvm.urem %[[V121]], %[[V115]] : i64
// CHECK-NEXT:    %[[V123:.*]] = llvm.sub %[[V121]], %[[V122]] : i64
// CHECK-NEXT:    %[[V124:.*]] = llvm.inttoptr %[[V123]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V125:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V126:.*]] = llvm.insertvalue %[[V117]], %[[V125]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V127:.*]] = llvm.insertvalue %[[V124]], %[[V126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V128:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V129:.*]] = llvm.insertvalue %[[V128]], %[[V127]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V130:.*]] = llvm.insertvalue %[[V110]], %[[V129]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V131:.*]] = llvm.insertvalue %[[V111]], %[[V130]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V132:.*]] = llvm.extractvalue %[[V131]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V133:.*]] = llvm.getelementptr inbounds|nuw %[[V132]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V102]], %[[V133]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V134:.*]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V135:.*]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V136:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V137:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V138:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V139:.*]] = llvm.getelementptr %[[V138]][%[[V136]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V140:.*]] = llvm.ptrtoint %[[V139]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V141:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V142:.*]] = llvm.add %[[V140]], %[[V141]] : i64
// CHECK-NEXT:    %[[V143:.*]] = llvm.call @malloc(%[[V142]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V144:.*]] = llvm.ptrtoint %[[V143]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V145:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V146:.*]] = llvm.sub %[[V141]], %[[V145]] : i64
// CHECK-NEXT:    %[[V147:.*]] = llvm.add %[[V144]], %[[V146]] : i64
// CHECK-NEXT:    %[[V148:.*]] = llvm.urem %[[V147]], %[[V141]] : i64
// CHECK-NEXT:    %[[V149:.*]] = llvm.sub %[[V147]], %[[V148]] : i64
// CHECK-NEXT:    %[[V150:.*]] = llvm.inttoptr %[[V149]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V151:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V152:.*]] = llvm.insertvalue %[[V143]], %[[V151]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V153:.*]] = llvm.insertvalue %[[V150]], %[[V152]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V154:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V155:.*]] = llvm.insertvalue %[[V154]], %[[V153]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V156:.*]] = llvm.insertvalue %[[V136]], %[[V155]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V157:.*]] = llvm.insertvalue %[[V137]], %[[V156]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V158:.*]] = llvm.extractvalue %[[V157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V159:.*]] = llvm.getelementptr inbounds|nuw %[[V158]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V134]], %[[V159]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V160:.*]] = llvm.extractvalue %[[V157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V161:.*]] = llvm.getelementptr inbounds|nuw %[[V160]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V135]], %[[V161]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V162:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V163:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V164:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V165:.*]] = llvm.getelementptr %[[V164]][%[[V162]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V166:.*]] = llvm.ptrtoint %[[V165]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V167:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V168:.*]] = llvm.add %[[V166]], %[[V167]] : i64
// CHECK-NEXT:    %[[V169:.*]] = llvm.call @malloc(%[[V168]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V170:.*]] = llvm.ptrtoint %[[V169]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V171:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V172:.*]] = llvm.sub %[[V167]], %[[V171]] : i64
// CHECK-NEXT:    %[[V173:.*]] = llvm.add %[[V170]], %[[V172]] : i64
// CHECK-NEXT:    %[[V174:.*]] = llvm.urem %[[V173]], %[[V167]] : i64
// CHECK-NEXT:    %[[V175:.*]] = llvm.sub %[[V173]], %[[V174]] : i64
// CHECK-NEXT:    %[[V176:.*]] = llvm.inttoptr %[[V175]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V177:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V178:.*]] = llvm.insertvalue %[[V169]], %[[V177]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V179:.*]] = llvm.insertvalue %[[V176]], %[[V178]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V180:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V181:.*]] = llvm.insertvalue %[[V180]], %[[V179]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V182:.*]] = llvm.insertvalue %[[V162]], %[[V181]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V183:.*]] = llvm.insertvalue %[[V163]], %[[V182]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V184:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V185:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V186:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V187:.*]] = llvm.getelementptr %[[V186]][%[[V184]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V188:.*]] = llvm.ptrtoint %[[V187]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V189:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V190:.*]] = llvm.add %[[V188]], %[[V189]] : i64
// CHECK-NEXT:    %[[V191:.*]] = llvm.call @malloc(%[[V190]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V192:.*]] = llvm.ptrtoint %[[V191]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V193:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V194:.*]] = llvm.sub %[[V189]], %[[V193]] : i64
// CHECK-NEXT:    %[[V195:.*]] = llvm.add %[[V192]], %[[V194]] : i64
// CHECK-NEXT:    %[[V196:.*]] = llvm.urem %[[V195]], %[[V189]] : i64
// CHECK-NEXT:    %[[V197:.*]] = llvm.sub %[[V195]], %[[V196]] : i64
// CHECK-NEXT:    %[[V198:.*]] = llvm.inttoptr %[[V197]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V199:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V200:.*]] = llvm.insertvalue %[[V191]], %[[V199]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V201:.*]] = llvm.insertvalue %[[V198]], %[[V200]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V202:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V203:.*]] = llvm.insertvalue %[[V202]], %[[V201]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V204:.*]] = llvm.insertvalue %[[V184]], %[[V203]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V205:.*]] = llvm.insertvalue %[[V185]], %[[V204]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb4(%[[V48]] : i64)
// CHECK-NEXT:    ^bb4(%[[V206:.*]]: i64):  // 2 preds: ^bb3, ^bb13
// CHECK-NEXT:    %[[V207:.*]] = llvm.icmp "slt" %[[V206]], %[[V23]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V207]], ^bb5, ^bb14
// CHECK-NEXT:    ^bb5:  // pred: ^bb4
// CHECK-NEXT:    %[[V208:.*]] = llvm.icmp "ult" %[[V206]], %[[V48]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V208]], ^bb6, ^bb7
// CHECK-NEXT:    ^bb6:  // pred: ^bb5
// CHECK-NEXT:    llvm.br ^bb8(%[[V47]] : i64)
// CHECK-NEXT:    ^bb7:  // pred: ^bb5
// CHECK-NEXT:    %[[V209:.*]] = llvm.extractvalue %[[V157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V210:.*]] = llvm.getelementptr inbounds|nuw %[[V209]][%[[V206]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V211:.*]] = llvm.load %[[V210]] : !llvm.ptr -> i64
// CHECK-NEXT:    llvm.br ^bb8(%[[V211]] : i64)
// CHECK-NEXT:    ^bb8(%[[V212:.*]]: i64):  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT:    llvm.br ^bb9
// CHECK-NEXT:    ^bb9:  // pred: ^bb8
// CHECK-NEXT:    %[[V213:.*]] = llvm.icmp "ult" %[[V206]], %[[V23]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V213]], ^bb10, ^bb11
// CHECK-NEXT:    ^bb10:  // pred: ^bb9
// CHECK-NEXT:    llvm.br ^bb12(%[[V212]] : i64)
// CHECK-NEXT:    ^bb11:  // pred: ^bb9
// CHECK-NEXT:    %[[V214:.*]] = llvm.sub %[[V206]], %[[V23]] : i64
// CHECK-NEXT:    %[[V215:.*]] = llvm.extractvalue %[[V183]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V216:.*]] = llvm.getelementptr inbounds|nuw %[[V215]][%[[V214]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V217:.*]] = llvm.load %[[V216]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V218:.*]] = llvm.icmp "eq" %[[V217]], %[[V47]] : i64
// CHECK-NEXT:    %[[V219:.*]] = llvm.select %[[V218]], %[[V212]], %[[V217]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb12(%[[V219]] : i64)
// CHECK-NEXT:    ^bb12(%[[V220:.*]]: i64):  // 2 preds: ^bb10, ^bb11
// CHECK-NEXT:    llvm.br ^bb13
// CHECK-NEXT:    ^bb13:  // pred: ^bb12
// CHECK-NEXT:    %[[V221:.*]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V222:.*]] = llvm.getelementptr inbounds|nuw %[[V221]][%[[V206]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V220]], %[[V222]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V223:.*]] = llvm.add %[[V206]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb4(%[[V223]] : i64)
// CHECK-NEXT:    ^bb14:  // pred: ^bb4
// CHECK-NEXT:    %[[V224:.*]] = llvm.extractvalue %[[V183]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V224]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V225:.*]] = llvm.extractvalue %[[V157]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V225]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V226:.*]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V227:.*]] = llvm.getelementptr inbounds|nuw %[[V226]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V228:.*]] = llvm.load %[[V227]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V229:.*]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V230:.*]] = llvm.getelementptr inbounds|nuw %[[V229]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V231:.*]] = llvm.load %[[V230]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V232:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V233:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V234:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V235:.*]] = llvm.getelementptr %[[V234]][%[[V232]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V236:.*]] = llvm.ptrtoint %[[V235]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V237:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V238:.*]] = llvm.add %[[V236]], %[[V237]] : i64
// CHECK-NEXT:    %[[V239:.*]] = llvm.call @malloc(%[[V238]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V240:.*]] = llvm.ptrtoint %[[V239]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V241:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V242:.*]] = llvm.sub %[[V237]], %[[V241]] : i64
// CHECK-NEXT:    %[[V243:.*]] = llvm.add %[[V240]], %[[V242]] : i64
// CHECK-NEXT:    %[[V244:.*]] = llvm.urem %[[V243]], %[[V237]] : i64
// CHECK-NEXT:    %[[V245:.*]] = llvm.sub %[[V243]], %[[V244]] : i64
// CHECK-NEXT:    %[[V246:.*]] = llvm.inttoptr %[[V245]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V247:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V248:.*]] = llvm.insertvalue %[[V239]], %[[V247]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V249:.*]] = llvm.insertvalue %[[V246]], %[[V248]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V250:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V251:.*]] = llvm.insertvalue %[[V250]], %[[V249]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V252:.*]] = llvm.insertvalue %[[V232]], %[[V251]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V253:.*]] = llvm.insertvalue %[[V233]], %[[V252]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V254:.*]] = llvm.extractvalue %[[V253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V255:.*]] = llvm.getelementptr inbounds|nuw %[[V254]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V228]], %[[V255]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V256:.*]] = llvm.extractvalue %[[V253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V257:.*]] = llvm.getelementptr inbounds|nuw %[[V256]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V231]], %[[V257]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V258:.*]] = llvm.extractvalue %[[V253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V259:.*]] = llvm.getelementptr inbounds|nuw %[[V258]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V259]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V260:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V261:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V262:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V263:.*]] = llvm.getelementptr %[[V262]][%[[V260]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V264:.*]] = llvm.ptrtoint %[[V263]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V265:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V266:.*]] = llvm.add %[[V264]], %[[V265]] : i64
// CHECK-NEXT:    %[[V267:.*]] = llvm.call @malloc(%[[V266]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V268:.*]] = llvm.ptrtoint %[[V267]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V269:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V270:.*]] = llvm.sub %[[V265]], %[[V269]] : i64
// CHECK-NEXT:    %[[V271:.*]] = llvm.add %[[V268]], %[[V270]] : i64
// CHECK-NEXT:    %[[V272:.*]] = llvm.urem %[[V271]], %[[V265]] : i64
// CHECK-NEXT:    %[[V273:.*]] = llvm.sub %[[V271]], %[[V272]] : i64
// CHECK-NEXT:    %[[V274:.*]] = llvm.inttoptr %[[V273]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V275:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V276:.*]] = llvm.insertvalue %[[V267]], %[[V275]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V277:.*]] = llvm.insertvalue %[[V274]], %[[V276]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V278:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V279:.*]] = llvm.insertvalue %[[V278]], %[[V277]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V280:.*]] = llvm.insertvalue %[[V260]], %[[V279]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V281:.*]] = llvm.insertvalue %[[V261]], %[[V280]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V282:.*]] = llvm.extractvalue %[[V281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V283:.*]] = llvm.getelementptr inbounds|nuw %[[V282]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V22]], %[[V283]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V284:.*]] = llvm.extractvalue %[[V281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V285:.*]] = llvm.getelementptr inbounds|nuw %[[V284]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V24]], %[[V285]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V286:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V287:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V288:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V289:.*]] = llvm.getelementptr %[[V288]][%[[V286]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V290:.*]] = llvm.ptrtoint %[[V289]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V291:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V292:.*]] = llvm.add %[[V290]], %[[V291]] : i64
// CHECK-NEXT:    %[[V293:.*]] = llvm.call @malloc(%[[V292]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V294:.*]] = llvm.ptrtoint %[[V293]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V295:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V296:.*]] = llvm.sub %[[V291]], %[[V295]] : i64
// CHECK-NEXT:    %[[V297:.*]] = llvm.add %[[V294]], %[[V296]] : i64
// CHECK-NEXT:    %[[V298:.*]] = llvm.urem %[[V297]], %[[V291]] : i64
// CHECK-NEXT:    %[[V299:.*]] = llvm.sub %[[V297]], %[[V298]] : i64
// CHECK-NEXT:    %[[V300:.*]] = llvm.inttoptr %[[V299]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V301:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V302:.*]] = llvm.insertvalue %[[V293]], %[[V301]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V303:.*]] = llvm.insertvalue %[[V300]], %[[V302]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V304:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V305:.*]] = llvm.insertvalue %[[V304]], %[[V303]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V306:.*]] = llvm.insertvalue %[[V286]], %[[V305]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V307:.*]] = llvm.insertvalue %[[V287]], %[[V306]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V308:.*]] = llvm.extractvalue %[[V307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V309:.*]] = llvm.getelementptr inbounds|nuw %[[V308]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V134]], %[[V309]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V310:.*]] = llvm.extractvalue %[[V307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V311:.*]] = llvm.getelementptr inbounds|nuw %[[V310]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V135]], %[[V311]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V312:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V313:.*]] = llvm.extractvalue %[[V281]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V314:.*]] = llvm.extractvalue %[[V281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V315:.*]] = llvm.insertvalue %[[V313]], %[[V312]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V316:.*]] = llvm.insertvalue %[[V314]], %[[V315]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V317:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V318:.*]] = llvm.insertvalue %[[V317]], %[[V316]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V319:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V320:.*]] = llvm.insertvalue %[[V319]], %[[V318]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V321:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V322:.*]] = llvm.insertvalue %[[V321]], %[[V320]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V323:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V324:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V325:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V326:.*]] = llvm.getelementptr %[[V325]][%[[V323]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V327:.*]] = llvm.ptrtoint %[[V326]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V328:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V329:.*]] = llvm.add %[[V327]], %[[V328]] : i64
// CHECK-NEXT:    %[[V330:.*]] = llvm.call @malloc(%[[V329]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V331:.*]] = llvm.ptrtoint %[[V330]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V332:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V333:.*]] = llvm.sub %[[V328]], %[[V332]] : i64
// CHECK-NEXT:    %[[V334:.*]] = llvm.add %[[V331]], %[[V333]] : i64
// CHECK-NEXT:    %[[V335:.*]] = llvm.urem %[[V334]], %[[V328]] : i64
// CHECK-NEXT:    %[[V336:.*]] = llvm.sub %[[V334]], %[[V335]] : i64
// CHECK-NEXT:    %[[V337:.*]] = llvm.inttoptr %[[V336]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V338:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V339:.*]] = llvm.insertvalue %[[V330]], %[[V338]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V340:.*]] = llvm.insertvalue %[[V337]], %[[V339]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V341:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V342:.*]] = llvm.insertvalue %[[V341]], %[[V340]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V343:.*]] = llvm.insertvalue %[[V323]], %[[V342]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V344:.*]] = llvm.insertvalue %[[V324]], %[[V343]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V345:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V346:.*]] = llvm.extractvalue %[[V307]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V347:.*]] = llvm.mul %[[V345]], %[[V346]] : i64
// CHECK-NEXT:    %[[V348:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V349:.*]] = llvm.getelementptr %[[V348]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V350:.*]] = llvm.ptrtoint %[[V349]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V351:.*]] = llvm.mul %[[V347]], %[[V350]] : i64
// CHECK-NEXT:    %[[V352:.*]] = llvm.extractvalue %[[V307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V353:.*]] = llvm.extractvalue %[[V307]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V354:.*]] = llvm.getelementptr %[[V352]][%[[V353]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V355:.*]] = llvm.extractvalue %[[V344]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V356:.*]] = llvm.extractvalue %[[V344]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V357:.*]] = llvm.getelementptr %[[V355]][%[[V356]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V357]], %[[V354]], %[[V351]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V358:.*]] = llvm.extractvalue %[[V307]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V358]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V359:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V360:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V361:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V362:.*]] = llvm.getelementptr %[[V361]][%[[V359]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V363:.*]] = llvm.ptrtoint %[[V362]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V364:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V365:.*]] = llvm.add %[[V363]], %[[V364]] : i64
// CHECK-NEXT:    %[[V366:.*]] = llvm.call @malloc(%[[V365]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V367:.*]] = llvm.ptrtoint %[[V366]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V368:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V369:.*]] = llvm.sub %[[V364]], %[[V368]] : i64
// CHECK-NEXT:    %[[V370:.*]] = llvm.add %[[V367]], %[[V369]] : i64
// CHECK-NEXT:    %[[V371:.*]] = llvm.urem %[[V370]], %[[V364]] : i64
// CHECK-NEXT:    %[[V372:.*]] = llvm.sub %[[V370]], %[[V371]] : i64
// CHECK-NEXT:    %[[V373:.*]] = llvm.inttoptr %[[V372]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V374:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V375:.*]] = llvm.insertvalue %[[V366]], %[[V374]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V376:.*]] = llvm.insertvalue %[[V373]], %[[V375]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V377:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V378:.*]] = llvm.insertvalue %[[V377]], %[[V376]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V379:.*]] = llvm.insertvalue %[[V359]], %[[V378]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V380:.*]] = llvm.insertvalue %[[V360]], %[[V379]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V381:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V382:.*]] = llvm.extractvalue %[[V380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V383:.*]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V384:.*]] = llvm.insertvalue %[[V382]], %[[V381]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V385:.*]] = llvm.insertvalue %[[V383]], %[[V384]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V386:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V387:.*]] = llvm.insertvalue %[[V386]], %[[V385]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V388:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V389:.*]] = llvm.insertvalue %[[V388]], %[[V387]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V390:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V391:.*]] = llvm.insertvalue %[[V390]], %[[V389]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V392:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V393:.*]] = llvm.extractvalue %[[V344]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V394:.*]] = llvm.mul %[[V392]], %[[V393]] : i64
// CHECK-NEXT:    %[[V395:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V396:.*]] = llvm.getelementptr %[[V395]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V397:.*]] = llvm.ptrtoint %[[V396]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V398:.*]] = llvm.mul %[[V394]], %[[V397]] : i64
// CHECK-NEXT:    %[[V399:.*]] = llvm.extractvalue %[[V344]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V400:.*]] = llvm.extractvalue %[[V344]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V401:.*]] = llvm.getelementptr %[[V399]][%[[V400]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V402:.*]] = llvm.extractvalue %[[V391]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V403:.*]] = llvm.extractvalue %[[V391]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V404:.*]] = llvm.getelementptr %[[V402]][%[[V403]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V404]], %[[V401]], %[[V398]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V405:.*]] = llvm.extractvalue %[[V344]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V405]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V406:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V407:.*]] = llvm.extractvalue %[[V380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V408:.*]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V409:.*]] = llvm.insertvalue %[[V407]], %[[V406]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V410:.*]] = llvm.insertvalue %[[V408]], %[[V409]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V411:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V412:.*]] = llvm.insertvalue %[[V411]], %[[V410]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V413:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V414:.*]] = llvm.insertvalue %[[V413]], %[[V412]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V415:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V416:.*]] = llvm.insertvalue %[[V415]], %[[V414]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V417:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V418:.*]] = llvm.extractvalue %[[V322]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V419:.*]] = llvm.mul %[[V417]], %[[V418]] : i64
// CHECK-NEXT:    %[[V420:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V421:.*]] = llvm.getelementptr %[[V420]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V422:.*]] = llvm.ptrtoint %[[V421]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V423:.*]] = llvm.mul %[[V419]], %[[V422]] : i64
// CHECK-NEXT:    %[[V424:.*]] = llvm.extractvalue %[[V322]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V425:.*]] = llvm.extractvalue %[[V322]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V426:.*]] = llvm.getelementptr %[[V424]][%[[V425]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V427:.*]] = llvm.extractvalue %[[V416]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V428:.*]] = llvm.extractvalue %[[V416]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V429:.*]] = llvm.getelementptr %[[V427]][%[[V428]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V429]], %[[V426]], %[[V423]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V430:.*]] = llvm.extractvalue %[[V281]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V430]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V431:.*]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V432:.*]] = llvm.getelementptr inbounds|nuw %[[V431]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V433:.*]] = llvm.load %[[V432]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V434:.*]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V435:.*]] = llvm.getelementptr inbounds|nuw %[[V434]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V436:.*]] = llvm.load %[[V435]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V437:.*]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V438:.*]] = llvm.getelementptr inbounds|nuw %[[V437]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V439:.*]] = llvm.load %[[V438]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V440:.*]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V441:.*]] = llvm.getelementptr inbounds|nuw %[[V440]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V442:.*]] = llvm.load %[[V441]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V443:.*]] = llvm.mul %[[V433]], %[[V436]] : i64
// CHECK-NEXT:    %[[V444:.*]] = llvm.add %[[V443]], %[[V20]] : i64
// CHECK-NEXT:    %[[V445:.*]] = llvm.udiv %[[V444]], %[[V21]] : i64
// CHECK-NEXT:    %[[V446:.*]] = llvm.mul %[[V445]], %[[V21]] : i64
// CHECK-NEXT:    %[[V447:.*]] = llvm.mul %[[V439]], %[[V19]] : i64
// CHECK-NEXT:    %[[V448:.*]] = llvm.mul %[[V447]], %[[V442]] : i64
// CHECK-NEXT:    %[[V449:.*]] = llvm.add %[[V448]], %[[V20]] : i64
// CHECK-NEXT:    %[[V450:.*]] = llvm.udiv %[[V449]], %[[V21]] : i64
// CHECK-NEXT:    %[[V451:.*]] = llvm.mul %[[V450]], %[[V21]] : i64
// CHECK-NEXT:    %[[V452:.*]] = llvm.add %[[V446]], %[[V451]] : i64
// CHECK-NEXT:    %[[V453:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V454:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V453]], %[[V452]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V455:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V456:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V457:.*]] = llvm.insertvalue %[[V454]], %[[V456]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V458:.*]] = llvm.insertvalue %[[V454]], %[[V457]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V459:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V460:.*]] = llvm.insertvalue %[[V459]], %[[V458]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V461:.*]] = llvm.insertvalue %[[V452]], %[[V460]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V462:.*]] = llvm.insertvalue %[[V455]], %[[V461]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V463:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V464:.*]] = llvm.extractvalue %[[V462]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V465:.*]] = llvm.insertvalue %[[V464]], %[[V463]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V466:.*]] = llvm.extractvalue %[[V462]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V467:.*]] = llvm.getelementptr %[[V466]][%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V468:.*]] = llvm.insertvalue %[[V467]], %[[V465]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V469:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V470:.*]] = llvm.insertvalue %[[V469]], %[[V468]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V471:.*]] = llvm.insertvalue %[[V436]], %[[V470]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V472:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V473:.*]] = llvm.insertvalue %[[V472]], %[[V471]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V474:.*]] = llvm.insertvalue %[[V433]], %[[V473]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V475:.*]] = llvm.mul %[[V472]], %[[V436]] : i64
// CHECK-NEXT:    %[[V476:.*]] = llvm.insertvalue %[[V475]], %[[V474]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V477:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V478:.*]] = llvm.extractvalue %[[V462]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V479:.*]] = llvm.insertvalue %[[V478]], %[[V477]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V480:.*]] = llvm.extractvalue %[[V462]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V481:.*]] = llvm.getelementptr %[[V480]][%[[V446]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V482:.*]] = llvm.insertvalue %[[V481]], %[[V479]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V483:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V484:.*]] = llvm.insertvalue %[[V483]], %[[V482]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V485:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V486:.*]] = llvm.insertvalue %[[V485]], %[[V484]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V487:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V488:.*]] = llvm.insertvalue %[[V487]], %[[V486]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V489:.*]] = llvm.insertvalue %[[V442]], %[[V488]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V490:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V491:.*]] = llvm.insertvalue %[[V490]], %[[V489]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V492:.*]] = llvm.insertvalue %[[V439]], %[[V491]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V493:.*]] = llvm.mul %[[V490]], %[[V442]] : i64
// CHECK-NEXT:    %[[V494:.*]] = llvm.insertvalue %[[V493]], %[[V492]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V495:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V496:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V497:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V498:.*]] = llvm.getelementptr %[[V497]][%[[V495]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V499:.*]] = llvm.ptrtoint %[[V498]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V500:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V501:.*]] = llvm.add %[[V499]], %[[V500]] : i64
// CHECK-NEXT:    %[[V502:.*]] = llvm.call @malloc(%[[V501]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V503:.*]] = llvm.ptrtoint %[[V502]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V504:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V505:.*]] = llvm.sub %[[V500]], %[[V504]] : i64
// CHECK-NEXT:    %[[V506:.*]] = llvm.add %[[V503]], %[[V505]] : i64
// CHECK-NEXT:    %[[V507:.*]] = llvm.urem %[[V506]], %[[V500]] : i64
// CHECK-NEXT:    %[[V508:.*]] = llvm.sub %[[V506]], %[[V507]] : i64
// CHECK-NEXT:    %[[V509:.*]] = llvm.inttoptr %[[V508]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V510:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V511:.*]] = llvm.insertvalue %[[V502]], %[[V510]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V512:.*]] = llvm.insertvalue %[[V509]], %[[V511]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V513:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V514:.*]] = llvm.insertvalue %[[V513]], %[[V512]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V515:.*]] = llvm.insertvalue %[[V495]], %[[V514]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V516:.*]] = llvm.insertvalue %[[V496]], %[[V515]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V517:.*]] = llvm.extractvalue %[[V15]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V518:.*]] = llvm.extractvalue %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V519:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V520:.*]] = llvm.insertvalue %[[V517]], %[[V519]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V521:.*]] = llvm.insertvalue %[[V518]], %[[V520]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V522:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V523:.*]] = llvm.insertvalue %[[V522]], %[[V521]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V524:.*]] = llvm.extractvalue %[[V15]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V525:.*]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V526:.*]] = llvm.extractvalue %[[V15]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V527:.*]] = llvm.extractvalue %[[V15]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V528:.*]] = llvm.extractvalue %[[V15]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V529:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V530:.*]] = llvm.mul %[[V525]], %[[V529]] overflow<nsw> : i64
// CHECK-NEXT:    %[[V531:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V532:.*]] = llvm.extractvalue %[[V523]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V533:.*]] = llvm.extractvalue %[[V523]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V534:.*]] = llvm.insertvalue %[[V532]], %[[V531]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V535:.*]] = llvm.insertvalue %[[V533]], %[[V534]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V536:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V537:.*]] = llvm.insertvalue %[[V536]], %[[V535]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V538:.*]] = llvm.insertvalue %[[V530]], %[[V537]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V539:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V540:.*]] = llvm.insertvalue %[[V539]], %[[V538]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V541:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V542:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V543:.*]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V544:.*]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V545:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V546:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V547:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V548:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V549:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V550:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V551:.*]] = llvm.extractvalue %[[V476]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V552:.*]] = llvm.extractvalue %[[V476]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V553:.*]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V554:.*]] = llvm.extractvalue %[[V31]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V555:.*]] = llvm.extractvalue %[[V476]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V556:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V557:.*]] = llvm.call @wrap_equal(%[[ARG0]], %[[V553]], %[[V554]], %[[V555]], %[[V541]], %[[V542]], %[[V543]], %[[V544]], %[[V545]], %[[V546]], %[[V547]], %[[V548]], %[[V549]], %[[V550]], %[[V551]], %[[V552]], %[[V556]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V558:.*]] = llvm.extractvalue %[[V476]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V559:.*]] = llvm.extractvalue %[[V476]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V560:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V561:.*]] = llvm.insertvalue %[[V558]], %[[V560]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V562:.*]] = llvm.insertvalue %[[V559]], %[[V561]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V563:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V564:.*]] = llvm.insertvalue %[[V563]], %[[V562]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V565:.*]] = llvm.extractvalue %[[V476]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V566:.*]] = llvm.extractvalue %[[V476]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V567:.*]] = llvm.extractvalue %[[V476]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V568:.*]] = llvm.extractvalue %[[V476]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V569:.*]] = llvm.extractvalue %[[V476]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V570:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V571:.*]] = llvm.extractvalue %[[V564]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V572:.*]] = llvm.extractvalue %[[V564]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V573:.*]] = llvm.insertvalue %[[V571]], %[[V570]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V574:.*]] = llvm.insertvalue %[[V572]], %[[V573]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V575:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V576:.*]] = llvm.insertvalue %[[V575]], %[[V574]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V577:.*]] = llvm.insertvalue %[[V566]], %[[V576]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V578:.*]] = llvm.insertvalue %[[V568]], %[[V577]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V579:.*]] = llvm.insertvalue %[[V567]], %[[V578]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V580:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V581:.*]] = llvm.insertvalue %[[V580]], %[[V579]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V582:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V583:.*]] = llvm.insertvalue %[[V582]], %[[V581]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V584:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V585:.*]] = llvm.insertvalue %[[V584]], %[[V583]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V586:.*]] = llvm.mlir.constant(248320 : i64) : i64
// CHECK-NEXT:    %[[V587:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V588:.*]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V589:.*]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V590:.*]] = llvm.extractvalue %[[V494]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V591:.*]] = llvm.extractvalue %[[V494]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V592:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V593:.*]] = llvm.mul %[[V586]], %[[V587]] : i64
// CHECK-NEXT:    %[[V594:.*]] = llvm.mul %[[V590]], %[[V591]] : i64
// CHECK-NEXT:    %[[V595:.*]] = llvm.mul %[[V594]], %[[V592]] : i64
// CHECK-NEXT:    %[[V596:.*]] = llvm.mul %[[V588]], %[[V589]] : i64
// CHECK-NEXT:    %[[V597:.*]] = llvm.extractvalue %[[V46]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V598:.*]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V599:.*]] = llvm.extractvalue %[[V494]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V600:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V601:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V602:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %[[V603:.*]] = llvm.call @wrap_gather(%[[ARG0]], %[[V597]], %[[V598]], %[[V599]], %[[V600]], %[[V593]], %[[V596]], %[[V595]], %[[V586]], %[[V587]], %[[V601]], %[[V602]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V604:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V605:.*]] = llvm.getelementptr inbounds|nuw %[[V604]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V439]], %[[V605]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V606:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V607:.*]] = llvm.getelementptr inbounds|nuw %[[V606]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V442]], %[[V607]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V608:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V609:.*]] = llvm.getelementptr inbounds|nuw %[[V608]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V18]], %[[V609]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V610:.*]] = llvm.extractvalue %[[V131]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V610]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V611:.*]] = llvm.extractvalue %[[V205]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V611]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V612:.*]] = llvm.extractvalue %[[V253]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V612]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V613:.*]] = llvm.extractvalue %[[V380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V613]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V614:.*]] = llvm.extractvalue %[[V71]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V614]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V615:.*]] = llvm.extractvalue %[[V585]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V616:.*]] = llvm.extractvalue %[[V585]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V617:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V618:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V619:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V620:.*]] = llvm.getelementptr %[[V619]][%[[V617]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V621:.*]] = llvm.ptrtoint %[[V620]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V622:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V623:.*]] = llvm.add %[[V621]], %[[V622]] : i64
// CHECK-NEXT:    %[[V624:.*]] = llvm.call @malloc(%[[V623]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V625:.*]] = llvm.ptrtoint %[[V624]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V626:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V627:.*]] = llvm.sub %[[V622]], %[[V626]] : i64
// CHECK-NEXT:    %[[V628:.*]] = llvm.add %[[V625]], %[[V627]] : i64
// CHECK-NEXT:    %[[V629:.*]] = llvm.urem %[[V628]], %[[V622]] : i64
// CHECK-NEXT:    %[[V630:.*]] = llvm.sub %[[V628]], %[[V629]] : i64
// CHECK-NEXT:    %[[V631:.*]] = llvm.inttoptr %[[V630]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V632:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V633:.*]] = llvm.insertvalue %[[V624]], %[[V632]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V634:.*]] = llvm.insertvalue %[[V631]], %[[V633]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V635:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V636:.*]] = llvm.insertvalue %[[V635]], %[[V634]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V637:.*]] = llvm.insertvalue %[[V617]], %[[V636]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V638:.*]] = llvm.insertvalue %[[V618]], %[[V637]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V639:.*]] = llvm.extractvalue %[[V638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V640:.*]] = llvm.getelementptr inbounds|nuw %[[V639]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V615]], %[[V640]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V641:.*]] = llvm.extractvalue %[[V638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V642:.*]] = llvm.getelementptr inbounds|nuw %[[V641]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V616]], %[[V642]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V643:.*]] = llvm.extractvalue %[[V638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V644:.*]] = llvm.getelementptr inbounds|nuw %[[V643]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V644]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V645:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V646:.*]] = llvm.getelementptr inbounds|nuw %[[V645]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V647:.*]] = llvm.load %[[V646]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V648:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V649:.*]] = llvm.getelementptr inbounds|nuw %[[V648]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V650:.*]] = llvm.load %[[V649]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V651:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V652:.*]] = llvm.getelementptr inbounds|nuw %[[V651]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V653:.*]] = llvm.load %[[V652]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V654:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V655:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V656:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V657:.*]] = llvm.getelementptr %[[V656]][%[[V654]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V658:.*]] = llvm.ptrtoint %[[V657]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V659:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V660:.*]] = llvm.add %[[V658]], %[[V659]] : i64
// CHECK-NEXT:    %[[V661:.*]] = llvm.call @malloc(%[[V660]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V662:.*]] = llvm.ptrtoint %[[V661]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V663:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V664:.*]] = llvm.sub %[[V659]], %[[V663]] : i64
// CHECK-NEXT:    %[[V665:.*]] = llvm.add %[[V662]], %[[V664]] : i64
// CHECK-NEXT:    %[[V666:.*]] = llvm.urem %[[V665]], %[[V659]] : i64
// CHECK-NEXT:    %[[V667:.*]] = llvm.sub %[[V665]], %[[V666]] : i64
// CHECK-NEXT:    %[[V668:.*]] = llvm.inttoptr %[[V667]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V669:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V670:.*]] = llvm.insertvalue %[[V661]], %[[V669]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V671:.*]] = llvm.insertvalue %[[V668]], %[[V670]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V672:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V673:.*]] = llvm.insertvalue %[[V672]], %[[V671]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V674:.*]] = llvm.insertvalue %[[V654]], %[[V673]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V675:.*]] = llvm.insertvalue %[[V655]], %[[V674]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V676:.*]] = llvm.extractvalue %[[V675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V677:.*]] = llvm.getelementptr inbounds|nuw %[[V676]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V647]], %[[V677]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V678:.*]] = llvm.extractvalue %[[V675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V679:.*]] = llvm.getelementptr inbounds|nuw %[[V678]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V650]], %[[V679]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V680:.*]] = llvm.extractvalue %[[V675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V681:.*]] = llvm.getelementptr inbounds|nuw %[[V680]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V653]], %[[V681]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V682:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V683:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V684:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V685:.*]] = llvm.getelementptr %[[V684]][%[[V682]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V686:.*]] = llvm.ptrtoint %[[V685]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V687:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V688:.*]] = llvm.add %[[V686]], %[[V687]] : i64
// CHECK-NEXT:    %[[V689:.*]] = llvm.call @malloc(%[[V688]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V690:.*]] = llvm.ptrtoint %[[V689]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V691:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V692:.*]] = llvm.sub %[[V687]], %[[V691]] : i64
// CHECK-NEXT:    %[[V693:.*]] = llvm.add %[[V690]], %[[V692]] : i64
// CHECK-NEXT:    %[[V694:.*]] = llvm.urem %[[V693]], %[[V687]] : i64
// CHECK-NEXT:    %[[V695:.*]] = llvm.sub %[[V693]], %[[V694]] : i64
// CHECK-NEXT:    %[[V696:.*]] = llvm.inttoptr %[[V695]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V697:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V698:.*]] = llvm.insertvalue %[[V689]], %[[V697]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V699:.*]] = llvm.insertvalue %[[V696]], %[[V698]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V700:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V701:.*]] = llvm.insertvalue %[[V700]], %[[V699]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V702:.*]] = llvm.insertvalue %[[V682]], %[[V701]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V703:.*]] = llvm.insertvalue %[[V683]], %[[V702]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb15(%[[V48]] : i64)
// CHECK-NEXT:    ^bb15(%[[V704:.*]]: i64):  // 2 preds: ^bb14, ^bb20
// CHECK-NEXT:    %[[V705:.*]] = llvm.icmp "slt" %[[V704]], %[[V49]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V705]], ^bb16, ^bb21
// CHECK-NEXT:    ^bb16:  // pred: ^bb15
// CHECK-NEXT:    %[[V706:.*]] = llvm.icmp "ult" %[[V704]], %[[V48]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V706]], ^bb17, ^bb18
// CHECK-NEXT:    ^bb17:  // pred: ^bb16
// CHECK-NEXT:    llvm.br ^bb19(%[[V47]] : i64)
// CHECK-NEXT:    ^bb18:  // pred: ^bb16
// CHECK-NEXT:    %[[V707:.*]] = llvm.extractvalue %[[V638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V708:.*]] = llvm.getelementptr inbounds|nuw %[[V707]][%[[V704]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V709:.*]] = llvm.load %[[V708]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V710:.*]] = llvm.extractvalue %[[V675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V711:.*]] = llvm.getelementptr inbounds|nuw %[[V710]][%[[V704]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V712:.*]] = llvm.load %[[V711]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V713:.*]] = llvm.icmp "eq" %[[V712]], %[[V47]] : i64
// CHECK-NEXT:    %[[V714:.*]] = llvm.select %[[V713]], %[[V709]], %[[V712]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb19(%[[V714]] : i64)
// CHECK-NEXT:    ^bb19(%[[V715:.*]]: i64):  // 2 preds: ^bb17, ^bb18
// CHECK-NEXT:    llvm.br ^bb20
// CHECK-NEXT:    ^bb20:  // pred: ^bb19
// CHECK-NEXT:    %[[V716:.*]] = llvm.extractvalue %[[V703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V717:.*]] = llvm.getelementptr inbounds|nuw %[[V716]][%[[V704]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V715]], %[[V717]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V718:.*]] = llvm.add %[[V704]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb15(%[[V718]] : i64)
// CHECK-NEXT:    ^bb21:  // pred: ^bb15
// CHECK-NEXT:    %[[V719:.*]] = llvm.extractvalue %[[V675]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V719]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V720:.*]] = llvm.extractvalue %[[V638]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V720]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V721:.*]] = llvm.extractvalue %[[V703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V722:.*]] = llvm.getelementptr inbounds|nuw %[[V721]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V723:.*]] = llvm.load %[[V722]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V724:.*]] = llvm.extractvalue %[[V703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V725:.*]] = llvm.getelementptr inbounds|nuw %[[V724]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V726:.*]] = llvm.load %[[V725]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V727:.*]] = llvm.extractvalue %[[V703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V728:.*]] = llvm.getelementptr inbounds|nuw %[[V727]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V729:.*]] = llvm.load %[[V728]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V730:.*]] = llvm.mul %[[V723]], %[[V726]] : i64
// CHECK-NEXT:    %[[V731:.*]] = llvm.mul %[[V730]], %[[V729]] : i64
// CHECK-NEXT:    %[[V732:.*]] = llvm.add %[[V731]], %[[V20]] : i64
// CHECK-NEXT:    %[[V733:.*]] = llvm.udiv %[[V732]], %[[V21]] : i64
// CHECK-NEXT:    %[[V734:.*]] = llvm.mul %[[V733]], %[[V21]] : i64
// CHECK-NEXT:    %[[V735:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V736:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V735]], %[[V734]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V737:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V738:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V739:.*]] = llvm.insertvalue %[[V736]], %[[V738]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V740:.*]] = llvm.insertvalue %[[V736]], %[[V739]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V741:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V742:.*]] = llvm.insertvalue %[[V741]], %[[V740]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V743:.*]] = llvm.insertvalue %[[V734]], %[[V742]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V744:.*]] = llvm.insertvalue %[[V737]], %[[V743]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V745:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V746:.*]] = llvm.extractvalue %[[V744]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V747:.*]] = llvm.insertvalue %[[V746]], %[[V745]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V748:.*]] = llvm.extractvalue %[[V744]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V749:.*]] = llvm.getelementptr %[[V748]][%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V750:.*]] = llvm.insertvalue %[[V749]], %[[V747]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V751:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V752:.*]] = llvm.insertvalue %[[V751]], %[[V750]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V753:.*]] = llvm.insertvalue %[[V729]], %[[V752]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V754:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V755:.*]] = llvm.insertvalue %[[V754]], %[[V753]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V756:.*]] = llvm.insertvalue %[[V726]], %[[V755]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V757:.*]] = llvm.mul %[[V754]], %[[V729]] : i64
// CHECK-NEXT:    %[[V758:.*]] = llvm.insertvalue %[[V757]], %[[V756]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V759:.*]] = llvm.insertvalue %[[V723]], %[[V758]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V760:.*]] = llvm.mul %[[V757]], %[[V726]] : i64
// CHECK-NEXT:    %[[V761:.*]] = llvm.insertvalue %[[V760]], %[[V759]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V762:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V763:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V764:.*]] = llvm.alloca %[[V762]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V765:.*]] = llvm.extractvalue %[[V585]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V766:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V767:.*]] = llvm.getelementptr %[[V764]][%[[V766]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V765]], %[[V767]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V768:.*]] = llvm.extractvalue %[[V585]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V769:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V770:.*]] = llvm.getelementptr %[[V764]][%[[V769]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V768]], %[[V770]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V771:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V772:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V773:.*]] = llvm.getelementptr %[[V764]][%[[V772]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V771]], %[[V773]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V774:.*]] = llvm.alloca %[[V762]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V775:.*]] = llvm.extractvalue %[[V761]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V776:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V777:.*]] = llvm.getelementptr %[[V774]][%[[V776]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V775]], %[[V777]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V778:.*]] = llvm.extractvalue %[[V761]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V779:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V780:.*]] = llvm.getelementptr %[[V774]][%[[V779]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V778]], %[[V780]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V781:.*]] = llvm.extractvalue %[[V761]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V782:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V783:.*]] = llvm.getelementptr %[[V774]][%[[V782]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V781]], %[[V783]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V784:.*]] = llvm.extractvalue %[[V585]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V785:.*]] = llvm.extractvalue %[[V761]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V786:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V787:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V788:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:    %[[V789:.*]] = llvm.call @wrap_expand(%[[ARG0]], %[[V784]], %[[V763]], %[[V785]], %[[V764]], %[[V786]], %[[V774]], %[[V787]], %[[V788]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V790:.*]] = llvm.extractvalue %[[V703]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V790]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V791:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V792:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V793:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V794:.*]] = llvm.getelementptr %[[V793]][%[[V791]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V795:.*]] = llvm.ptrtoint %[[V794]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V796:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V797:.*]] = llvm.add %[[V795]], %[[V796]] : i64
// CHECK-NEXT:    %[[V798:.*]] = llvm.call @malloc(%[[V797]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V799:.*]] = llvm.ptrtoint %[[V798]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V800:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V801:.*]] = llvm.sub %[[V796]], %[[V800]] : i64
// CHECK-NEXT:    %[[V802:.*]] = llvm.add %[[V799]], %[[V801]] : i64
// CHECK-NEXT:    %[[V803:.*]] = llvm.urem %[[V802]], %[[V796]] : i64
// CHECK-NEXT:    %[[V804:.*]] = llvm.sub %[[V802]], %[[V803]] : i64
// CHECK-NEXT:    %[[V805:.*]] = llvm.inttoptr %[[V804]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V806:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V807:.*]] = llvm.insertvalue %[[V798]], %[[V806]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V808:.*]] = llvm.insertvalue %[[V805]], %[[V807]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V809:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V810:.*]] = llvm.insertvalue %[[V809]], %[[V808]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V811:.*]] = llvm.insertvalue %[[V791]], %[[V810]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V812:.*]] = llvm.insertvalue %[[V792]], %[[V811]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V813:.*]] = llvm.extractvalue %[[V812]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V814:.*]] = llvm.getelementptr inbounds|nuw %[[V813]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V814]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V815:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V816:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V817:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V818:.*]] = llvm.getelementptr %[[V817]][%[[V815]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V819:.*]] = llvm.ptrtoint %[[V818]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V820:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V821:.*]] = llvm.add %[[V819]], %[[V820]] : i64
// CHECK-NEXT:    %[[V822:.*]] = llvm.call @malloc(%[[V821]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V823:.*]] = llvm.ptrtoint %[[V822]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V824:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V825:.*]] = llvm.sub %[[V820]], %[[V824]] : i64
// CHECK-NEXT:    %[[V826:.*]] = llvm.add %[[V823]], %[[V825]] : i64
// CHECK-NEXT:    %[[V827:.*]] = llvm.urem %[[V826]], %[[V820]] : i64
// CHECK-NEXT:    %[[V828:.*]] = llvm.sub %[[V826]], %[[V827]] : i64
// CHECK-NEXT:    %[[V829:.*]] = llvm.inttoptr %[[V828]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V830:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V831:.*]] = llvm.insertvalue %[[V822]], %[[V830]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V832:.*]] = llvm.insertvalue %[[V829]], %[[V831]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V833:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V834:.*]] = llvm.insertvalue %[[V833]], %[[V832]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V835:.*]] = llvm.insertvalue %[[V815]], %[[V834]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V836:.*]] = llvm.insertvalue %[[V816]], %[[V835]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V837:.*]] = llvm.extractvalue %[[V836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V838:.*]] = llvm.getelementptr inbounds|nuw %[[V837]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V723]], %[[V838]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V839:.*]] = llvm.extractvalue %[[V836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V840:.*]] = llvm.getelementptr inbounds|nuw %[[V839]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V726]], %[[V840]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V841:.*]] = llvm.extractvalue %[[V836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V842:.*]] = llvm.getelementptr inbounds|nuw %[[V841]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V729]], %[[V842]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V843:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V844:.*]] = llvm.getelementptr inbounds|nuw %[[V843]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V845:.*]] = llvm.load %[[V844]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V846:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V847:.*]] = llvm.getelementptr inbounds|nuw %[[V846]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V848:.*]] = llvm.load %[[V847]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V849:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V850:.*]] = llvm.getelementptr inbounds|nuw %[[V849]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V851:.*]] = llvm.load %[[V850]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V852:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V853:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V854:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V855:.*]] = llvm.getelementptr %[[V854]][%[[V852]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V856:.*]] = llvm.ptrtoint %[[V855]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V857:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V858:.*]] = llvm.add %[[V856]], %[[V857]] : i64
// CHECK-NEXT:    %[[V859:.*]] = llvm.call @malloc(%[[V858]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V860:.*]] = llvm.ptrtoint %[[V859]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V861:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V862:.*]] = llvm.sub %[[V857]], %[[V861]] : i64
// CHECK-NEXT:    %[[V863:.*]] = llvm.add %[[V860]], %[[V862]] : i64
// CHECK-NEXT:    %[[V864:.*]] = llvm.urem %[[V863]], %[[V857]] : i64
// CHECK-NEXT:    %[[V865:.*]] = llvm.sub %[[V863]], %[[V864]] : i64
// CHECK-NEXT:    %[[V866:.*]] = llvm.inttoptr %[[V865]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V867:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V868:.*]] = llvm.insertvalue %[[V859]], %[[V867]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V869:.*]] = llvm.insertvalue %[[V866]], %[[V868]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V870:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V871:.*]] = llvm.insertvalue %[[V870]], %[[V869]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V872:.*]] = llvm.insertvalue %[[V852]], %[[V871]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V873:.*]] = llvm.insertvalue %[[V853]], %[[V872]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V874:.*]] = llvm.extractvalue %[[V873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V875:.*]] = llvm.getelementptr inbounds|nuw %[[V874]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V845]], %[[V875]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V876:.*]] = llvm.extractvalue %[[V873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V877:.*]] = llvm.getelementptr inbounds|nuw %[[V876]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V848]], %[[V877]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V878:.*]] = llvm.extractvalue %[[V873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V879:.*]] = llvm.getelementptr inbounds|nuw %[[V878]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V851]], %[[V879]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V880:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V881:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V882:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V883:.*]] = llvm.getelementptr %[[V882]][%[[V880]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V884:.*]] = llvm.ptrtoint %[[V883]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V885:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V886:.*]] = llvm.add %[[V884]], %[[V885]] : i64
// CHECK-NEXT:    %[[V887:.*]] = llvm.call @malloc(%[[V886]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V888:.*]] = llvm.ptrtoint %[[V887]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V889:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V890:.*]] = llvm.sub %[[V885]], %[[V889]] : i64
// CHECK-NEXT:    %[[V891:.*]] = llvm.add %[[V888]], %[[V890]] : i64
// CHECK-NEXT:    %[[V892:.*]] = llvm.urem %[[V891]], %[[V885]] : i64
// CHECK-NEXT:    %[[V893:.*]] = llvm.sub %[[V891]], %[[V892]] : i64
// CHECK-NEXT:    %[[V894:.*]] = llvm.inttoptr %[[V893]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V895:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V896:.*]] = llvm.insertvalue %[[V887]], %[[V895]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V897:.*]] = llvm.insertvalue %[[V894]], %[[V896]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V898:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V899:.*]] = llvm.insertvalue %[[V898]], %[[V897]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V900:.*]] = llvm.insertvalue %[[V880]], %[[V899]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V901:.*]] = llvm.insertvalue %[[V881]], %[[V900]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb22(%[[V48]] : i64)
// CHECK-NEXT:    ^bb22(%[[V902:.*]]: i64):  // 2 preds: ^bb21, ^bb27
// CHECK-NEXT:    %[[V903:.*]] = llvm.icmp "slt" %[[V902]], %[[V49]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V903]], ^bb23, ^bb28
// CHECK-NEXT:    ^bb23:  // pred: ^bb22
// CHECK-NEXT:    %[[V904:.*]] = llvm.icmp "ult" %[[V902]], %[[V48]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V904]], ^bb24, ^bb25
// CHECK-NEXT:    ^bb24:  // pred: ^bb23
// CHECK-NEXT:    llvm.br ^bb26(%[[V47]] : i64)
// CHECK-NEXT:    ^bb25:  // pred: ^bb23
// CHECK-NEXT:    %[[V905:.*]] = llvm.extractvalue %[[V836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V906:.*]] = llvm.getelementptr inbounds|nuw %[[V905]][%[[V902]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V907:.*]] = llvm.load %[[V906]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V908:.*]] = llvm.extractvalue %[[V873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V909:.*]] = llvm.getelementptr inbounds|nuw %[[V908]][%[[V902]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V910:.*]] = llvm.load %[[V909]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V911:.*]] = llvm.icmp "eq" %[[V910]], %[[V47]] : i64
// CHECK-NEXT:    %[[V912:.*]] = llvm.select %[[V911]], %[[V907]], %[[V910]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb26(%[[V912]] : i64)
// CHECK-NEXT:    ^bb26(%[[V913:.*]]: i64):  // 2 preds: ^bb24, ^bb25
// CHECK-NEXT:    llvm.br ^bb27
// CHECK-NEXT:    ^bb27:  // pred: ^bb26
// CHECK-NEXT:    %[[V914:.*]] = llvm.extractvalue %[[V901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V915:.*]] = llvm.getelementptr inbounds|nuw %[[V914]][%[[V902]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V913]], %[[V915]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V916:.*]] = llvm.add %[[V902]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb22(%[[V916]] : i64)
// CHECK-NEXT:    ^bb28:  // pred: ^bb22
// CHECK-NEXT:    %[[V917:.*]] = llvm.extractvalue %[[V873]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V917]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V918:.*]] = llvm.extractvalue %[[V836]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V918]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb29(%[[V48]], %[[V47]] : i64, i64)
// CHECK-NEXT:    ^bb29(%[[V919:.*]]: i64, %[[V920:.*]]: i64):  // 2 preds: ^bb28, ^bb30
// CHECK-NEXT:    %[[V921:.*]] = llvm.icmp "slt" %[[V919]], %[[V49]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V921]], ^bb30, ^bb31
// CHECK-NEXT:    ^bb30:  // pred: ^bb29
// CHECK-NEXT:    %[[V922:.*]] = llvm.extractvalue %[[V901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V923:.*]] = llvm.getelementptr inbounds|nuw %[[V922]][%[[V919]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V924:.*]] = llvm.load %[[V923]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V925:.*]] = llvm.mul %[[V924]], %[[V920]] : i64
// CHECK-NEXT:    %[[V926:.*]] = llvm.add %[[V919]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb29(%[[V926]], %[[V925]] : i64, i64)
// CHECK-NEXT:    ^bb31:  // pred: ^bb29
// CHECK-NEXT:    %[[V927:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V928:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V929:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V930:.*]] = llvm.getelementptr %[[V929]][%[[V927]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V931:.*]] = llvm.ptrtoint %[[V930]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V932:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V933:.*]] = llvm.add %[[V931]], %[[V932]] : i64
// CHECK-NEXT:    %[[V934:.*]] = llvm.call @malloc(%[[V933]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V935:.*]] = llvm.ptrtoint %[[V934]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V936:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V937:.*]] = llvm.sub %[[V932]], %[[V936]] : i64
// CHECK-NEXT:    %[[V938:.*]] = llvm.add %[[V935]], %[[V937]] : i64
// CHECK-NEXT:    %[[V939:.*]] = llvm.urem %[[V938]], %[[V932]] : i64
// CHECK-NEXT:    %[[V940:.*]] = llvm.sub %[[V938]], %[[V939]] : i64
// CHECK-NEXT:    %[[V941:.*]] = llvm.inttoptr %[[V940]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V942:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V943:.*]] = llvm.insertvalue %[[V934]], %[[V942]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V944:.*]] = llvm.insertvalue %[[V941]], %[[V943]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V945:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V946:.*]] = llvm.insertvalue %[[V945]], %[[V944]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V947:.*]] = llvm.insertvalue %[[V927]], %[[V946]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V948:.*]] = llvm.insertvalue %[[V928]], %[[V947]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V949:.*]] = llvm.extractvalue %[[V948]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V950:.*]] = llvm.getelementptr inbounds|nuw %[[V949]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V49]], %[[V950]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V951:.*]] = llvm.extractvalue %[[V948]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V952:.*]] = llvm.getelementptr inbounds|nuw %[[V951]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V920]], %[[V952]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V953:.*]] = llvm.extractvalue %[[V901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V954:.*]] = llvm.getelementptr inbounds|nuw %[[V953]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V955:.*]] = llvm.load %[[V954]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V956:.*]] = llvm.extractvalue %[[V901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V957:.*]] = llvm.getelementptr inbounds|nuw %[[V956]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V958:.*]] = llvm.load %[[V957]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V959:.*]] = llvm.extractvalue %[[V901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V960:.*]] = llvm.getelementptr inbounds|nuw %[[V959]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V961:.*]] = llvm.load %[[V960]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V962:.*]] = llvm.extractvalue %[[V948]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V963:.*]] = llvm.getelementptr inbounds|nuw %[[V962]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V964:.*]] = llvm.load %[[V963]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V965:.*]] = llvm.mul %[[V955]], %[[V958]] : i64
// CHECK-NEXT:    %[[V966:.*]] = llvm.mul %[[V965]], %[[V961]] : i64
// CHECK-NEXT:    %[[V967:.*]] = llvm.add %[[V966]], %[[V20]] : i64
// CHECK-NEXT:    %[[V968:.*]] = llvm.udiv %[[V967]], %[[V21]] : i64
// CHECK-NEXT:    %[[V969:.*]] = llvm.mul %[[V968]], %[[V21]] : i64
// CHECK-NEXT:    %[[V970:.*]] = llvm.mul %[[V964]], %[[V17]] : i64
// CHECK-NEXT:    %[[V971:.*]] = llvm.add %[[V970]], %[[V20]] : i64
// CHECK-NEXT:    %[[V972:.*]] = llvm.udiv %[[V971]], %[[V21]] : i64
// CHECK-NEXT:    %[[V973:.*]] = llvm.mul %[[V972]], %[[V21]] : i64
// CHECK-NEXT:    %[[V974:.*]] = llvm.add %[[V969]], %[[V973]] : i64
// CHECK-NEXT:    %[[V975:.*]] = llvm.add %[[V974]], %[[V21]] : i64
// CHECK-NEXT:    %[[V976:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V977:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V976]], %[[V975]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V978:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V979:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V980:.*]] = llvm.insertvalue %[[V977]], %[[V979]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V981:.*]] = llvm.insertvalue %[[V977]], %[[V980]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V982:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V983:.*]] = llvm.insertvalue %[[V982]], %[[V981]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V984:.*]] = llvm.insertvalue %[[V975]], %[[V983]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V985:.*]] = llvm.insertvalue %[[V978]], %[[V984]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V986:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V987:.*]] = llvm.extractvalue %[[V985]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V988:.*]] = llvm.insertvalue %[[V987]], %[[V986]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V989:.*]] = llvm.extractvalue %[[V985]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V990:.*]] = llvm.getelementptr %[[V989]][%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V991:.*]] = llvm.insertvalue %[[V990]], %[[V988]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V992:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V993:.*]] = llvm.insertvalue %[[V992]], %[[V991]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V994:.*]] = llvm.insertvalue %[[V961]], %[[V993]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V995:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V996:.*]] = llvm.insertvalue %[[V995]], %[[V994]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V997:.*]] = llvm.insertvalue %[[V958]], %[[V996]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V998:.*]] = llvm.mul %[[V995]], %[[V961]] : i64
// CHECK-NEXT:    %[[V999:.*]] = llvm.insertvalue %[[V998]], %[[V997]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1000:.*]] = llvm.insertvalue %[[V955]], %[[V999]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1001:.*]] = llvm.mul %[[V998]], %[[V958]] : i64
// CHECK-NEXT:    %[[V1002:.*]] = llvm.insertvalue %[[V1001]], %[[V1000]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1003:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1004:.*]] = llvm.extractvalue %[[V985]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1005:.*]] = llvm.insertvalue %[[V1004]], %[[V1003]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1006:.*]] = llvm.extractvalue %[[V985]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1007:.*]] = llvm.getelementptr %[[V1006]][%[[V969]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1008:.*]] = llvm.insertvalue %[[V1007]], %[[V1005]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1009:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1010:.*]] = llvm.insertvalue %[[V1009]], %[[V1008]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1011:.*]] = llvm.insertvalue %[[V964]], %[[V1010]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1012:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1013:.*]] = llvm.insertvalue %[[V1012]], %[[V1011]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1014:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1015:.*]] = llvm.insertvalue %[[V1014]], %[[V1013]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1016:.*]] = llvm.mul %[[V1012]], %[[V964]] : i64
// CHECK-NEXT:    %[[V1017:.*]] = llvm.insertvalue %[[V1016]], %[[V1015]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1018:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1019:.*]] = llvm.extractvalue %[[V985]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1020:.*]] = llvm.insertvalue %[[V1019]], %[[V1018]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1021:.*]] = llvm.extractvalue %[[V985]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1022:.*]] = llvm.getelementptr %[[V1021]][%[[V974]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1023:.*]] = llvm.insertvalue %[[V1022]], %[[V1020]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1024:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1025:.*]] = llvm.insertvalue %[[V1024]], %[[V1023]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1026:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1027:.*]] = llvm.insertvalue %[[V1026]], %[[V1025]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1028:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1029:.*]] = llvm.insertvalue %[[V1028]], %[[V1027]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1030:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1031:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1032:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1033:.*]] = llvm.getelementptr %[[V1032]][%[[V1030]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %[[V1034:.*]] = llvm.ptrtoint %[[V1033]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1035:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1036:.*]] = llvm.add %[[V1034]], %[[V1035]] : i64
// CHECK-NEXT:    %[[V1037:.*]] = llvm.call @malloc(%[[V1036]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1038:.*]] = llvm.ptrtoint %[[V1037]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1039:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1040:.*]] = llvm.sub %[[V1035]], %[[V1039]] : i64
// CHECK-NEXT:    %[[V1041:.*]] = llvm.add %[[V1038]], %[[V1040]] : i64
// CHECK-NEXT:    %[[V1042:.*]] = llvm.urem %[[V1041]], %[[V1035]] : i64
// CHECK-NEXT:    %[[V1043:.*]] = llvm.sub %[[V1041]], %[[V1042]] : i64
// CHECK-NEXT:    %[[V1044:.*]] = llvm.inttoptr %[[V1043]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1045:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1046:.*]] = llvm.insertvalue %[[V1037]], %[[V1045]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1047:.*]] = llvm.insertvalue %[[V1044]], %[[V1046]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1048:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1049:.*]] = llvm.insertvalue %[[V1048]], %[[V1047]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1050:.*]] = llvm.insertvalue %[[V1030]], %[[V1049]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1051:.*]] = llvm.insertvalue %[[V1031]], %[[V1050]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1052:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1053:.*]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1054:.*]] = llvm.alloca %[[V1052]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1055:.*]] = llvm.extractvalue %[[V761]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1056:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V1057:.*]] = llvm.getelementptr %[[V1054]][%[[V1056]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1055]], %[[V1057]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1058:.*]] = llvm.extractvalue %[[V761]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1059:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V1060:.*]] = llvm.getelementptr %[[V1054]][%[[V1059]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1058]], %[[V1060]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1061:.*]] = llvm.extractvalue %[[V761]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1062:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V1063:.*]] = llvm.getelementptr %[[V1054]][%[[V1062]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1061]], %[[V1063]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1064:.*]] = llvm.alloca %[[V1052]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1065:.*]] = llvm.extractvalue %[[V1002]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1066:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V1067:.*]] = llvm.getelementptr %[[V1064]][%[[V1066]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1065]], %[[V1067]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1068:.*]] = llvm.extractvalue %[[V1002]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1069:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V1070:.*]] = llvm.getelementptr %[[V1064]][%[[V1069]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1068]], %[[V1070]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1071:.*]] = llvm.extractvalue %[[V1002]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1072:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V1073:.*]] = llvm.getelementptr %[[V1064]][%[[V1072]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1071]], %[[V1073]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1074:.*]] = llvm.extractvalue %[[V761]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1075:.*]] = llvm.extractvalue %[[V1002]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1076:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1077:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1078:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:    %[[V1079:.*]] = llvm.call @wrap_expand(%[[ARG0]], %[[V1074]], %[[V1053]], %[[V1075]], %[[V1054]], %[[V1076]], %[[V1064]], %[[V1077]], %[[V1078]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V1080:.*]] = llvm.extractvalue %[[V516]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1080]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1081:.*]] = llvm.extractvalue %[[V1002]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1082:.*]] = llvm.extractvalue %[[V1002]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1083:.*]] = llvm.extractvalue %[[V1002]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1084:.*]] = llvm.mul %[[V1081]], %[[V1082]] : i64
// CHECK-NEXT:    %[[V1085:.*]] = llvm.mul %[[V1084]], %[[V1083]] : i64
// CHECK-NEXT:    %[[V1086:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1087:.*]] = llvm.extractvalue %[[V1017]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1088:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1089:.*]] = llvm.alloca %[[V1088]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1090:.*]] = llvm.getelementptr %[[V1089]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1081]], %[[V1090]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1091:.*]] = llvm.getelementptr %[[V1089]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1082]], %[[V1091]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1092:.*]] = llvm.getelementptr %[[V1089]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1083]], %[[V1092]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1093:.*]] = llvm.extractvalue %[[V1002]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1094:.*]] = llvm.extractvalue %[[V1017]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1095:.*]] = llvm.extractvalue %[[V1029]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1096:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1097:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:    %[[V1098:.*]] = llvm.call @wrap_nonzero(%[[ARG0]], %[[V1093]], %[[V1094]], %[[V1095]], %[[V1085]], %[[V1096]], %[[V1089]], %[[V1087]], %[[V1097]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V1099:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1100:.*]] = llvm.getelementptr %[[V1099]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK-NEXT:    %[[V1101:.*]] = llvm.ptrtoint %[[V1100]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1102:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1103:.*]] = llvm.mul %[[V1101]], %[[V1102]] : i64
// CHECK-NEXT:    %[[V1104:.*]] = llvm.extractvalue %[[V1051]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1105:.*]] = llvm.extractvalue %[[V1029]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1106:.*]] = llvm.call @wrap_copy_d2h(%[[ARG0]], %[[V1104]], %[[V1105]], %[[V1103]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT:    %[[V1107:.*]] = llvm.extractvalue %[[V901]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1107]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1108:.*]] = llvm.extractvalue %[[V948]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1108]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1109:.*]] = llvm.extractvalue %[[V812]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1109]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1110:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1111:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1112:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1113:.*]] = llvm.getelementptr %[[V1112]][%[[V1110]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1114:.*]] = llvm.ptrtoint %[[V1113]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1115:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1116:.*]] = llvm.add %[[V1114]], %[[V1115]] : i64
// CHECK-NEXT:    %[[V1117:.*]] = llvm.call @malloc(%[[V1116]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1118:.*]] = llvm.ptrtoint %[[V1117]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1119:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1120:.*]] = llvm.sub %[[V1115]], %[[V1119]] : i64
// CHECK-NEXT:    %[[V1121:.*]] = llvm.add %[[V1118]], %[[V1120]] : i64
// CHECK-NEXT:    %[[V1122:.*]] = llvm.urem %[[V1121]], %[[V1115]] : i64
// CHECK-NEXT:    %[[V1123:.*]] = llvm.sub %[[V1121]], %[[V1122]] : i64
// CHECK-NEXT:    %[[V1124:.*]] = llvm.inttoptr %[[V1123]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1125:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1126:.*]] = llvm.insertvalue %[[V1117]], %[[V1125]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1127:.*]] = llvm.insertvalue %[[V1124]], %[[V1126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1128:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1129:.*]] = llvm.insertvalue %[[V1128]], %[[V1127]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1130:.*]] = llvm.insertvalue %[[V1110]], %[[V1129]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1131:.*]] = llvm.insertvalue %[[V1111]], %[[V1130]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1132:.*]] = llvm.extractvalue %[[V1131]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1133:.*]] = llvm.getelementptr inbounds|nuw %[[V1132]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V1133]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1134:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1135:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1136:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1137:.*]] = llvm.getelementptr %[[V1136]][%[[V1134]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1138:.*]] = llvm.ptrtoint %[[V1137]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1139:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1140:.*]] = llvm.add %[[V1138]], %[[V1139]] : i64
// CHECK-NEXT:    %[[V1141:.*]] = llvm.call @malloc(%[[V1140]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1142:.*]] = llvm.ptrtoint %[[V1141]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1143:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1144:.*]] = llvm.sub %[[V1139]], %[[V1143]] : i64
// CHECK-NEXT:    %[[V1145:.*]] = llvm.add %[[V1142]], %[[V1144]] : i64
// CHECK-NEXT:    %[[V1146:.*]] = llvm.urem %[[V1145]], %[[V1139]] : i64
// CHECK-NEXT:    %[[V1147:.*]] = llvm.sub %[[V1145]], %[[V1146]] : i64
// CHECK-NEXT:    %[[V1148:.*]] = llvm.inttoptr %[[V1147]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1149:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1150:.*]] = llvm.insertvalue %[[V1141]], %[[V1149]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1151:.*]] = llvm.insertvalue %[[V1148]], %[[V1150]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1152:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1153:.*]] = llvm.insertvalue %[[V1152]], %[[V1151]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1154:.*]] = llvm.insertvalue %[[V1134]], %[[V1153]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1155:.*]] = llvm.insertvalue %[[V1135]], %[[V1154]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1156:.*]] = llvm.extractvalue %[[V1155]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1157:.*]] = llvm.getelementptr inbounds|nuw %[[V1156]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V23]], %[[V1157]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1158:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1159:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1160:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1161:.*]] = llvm.getelementptr %[[V1160]][%[[V1158]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1162:.*]] = llvm.ptrtoint %[[V1161]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1163:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1164:.*]] = llvm.add %[[V1162]], %[[V1163]] : i64
// CHECK-NEXT:    %[[V1165:.*]] = llvm.call @malloc(%[[V1164]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1166:.*]] = llvm.ptrtoint %[[V1165]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1167:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1168:.*]] = llvm.sub %[[V1163]], %[[V1167]] : i64
// CHECK-NEXT:    %[[V1169:.*]] = llvm.add %[[V1166]], %[[V1168]] : i64
// CHECK-NEXT:    %[[V1170:.*]] = llvm.urem %[[V1169]], %[[V1163]] : i64
// CHECK-NEXT:    %[[V1171:.*]] = llvm.sub %[[V1169]], %[[V1170]] : i64
// CHECK-NEXT:    %[[V1172:.*]] = llvm.inttoptr %[[V1171]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1173:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1174:.*]] = llvm.insertvalue %[[V1165]], %[[V1173]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1175:.*]] = llvm.insertvalue %[[V1172]], %[[V1174]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1176:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1177:.*]] = llvm.insertvalue %[[V1176]], %[[V1175]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1178:.*]] = llvm.insertvalue %[[V1158]], %[[V1177]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1179:.*]] = llvm.insertvalue %[[V1159]], %[[V1178]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1180:.*]] = llvm.extractvalue %[[V1051]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1181:.*]] = llvm.getelementptr inbounds|nuw %[[V1180]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %[[V1182:.*]] = llvm.load %[[V1181]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[V1183:.*]] = llvm.extractvalue %[[V1051]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1183]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1184:.*]] = llvm.sext %[[V1182]] : i32 to i64
// CHECK-NEXT:    %[[V1185:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V1186:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1187:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1188:.*]] = llvm.getelementptr %[[V1187]][%[[V1185]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1189:.*]] = llvm.ptrtoint %[[V1188]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1190:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1191:.*]] = llvm.add %[[V1189]], %[[V1190]] : i64
// CHECK-NEXT:    %[[V1192:.*]] = llvm.call @malloc(%[[V1191]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1193:.*]] = llvm.ptrtoint %[[V1192]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1194:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1195:.*]] = llvm.sub %[[V1190]], %[[V1194]] : i64
// CHECK-NEXT:    %[[V1196:.*]] = llvm.add %[[V1193]], %[[V1195]] : i64
// CHECK-NEXT:    %[[V1197:.*]] = llvm.urem %[[V1196]], %[[V1190]] : i64
// CHECK-NEXT:    %[[V1198:.*]] = llvm.sub %[[V1196]], %[[V1197]] : i64
// CHECK-NEXT:    %[[V1199:.*]] = llvm.inttoptr %[[V1198]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1200:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1201:.*]] = llvm.insertvalue %[[V1192]], %[[V1200]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1202:.*]] = llvm.insertvalue %[[V1199]], %[[V1201]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1203:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1204:.*]] = llvm.insertvalue %[[V1203]], %[[V1202]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1205:.*]] = llvm.insertvalue %[[V1185]], %[[V1204]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1206:.*]] = llvm.insertvalue %[[V1186]], %[[V1205]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1207:.*]] = llvm.extractvalue %[[V1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1208:.*]] = llvm.getelementptr inbounds|nuw %[[V1207]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V49]], %[[V1208]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1209:.*]] = llvm.extractvalue %[[V1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1210:.*]] = llvm.getelementptr inbounds|nuw %[[V1209]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1184]], %[[V1210]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1211:.*]] = llvm.extractvalue %[[V1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1212:.*]] = llvm.getelementptr inbounds|nuw %[[V1211]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1213:.*]] = llvm.load %[[V1212]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1214:.*]] = llvm.extractvalue %[[V1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1215:.*]] = llvm.getelementptr inbounds|nuw %[[V1214]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1216:.*]] = llvm.load %[[V1215]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1217:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V1218:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1219:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1220:.*]] = llvm.getelementptr %[[V1219]][%[[V1217]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1221:.*]] = llvm.ptrtoint %[[V1220]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1222:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1223:.*]] = llvm.add %[[V1221]], %[[V1222]] : i64
// CHECK-NEXT:    %[[V1224:.*]] = llvm.call @malloc(%[[V1223]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1225:.*]] = llvm.ptrtoint %[[V1224]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1226:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1227:.*]] = llvm.sub %[[V1222]], %[[V1226]] : i64
// CHECK-NEXT:    %[[V1228:.*]] = llvm.add %[[V1225]], %[[V1227]] : i64
// CHECK-NEXT:    %[[V1229:.*]] = llvm.urem %[[V1228]], %[[V1222]] : i64
// CHECK-NEXT:    %[[V1230:.*]] = llvm.sub %[[V1228]], %[[V1229]] : i64
// CHECK-NEXT:    %[[V1231:.*]] = llvm.inttoptr %[[V1230]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1232:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1233:.*]] = llvm.insertvalue %[[V1224]], %[[V1232]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1234:.*]] = llvm.insertvalue %[[V1231]], %[[V1233]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1235:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1236:.*]] = llvm.insertvalue %[[V1235]], %[[V1234]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1237:.*]] = llvm.insertvalue %[[V1217]], %[[V1236]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1238:.*]] = llvm.insertvalue %[[V1218]], %[[V1237]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1239:.*]] = llvm.extractvalue %[[V1238]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1240:.*]] = llvm.getelementptr inbounds|nuw %[[V1239]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1213]], %[[V1240]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1241:.*]] = llvm.extractvalue %[[V1238]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1242:.*]] = llvm.getelementptr inbounds|nuw %[[V1241]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1216]], %[[V1242]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1243:.*]] = llvm.extractvalue %[[V1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1244:.*]] = llvm.getelementptr inbounds|nuw %[[V1243]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1245:.*]] = llvm.load %[[V1244]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1246:.*]] = llvm.extractvalue %[[V1238]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1247:.*]] = llvm.getelementptr inbounds|nuw %[[V1246]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1248:.*]] = llvm.load %[[V1247]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1249:.*]] = llvm.mul %[[V1245]], %[[V17]] : i64
// CHECK-NEXT:    %[[V1250:.*]] = llvm.mul %[[V1248]], %[[V17]] : i64
// CHECK-NEXT:    %[[V1251:.*]] = llvm.intr.umax(%[[V1249]], %[[V1250]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1252:.*]] = llvm.add %[[V1251]], %[[V20]] : i64
// CHECK-NEXT:    %[[V1253:.*]] = llvm.udiv %[[V1252]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1254:.*]] = llvm.mul %[[V1253]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1255:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT:    %[[V1256:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V1255]], %[[V1254]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V1257:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1258:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1259:.*]] = llvm.insertvalue %[[V1256]], %[[V1258]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1260:.*]] = llvm.insertvalue %[[V1256]], %[[V1259]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1261:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1262:.*]] = llvm.insertvalue %[[V1261]], %[[V1260]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1263:.*]] = llvm.insertvalue %[[V1254]], %[[V1262]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1264:.*]] = llvm.insertvalue %[[V1257]], %[[V1263]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1265:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1266:.*]] = llvm.extractvalue %[[V1264]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1267:.*]] = llvm.insertvalue %[[V1266]], %[[V1265]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1268:.*]] = llvm.extractvalue %[[V1264]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1269:.*]] = llvm.getelementptr %[[V1268]][%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1270:.*]] = llvm.insertvalue %[[V1269]], %[[V1267]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1271:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1272:.*]] = llvm.insertvalue %[[V1271]], %[[V1270]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1273:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1274:.*]] = llvm.insertvalue %[[V1273]], %[[V1272]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1275:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1276:.*]] = llvm.insertvalue %[[V1275]], %[[V1274]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1277:.*]] = llvm.insertvalue %[[V1248]], %[[V1276]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1278:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1279:.*]] = llvm.insertvalue %[[V1278]], %[[V1277]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1280:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V1281:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1282:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1283:.*]] = llvm.getelementptr %[[V1282]][%[[V1280]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1284:.*]] = llvm.ptrtoint %[[V1283]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1285:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1286:.*]] = llvm.add %[[V1284]], %[[V1285]] : i64
// CHECK-NEXT:    %[[V1287:.*]] = llvm.call @malloc(%[[V1286]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1288:.*]] = llvm.ptrtoint %[[V1287]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1289:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1290:.*]] = llvm.sub %[[V1285]], %[[V1289]] : i64
// CHECK-NEXT:    %[[V1291:.*]] = llvm.add %[[V1288]], %[[V1290]] : i64
// CHECK-NEXT:    %[[V1292:.*]] = llvm.urem %[[V1291]], %[[V1285]] : i64
// CHECK-NEXT:    %[[V1293:.*]] = llvm.sub %[[V1291]], %[[V1292]] : i64
// CHECK-NEXT:    %[[V1294:.*]] = llvm.inttoptr %[[V1293]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1295:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1296:.*]] = llvm.insertvalue %[[V1287]], %[[V1295]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1297:.*]] = llvm.insertvalue %[[V1294]], %[[V1296]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1298:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1299:.*]] = llvm.insertvalue %[[V1298]], %[[V1297]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1300:.*]] = llvm.insertvalue %[[V1280]], %[[V1299]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1301:.*]] = llvm.insertvalue %[[V1281]], %[[V1300]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1302:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1303:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1304:.*]] = llvm.getelementptr %[[V1303]][%[[V1302]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1305:.*]] = llvm.ptrtoint %[[V1304]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1306:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1307:.*]] = llvm.add %[[V1305]], %[[V1306]] : i64
// CHECK-NEXT:    %[[V1308:.*]] = llvm.call @malloc(%[[V1307]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1309:.*]] = llvm.ptrtoint %[[V1308]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1310:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1311:.*]] = llvm.sub %[[V1306]], %[[V1310]] : i64
// CHECK-NEXT:    %[[V1312:.*]] = llvm.add %[[V1309]], %[[V1311]] : i64
// CHECK-NEXT:    %[[V1313:.*]] = llvm.urem %[[V1312]], %[[V1306]] : i64
// CHECK-NEXT:    %[[V1314:.*]] = llvm.sub %[[V1312]], %[[V1313]] : i64
// CHECK-NEXT:    %[[V1315:.*]] = llvm.inttoptr %[[V1314]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1316:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1317:.*]] = llvm.insertvalue %[[V1308]], %[[V1316]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1318:.*]] = llvm.insertvalue %[[V1315]], %[[V1317]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1319:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1320:.*]] = llvm.insertvalue %[[V1319]], %[[V1318]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1321:.*]] = llvm.extractvalue %[[V1017]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1322:.*]] = llvm.extractvalue %[[V1017]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1323:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1324:.*]] = llvm.insertvalue %[[V1321]], %[[V1323]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1325:.*]] = llvm.insertvalue %[[V1322]], %[[V1324]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1326:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1327:.*]] = llvm.insertvalue %[[V1326]], %[[V1325]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1328:.*]] = llvm.extractvalue %[[V1017]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1329:.*]] = llvm.extractvalue %[[V1017]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1330:.*]] = llvm.extractvalue %[[V1017]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1331:.*]] = llvm.extractvalue %[[V1017]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1332:.*]] = llvm.extractvalue %[[V1017]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1333:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1334:.*]] = llvm.extractvalue %[[V1327]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1335:.*]] = llvm.extractvalue %[[V1327]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1336:.*]] = llvm.insertvalue %[[V1334]], %[[V1333]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1337:.*]] = llvm.insertvalue %[[V1335]], %[[V1336]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1338:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1339:.*]] = llvm.insertvalue %[[V1338]], %[[V1337]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1340:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1341:.*]] = llvm.insertvalue %[[V1340]], %[[V1339]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1342:.*]] = llvm.insertvalue %[[V1331]], %[[V1341]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1343:.*]] = llvm.insertvalue %[[V1245]], %[[V1342]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1344:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1345:.*]] = llvm.insertvalue %[[V1344]], %[[V1343]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1346:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1347:.*]] = llvm.extractvalue %[[V1345]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1348:.*]] = llvm.mul %[[V1346]], %[[V1347]] : i64
// CHECK-NEXT:    %[[V1349:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1350:.*]] = llvm.alloca %[[V1349]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1351:.*]] = llvm.getelementptr %[[V1350]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1346]], %[[V1351]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1352:.*]] = llvm.getelementptr %[[V1350]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1347]], %[[V1352]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1353:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1354:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1355:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1356:.*]] = llvm.alloca %[[V1355]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1357:.*]] = llvm.getelementptr %[[V1356]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1353]], %[[V1357]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1358:.*]] = llvm.getelementptr %[[V1356]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1354]], %[[V1358]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1359:.*]] = llvm.extractvalue %[[V1345]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1360:.*]] = llvm.extractvalue %[[V1279]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1361:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V1362:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %[[V1363:.*]] = llvm.call @wrap_transpose(%[[ARG0]], %[[V1359]], %[[V1360]], %[[V1361]], %[[V1350]], %[[V1356]], %[[V1348]], %[[V1362]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V1364:.*]] = llvm.extractvalue %[[V1301]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1365:.*]] = llvm.getelementptr inbounds|nuw %[[V1364]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1248]], %[[V1365]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1366:.*]] = llvm.extractvalue %[[V1301]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1367:.*]] = llvm.getelementptr inbounds|nuw %[[V1366]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V16]], %[[V1367]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1368:.*]] = llvm.extractvalue %[[V1301]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1369:.*]] = llvm.getelementptr inbounds|nuw %[[V1368]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1370:.*]] = llvm.load %[[V1369]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1371:.*]] = llvm.extractvalue %[[V1320]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    llvm.store %[[V1370]], %[[V1371]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1372:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1373:.*]] = llvm.extractvalue %[[V1320]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1374:.*]] = llvm.extractvalue %[[V1320]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1375:.*]] = llvm.insertvalue %[[V1373]], %[[V1372]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1376:.*]] = llvm.insertvalue %[[V1374]], %[[V1375]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1377:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1378:.*]] = llvm.insertvalue %[[V1377]], %[[V1376]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1379:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1380:.*]] = llvm.insertvalue %[[V1379]], %[[V1378]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1381:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1382:.*]] = llvm.insertvalue %[[V1381]], %[[V1380]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1383:.*]] = llvm.extractvalue %[[V1206]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1383]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1384:.*]] = llvm.extractvalue %[[V1238]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1384]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1385:.*]] = llvm.extractvalue %[[V1301]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1385]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1386:.*]] = llvm.extractvalue %[[V1155]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1386]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1387:.*]] = llvm.extractvalue %[[V1179]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1387]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1388:.*]] = llvm.extractvalue %[[V1131]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1388]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1389:.*]] = llvm.extractvalue %[[V540]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1390:.*]] = llvm.extractvalue %[[V1382]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1391:.*]] = llvm.getelementptr inbounds|nuw %[[V1390]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1392:.*]] = llvm.load %[[V1391]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1393:.*]] = llvm.icmp "slt" %[[V1392]], %[[V48]] : i64
// CHECK-NEXT:    %[[V1394:.*]] = llvm.add %[[V1392]], %[[V1389]] : i64
// CHECK-NEXT:    %[[V1395:.*]] = llvm.select %[[V1393]], %[[V1394]], %[[V1392]] : i1, i64
// CHECK-NEXT:    %[[V1396:.*]] = llvm.intr.smin(%[[V1389]], %[[V48]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1397:.*]] = llvm.intr.smax(%[[V1395]], %[[V48]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1398:.*]] = llvm.intr.smin(%[[V1397]], %[[V1389]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1399:.*]] = llvm.sub %[[V1398]], %[[V1396]] : i64
// CHECK-NEXT:    %[[V1400:.*]] = llvm.intr.smax(%[[V1399]], %[[V48]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1401:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1402:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1403:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1404:.*]] = llvm.getelementptr %[[V1403]][%[[V1401]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1405:.*]] = llvm.ptrtoint %[[V1404]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1406:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1407:.*]] = llvm.add %[[V1405]], %[[V1406]] : i64
// CHECK-NEXT:    %[[V1408:.*]] = llvm.call @malloc(%[[V1407]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1409:.*]] = llvm.ptrtoint %[[V1408]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1410:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1411:.*]] = llvm.sub %[[V1406]], %[[V1410]] : i64
// CHECK-NEXT:    %[[V1412:.*]] = llvm.add %[[V1409]], %[[V1411]] : i64
// CHECK-NEXT:    %[[V1413:.*]] = llvm.urem %[[V1412]], %[[V1406]] : i64
// CHECK-NEXT:    %[[V1414:.*]] = llvm.sub %[[V1412]], %[[V1413]] : i64
// CHECK-NEXT:    %[[V1415:.*]] = llvm.inttoptr %[[V1414]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1416:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1417:.*]] = llvm.insertvalue %[[V1408]], %[[V1416]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1418:.*]] = llvm.insertvalue %[[V1415]], %[[V1417]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1419:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1420:.*]] = llvm.insertvalue %[[V1419]], %[[V1418]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1421:.*]] = llvm.insertvalue %[[V1401]], %[[V1420]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1422:.*]] = llvm.insertvalue %[[V1402]], %[[V1421]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1423:.*]] = llvm.extractvalue %[[V1422]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1424:.*]] = llvm.getelementptr inbounds|nuw %[[V1423]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1400]], %[[V1424]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1425:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1426:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1427:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1428:.*]] = llvm.getelementptr %[[V1427]][%[[V1425]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1429:.*]] = llvm.ptrtoint %[[V1428]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1430:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1431:.*]] = llvm.add %[[V1429]], %[[V1430]] : i64
// CHECK-NEXT:    %[[V1432:.*]] = llvm.call @malloc(%[[V1431]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1433:.*]] = llvm.ptrtoint %[[V1432]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1434:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1435:.*]] = llvm.sub %[[V1430]], %[[V1434]] : i64
// CHECK-NEXT:    %[[V1436:.*]] = llvm.add %[[V1433]], %[[V1435]] : i64
// CHECK-NEXT:    %[[V1437:.*]] = llvm.urem %[[V1436]], %[[V1430]] : i64
// CHECK-NEXT:    %[[V1438:.*]] = llvm.sub %[[V1436]], %[[V1437]] : i64
// CHECK-NEXT:    %[[V1439:.*]] = llvm.inttoptr %[[V1438]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1440:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1441:.*]] = llvm.insertvalue %[[V1432]], %[[V1440]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1442:.*]] = llvm.insertvalue %[[V1439]], %[[V1441]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1443:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1444:.*]] = llvm.insertvalue %[[V1443]], %[[V1442]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1445:.*]] = llvm.insertvalue %[[V1425]], %[[V1444]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1446:.*]] = llvm.insertvalue %[[V1426]], %[[V1445]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1447:.*]] = llvm.extractvalue %[[V1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1448:.*]] = llvm.getelementptr inbounds|nuw %[[V1447]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V439]], %[[V1448]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1449:.*]] = llvm.extractvalue %[[V1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1450:.*]] = llvm.getelementptr inbounds|nuw %[[V1449]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V442]], %[[V1450]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1451:.*]] = llvm.extractvalue %[[V1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1452:.*]] = llvm.getelementptr inbounds|nuw %[[V1451]][%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V24]], %[[V1452]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1453:.*]] = llvm.extractvalue %[[V1422]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1454:.*]] = llvm.getelementptr inbounds|nuw %[[V1453]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1455:.*]] = llvm.load %[[V1454]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1456:.*]] = llvm.mul %[[V1455]], %[[V23]] : i64
// CHECK-NEXT:    %[[V1457:.*]] = llvm.add %[[V1456]], %[[V20]] : i64
// CHECK-NEXT:    %[[V1458:.*]] = llvm.udiv %[[V1457]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1459:.*]] = llvm.mul %[[V1458]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1460:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT:    %[[V1461:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V1460]], %[[V1459]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V1462:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1463:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1464:.*]] = llvm.insertvalue %[[V1461]], %[[V1463]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1465:.*]] = llvm.insertvalue %[[V1461]], %[[V1464]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1466:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1467:.*]] = llvm.insertvalue %[[V1466]], %[[V1465]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1468:.*]] = llvm.insertvalue %[[V1459]], %[[V1467]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1469:.*]] = llvm.insertvalue %[[V1462]], %[[V1468]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1470:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1471:.*]] = llvm.extractvalue %[[V1469]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1472:.*]] = llvm.insertvalue %[[V1471]], %[[V1470]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1473:.*]] = llvm.extractvalue %[[V1469]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1474:.*]] = llvm.getelementptr %[[V1473]][%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1475:.*]] = llvm.insertvalue %[[V1474]], %[[V1472]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1476:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1477:.*]] = llvm.insertvalue %[[V1476]], %[[V1475]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1478:.*]] = llvm.insertvalue %[[V1455]], %[[V1477]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1479:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1480:.*]] = llvm.insertvalue %[[V1479]], %[[V1478]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1481:.*]] = llvm.extractvalue %[[V1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1482:.*]] = llvm.getelementptr inbounds|nuw %[[V1481]][%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1483:.*]] = llvm.load %[[V1482]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1484:.*]] = llvm.extractvalue %[[V1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1485:.*]] = llvm.getelementptr inbounds|nuw %[[V1484]][%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1486:.*]] = llvm.load %[[V1485]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1487:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V1488:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1489:.*]] = llvm.mul %[[V1487]], %[[V1486]] : i64
// CHECK-NEXT:    %[[V1490:.*]] = llvm.mul %[[V1489]], %[[V1483]] : i64
// CHECK-NEXT:    %[[V1491:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1492:.*]] = llvm.getelementptr %[[V1491]][%[[V1490]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-NEXT:    %[[V1493:.*]] = llvm.ptrtoint %[[V1492]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1494:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1495:.*]] = llvm.alloca %[[V1494]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1496:.*]] = llvm.getelementptr %[[V1495]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1483]], %[[V1496]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1497:.*]] = llvm.getelementptr %[[V1495]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1486]], %[[V1497]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1498:.*]] = llvm.getelementptr %[[V1495]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1487]], %[[V1498]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1499:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1500:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1501:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V1502:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V1499]], %[[V1495]], %[[V1500]], %[[V1501]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1503:.*]] = llvm.addrspacecast %[[V1502]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V1504:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1505:.*]] = llvm.insertvalue %[[V1503]], %[[V1504]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1506:.*]] = llvm.insertvalue %[[V1503]], %[[V1505]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1507:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1508:.*]] = llvm.insertvalue %[[V1507]], %[[V1506]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1509:.*]] = llvm.insertvalue %[[V1483]], %[[V1508]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1510:.*]] = llvm.insertvalue %[[V1486]], %[[V1509]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1511:.*]] = llvm.insertvalue %[[V1487]], %[[V1510]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1512:.*]] = llvm.insertvalue %[[V1489]], %[[V1511]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1513:.*]] = llvm.insertvalue %[[V1487]], %[[V1512]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1514:.*]] = llvm.insertvalue %[[V1488]], %[[V1513]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1515:.*]] = llvm.extractvalue %[[V540]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1516:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1517:.*]] = llvm.alloca %[[V1516]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1518:.*]] = llvm.getelementptr %[[V1517]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1515]], %[[V1518]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1519:.*]] = llvm.extractvalue %[[V1480]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1520:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1521:.*]] = llvm.alloca %[[V1520]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1522:.*]] = llvm.getelementptr %[[V1521]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1519]], %[[V1522]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1523:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1524:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1525:.*]] = llvm.alloca %[[V1524]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1526:.*]] = llvm.getelementptr %[[V1525]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1523]], %[[V1526]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1527:.*]] = llvm.extractvalue %[[V1382]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1528:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1529:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1530:.*]] = llvm.alloca %[[V1529]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1531:.*]] = llvm.getelementptr %[[V1530]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1528]], %[[V1531]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1532:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1533:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1534:.*]] = llvm.alloca %[[V1533]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1535:.*]] = llvm.getelementptr %[[V1534]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1532]], %[[V1535]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1536:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1537:.*]] = llvm.extractvalue %[[V540]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1538:.*]] = llvm.extractvalue %[[V1480]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1539:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1540:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1541:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1542:.*]] = llvm.call @wrap_slice(%[[ARG0]], %[[V1537]], %[[V1525]], %[[V1527]], %[[V1530]], %[[V1534]], %[[V1538]], %[[V1517]], %[[V1539]], %[[V1521]], %[[V1540]], %[[V1536]], %[[V1536]], %[[V1536]], %[[V1541]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V1543:.*]] = llvm.extractvalue %[[V1320]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    llvm.call @free(%[[V1543]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1544:.*]] = llvm.extractvalue %[[V494]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1545:.*]] = llvm.extractvalue %[[V494]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1546:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V1547:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1548:.*]] = llvm.alloca %[[V1547]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1549:.*]] = llvm.getelementptr %[[V1548]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1544]], %[[V1549]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1550:.*]] = llvm.getelementptr %[[V1548]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1545]], %[[V1550]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1551:.*]] = llvm.getelementptr %[[V1548]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1546]], %[[V1551]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1552:.*]] = llvm.extractvalue %[[V1279]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1553:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1554:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1555:.*]] = llvm.alloca %[[V1554]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1556:.*]] = llvm.getelementptr %[[V1555]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1552]], %[[V1556]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1557:.*]] = llvm.getelementptr %[[V1555]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1553]], %[[V1557]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1558:.*]] = llvm.extractvalue %[[V1480]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1559:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1560:.*]] = llvm.alloca %[[V1559]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1561:.*]] = llvm.getelementptr %[[V1560]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1558]], %[[V1561]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1562:.*]] = llvm.extractvalue %[[V1514]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1563:.*]] = llvm.extractvalue %[[V1514]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1564:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V1565:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1566:.*]] = llvm.alloca %[[V1565]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1567:.*]] = llvm.getelementptr %[[V1566]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1562]], %[[V1567]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1568:.*]] = llvm.getelementptr %[[V1566]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1563]], %[[V1568]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1569:.*]] = llvm.getelementptr %[[V1566]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1564]], %[[V1569]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1570:.*]] = llvm.extractvalue %[[V494]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1571:.*]] = llvm.extractvalue %[[V1279]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1572:.*]] = llvm.extractvalue %[[V1480]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1573:.*]] = llvm.extractvalue %[[V1514]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1574:.*]] = llvm.mlir.zero : !llvm.ptr<1>
// CHECK-NEXT:    %[[V1575:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1576:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V1577:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1578:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1579:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1580:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1581:.*]] = llvm.call @wrap_scatter_nd(%[[ARG0]], %[[V1570]], %[[V1571]], %[[V1572]], %[[V1573]], %[[V1574]], %[[V1548]], %[[V1575]], %[[V1555]], %[[V1576]], %[[V1560]], %[[V1577]], %[[V1566]], %[[V1578]], %[[V1579]], %[[V1580]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V1582:.*]] = llvm.extractvalue %[[V1422]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1582]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1583:.*]] = llvm.extractvalue %[[V1446]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1583]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V1514]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:  }

// generate-interface emits the public C entry points used by the runtime.
// CHECK-LABEL: llvm.func @inference_init(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr, %[[ARG2:[^,]*]]: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    %[[MLIR_0:.*]] = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
// CHECK-NEXT:    %[[MLIR_1:.*]] = llvm.mlir.constant(312 : i64) : i64
// CHECK-NEXT:    %[[CALL_0:.*]] = llvm.call @hipdnn_ep_state_init_with_fs(%[[ARG0]], %[[ARG1]], %[[MLIR_0]], %[[MLIR_1]], %[[ARG2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return %[[CALL_0]] : i32
// CHECK-NEXT:  }
// CHECK-LABEL: llvm.func @inference_compute(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    llvm.call @hipdnn_ep_runtime_begin_compute(%[[ARG0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[MLIR_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[ALLOCA_0:.*]] = llvm.alloca %[[MLIR_1]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[MLIR_2:.*]] = llvm.mlir.constant(48 : i64) : i64
// CHECK-NEXT:    %[[ALLOCA_1:.*]] = llvm.alloca %[[MLIR_2]] x i8 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[ALLOCA_2:.*]] = llvm.alloca %[[MLIR_2]] x i8 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[MLIR_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[MLIR_4:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[CALL_0:.*]] = llvm.call @hipdnn_ep_tensor_prepare_input(%[[ARG0]], %[[ARG1]], %[[MLIR_3]], %[[MLIR_4]], %[[ALLOCA_1]]) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[MLIR_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[ICMP_0:.*]] = llvm.icmp "ne" %[[CALL_0]], %[[MLIR_5]] : i32
// CHECK-NEXT:    llvm.cond_br %[[ICMP_0]], ^bb3, ^bb2
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:    %[[LOAD_0:.*]] = llvm.load %[[ALLOCA_0]] : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[ALLOCA_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[ALLOCA_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[LOAD_0]] : i32
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:    %[[MLIR_6:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[MLIR_7:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[CALL_1:.*]] = llvm.call @hipdnn_ep_tensor_prepare_input(%[[ARG0]], %[[ARG1]], %[[MLIR_6]], %[[MLIR_7]], %[[ALLOCA_2]]) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[MLIR_8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[ICMP_1:.*]] = llvm.icmp "ne" %[[CALL_1]], %[[MLIR_8]] : i32
// CHECK-NEXT:    llvm.cond_br %[[ICMP_1]], ^bb5, ^bb4
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:    llvm.store %[[CALL_0]], %[[ALLOCA_0]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb4:
// CHECK-NEXT:    %[[MLIR_9:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[ALLOCA_3:.*]] = llvm.alloca %[[MLIR_9]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[CALL_2:.*]] = llvm.call @hipdnn_ep_tensor_buffer_get_gpu_ptr(%[[ALLOCA_1]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[CALL_3:.*]] = llvm.call @hipdnn_ep_tensor_buffer_get_shape_ptr(%[[ALLOCA_1]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[ADDRSPACECAST_0:.*]] = llvm.addrspacecast %[[CALL_2]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[MLIR_10:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ADDRSPACECAST_0]], %[[MLIR_10]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ADDRSPACECAST_0]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[MLIR_11:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[MLIR_11]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[MLIR_12:.*]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_13:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[CALL_3]]{{\[}}%[[MLIR_13]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_1:.*]] = llvm.load %[[GETELEMENTPTR_0]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[LOAD_1]], %[[MLIR_12]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_14:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[CALL_3]]{{\[}}%[[MLIR_14]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_2:.*]] = llvm.load %[[GETELEMENTPTR_1]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[LOAD_2]], %[[INSERTVALUE_3]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[INSERTVALUE_4]], %[[INSERTVALUE_2]][3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[MLIR_15:.*]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_16:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[INSERTVALUE_6:.*]] = llvm.insertvalue %[[MLIR_16]], %[[MLIR_15]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_17:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_2:.*]] = llvm.getelementptr %[[CALL_3]]{{\[}}%[[MLIR_17]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_3:.*]] = llvm.load %[[GETELEMENTPTR_2]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[MUL_0:.*]] = llvm.mul %[[MLIR_16]], %[[LOAD_3]] : i64
// CHECK-NEXT:    %[[INSERTVALUE_7:.*]] = llvm.insertvalue %[[MUL_0]], %[[INSERTVALUE_6]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[INSERTVALUE_8:.*]] = llvm.insertvalue %[[INSERTVALUE_7]], %[[INSERTVALUE_5]][4] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[ALLOCA_4:.*]] = llvm.alloca %[[MLIR_1]] x !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %[[INSERTVALUE_8]], %[[ALLOCA_4]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
// CHECK-NEXT:    %[[MLIR_18:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_3:.*]] = llvm.getelementptr %[[ALLOCA_3]]{{\[}}%[[MLIR_18]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %[[ALLOCA_4]], %[[GETELEMENTPTR_3]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[CALL_4:.*]] = llvm.call @hipdnn_ep_tensor_buffer_get_gpu_ptr(%[[ALLOCA_2]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[CALL_5:.*]] = llvm.call @hipdnn_ep_tensor_buffer_get_shape_ptr(%[[ALLOCA_2]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[ADDRSPACECAST_1:.*]] = llvm.addrspacecast %[[CALL_4]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[MLIR_19:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERTVALUE_9:.*]] = llvm.insertvalue %[[ADDRSPACECAST_1]], %[[MLIR_19]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERTVALUE_10:.*]] = llvm.insertvalue %[[ADDRSPACECAST_1]], %[[INSERTVALUE_9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[MLIR_20:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[INSERTVALUE_11:.*]] = llvm.insertvalue %[[MLIR_20]], %[[INSERTVALUE_10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[MLIR_21:.*]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_22:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_4:.*]] = llvm.getelementptr %[[CALL_5]]{{\[}}%[[MLIR_22]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_4:.*]] = llvm.load %[[GETELEMENTPTR_4]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[INSERTVALUE_12:.*]] = llvm.insertvalue %[[LOAD_4]], %[[MLIR_21]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_23:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_5:.*]] = llvm.getelementptr %[[CALL_5]]{{\[}}%[[MLIR_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_5:.*]] = llvm.load %[[GETELEMENTPTR_5]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[INSERTVALUE_13:.*]] = llvm.insertvalue %[[LOAD_5]], %[[INSERTVALUE_12]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[INSERTVALUE_14:.*]] = llvm.insertvalue %[[INSERTVALUE_13]], %[[INSERTVALUE_11]][3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[MLIR_24:.*]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_25:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[INSERTVALUE_15:.*]] = llvm.insertvalue %[[MLIR_25]], %[[MLIR_24]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[MLIR_26:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_6:.*]] = llvm.getelementptr %[[CALL_5]]{{\[}}%[[MLIR_26]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[LOAD_6:.*]] = llvm.load %[[GETELEMENTPTR_6]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[MUL_1:.*]] = llvm.mul %[[MLIR_25]], %[[LOAD_6]] : i64
// CHECK-NEXT:    %[[INSERTVALUE_16:.*]] = llvm.insertvalue %[[MUL_1]], %[[INSERTVALUE_15]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[INSERTVALUE_17:.*]] = llvm.insertvalue %[[INSERTVALUE_16]], %[[INSERTVALUE_14]][4] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[ALLOCA_5:.*]] = llvm.alloca %[[MLIR_1]] x !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %[[INSERTVALUE_17]], %[[ALLOCA_5]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
// CHECK-NEXT:    %[[MLIR_27:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[GETELEMENTPTR_7:.*]] = llvm.getelementptr %[[ALLOCA_3]]{{\[}}%[[MLIR_27]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %[[ALLOCA_5]], %[[GETELEMENTPTR_7]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[CALL_6:.*]] = llvm.call @hipdnn_ep_state_reset_error_flag(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[MLIR_28:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[ICMP_2:.*]] = llvm.icmp "ne" %[[CALL_6]], %[[MLIR_28]] : i32
// CHECK-NEXT:    llvm.cond_br %[[ICMP_2]], ^bb7, ^bb6
// CHECK-NEXT:    ^bb5:
// CHECK-NEXT:    llvm.store %[[CALL_1]], %[[ALLOCA_0]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb6:
// CHECK-NEXT:    %[[CALL_7:.*]] = llvm.call @main_graph(%[[ARG0]], %[[ALLOCA_3]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[MLIR_29:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[ICMP_3:.*]] = llvm.icmp "ne" %[[CALL_7]], %[[MLIR_29]] : i32
// CHECK-NEXT:    llvm.cond_br %[[ICMP_3]], ^bb10, ^bb9
// CHECK-NEXT:    ^bb7:
// CHECK-NEXT:    llvm.store %[[CALL_6]], %[[ALLOCA_0]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb8:
// CHECK-NEXT:    %[[CALL_8:.*]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[MLIR_30:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[ICMP_4:.*]] = llvm.icmp "ne" %[[CALL_8]], %[[MLIR_30]] : i32
// CHECK-NEXT:    llvm.cond_br %[[ICMP_4]], ^bb12, ^bb11
// CHECK-NEXT:    ^bb9:
// CHECK-NEXT:    llvm.br ^bb8
// CHECK-NEXT:    ^bb10:
// CHECK-NEXT:    llvm.store %[[CALL_7]], %[[ALLOCA_0]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb11:
// CHECK-NEXT:    %[[CALL_9:.*]] = llvm.call @hipdnn_ep_state_read_and_clear_error_flag(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[MLIR_31:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[ICMP_5:.*]] = llvm.icmp "ne" %[[CALL_9]], %[[MLIR_31]] : i32
// CHECK-NEXT:    llvm.cond_br %[[ICMP_5]], ^bb14, ^bb13
// CHECK-NEXT:    ^bb12:
// CHECK-NEXT:    llvm.store %[[CALL_8]], %[[ALLOCA_0]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb13:
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[ALLOCA_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[ALLOCA_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[MLIR_0]] : i32
// CHECK-NEXT:    ^bb14:
// CHECK-NEXT:    llvm.store %[[CALL_9]], %[[ALLOCA_0]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  }
// CHECK-LABEL: llvm.func @inference_cleanup(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    %[[CALL_0:.*]] = llvm.call @hipdnn_ep_state_cleanup(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return %[[CALL_0]] : i32
// CHECK-NEXT:  }
// CHECK-LABEL: llvm.func @inference_get_metadata_json() -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    %[[MLIR_0:.*]] = llvm.mlir.addressof @__metadata_json : !llvm.ptr
// CHECK-NEXT:    llvm.return %[[MLIR_0]] : !llvm.ptr
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
