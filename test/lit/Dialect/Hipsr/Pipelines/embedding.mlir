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
// The embedding table lives in an external file, so the RUN line creates a
// file of the right length to map. Only the length matters; nothing reads the
// weights.

// RUN: %python %S/../../../Inputs/make_external_data.py %t/embedding.onnx.data 2034237440 && cd %t && hip-mlir-opt --onnx-dialect=modeled --hipsr-pipeline --mlir-elide-resource-strings-if-larger=32 %s | FileCheck %s

// CHECK-LABEL: module attributes {hip.constants_file = "constants.bin", hipdnn.constant_offsets = array<i64: 0, 64>, hipdnn.constant_sizes = array<i64: 8, 2034237440>, hipdnn.input_ranks = array<i64: 2, 2>} {
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_json(
// CHECK-NEXT: llvm.func @hipdnn_ep_state_cleanup(!llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__hipsr_input_ranks(dense<2> : tensor<2xi64>) {addr_space = 0 : i32} : !llvm.array<2 x i64>
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_compute(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_blob(
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.func @wrap_scatter_nd(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.func @wrap_slice(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT: llvm.func @wrap_transpose(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: llvm.func @wrap_copy_d2h(!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT: llvm.func @wrap_nonzero(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: llvm.func @wrap_expand(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: llvm.func @wrap_gather(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.func @wrap_equal(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: llvm.func @free(!llvm.ptr)
// CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-NEXT: llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: llvm.func private @main_graph(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 attributes {passthrough = ["noinline"]} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_2]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_2]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_2]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_7:.*]] = llvm.extractvalue %[[VAL_2]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_8:.*]] = llvm.extractvalue %[[VAL_2]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_9:.*]] = llvm.extractvalue %[[VAL_2]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_10:.*]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_13:.*]] = llvm.extractvalue %[[VAL_12]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_14:.*]] = llvm.extractvalue %[[VAL_12]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_15:.*]] = llvm.extractvalue %[[VAL_12]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_16:.*]] = llvm.extractvalue %[[VAL_12]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_17:.*]] = llvm.extractvalue %[[VAL_12]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_18:.*]] = llvm.extractvalue %[[VAL_12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_19:.*]] = llvm.extractvalue %[[VAL_12]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_20:.*]] = llvm.call @main_graph_internal(%arg0, %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_21:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %[[VAL_21]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func private @main_graph_internal(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: !llvm.ptr<1>, %arg9: !llvm.ptr<1>, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64) -> (!llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.insertvalue %arg1, %[[VAL_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.insertvalue %arg2, %[[VAL_1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.insertvalue %arg3, %[[VAL_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.insertvalue %arg4, %[[VAL_3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_5:.*]] = llvm.insertvalue %arg6, %[[VAL_4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_6:.*]] = llvm.insertvalue %arg5, %[[VAL_5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_7:.*]] = llvm.insertvalue %arg7, %[[VAL_6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_8:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_9:.*]] = llvm.insertvalue %arg8, %[[VAL_8]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_10:.*]] = llvm.insertvalue %arg9, %[[VAL_9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_11:.*]] = llvm.insertvalue %arg10, %[[VAL_10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_12:.*]] = llvm.insertvalue %arg11, %[[VAL_11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_13:.*]] = llvm.insertvalue %arg13, %[[VAL_12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_14:.*]] = llvm.insertvalue %arg12, %[[VAL_13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_15:.*]] = llvm.insertvalue %arg14, %[[VAL_14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_16:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_17:.*]] = llvm.mlir.constant(24 : index) : i64
// CHECK-NEXT: %[[VAL_18:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT: %[[VAL_19:.*]] = llvm.mlir.constant(8192 : index) : i64
// CHECK-NEXT: %[[VAL_20:.*]] = llvm.mlir.constant(255 : index) : i64
// CHECK-NEXT: %[[VAL_21:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT: %[[VAL_22:.*]] = llvm.mlir.constant(248320 : index) : i64
// CHECK-NEXT: %[[VAL_23:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_24:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT: %[[VAL_25:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_26:.*]] = llvm.call @hipdnn_ep_constant_get(%arg0, %[[VAL_25]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_27:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_26]], %[[VAL_27]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_26]], %[[VAL_28]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_30:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_31:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_29]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_32:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_33:.*]] = llvm.call @hipdnn_ep_constant_get(%arg0, %[[VAL_32]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_34:.*]] = llvm.mlir.constant(248320 : i64) : i64
// CHECK-NEXT: %[[VAL_35:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT: %[[VAL_36:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_37:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT: %[[VAL_38:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_39:.*]] = llvm.insertvalue %[[VAL_33]], %[[VAL_38]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_33]], %[[VAL_39]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_41:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_41]], %[[VAL_40]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_34]], %[[VAL_42]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_44:.*]] = llvm.insertvalue %[[VAL_35]], %[[VAL_43]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_45:.*]] = llvm.insertvalue %[[VAL_37]], %[[VAL_44]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_46:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_45]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_47:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_48:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_49:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_50:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_51:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_52:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_53:.*]] = llvm.getelementptr %[[VAL_52]][%[[VAL_50]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_54:.*]] = llvm.ptrtoint %[[VAL_53]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_55:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_56:.*]] = llvm.add %[[VAL_54]], %[[VAL_55]] : i64
// CHECK-NEXT: %[[VAL_57:.*]] = llvm.call @malloc(%[[VAL_56]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_58:.*]] = llvm.ptrtoint %[[VAL_57]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_59:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_60:.*]] = llvm.sub %[[VAL_55]], %[[VAL_59]] : i64
// CHECK-NEXT: %[[VAL_61:.*]] = llvm.add %[[VAL_58]], %[[VAL_60]] : i64
// CHECK-NEXT: %[[VAL_62:.*]] = llvm.urem %[[VAL_61]], %[[VAL_55]] : i64
// CHECK-NEXT: %[[VAL_63:.*]] = llvm.sub %[[VAL_61]], %[[VAL_62]] : i64
// CHECK-NEXT: %[[VAL_64:.*]] = llvm.inttoptr %[[VAL_63]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_65:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_66:.*]] = llvm.insertvalue %[[VAL_57]], %[[VAL_65]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_67:.*]] = llvm.insertvalue %[[VAL_64]], %[[VAL_66]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_68:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_69:.*]] = llvm.insertvalue %[[VAL_68]], %[[VAL_67]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_70:.*]] = llvm.insertvalue %[[VAL_50]], %[[VAL_69]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_71:.*]] = llvm.insertvalue %[[VAL_51]], %[[VAL_70]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_72:.*]] = llvm.extractvalue %[[VAL_71]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_73:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_72]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_49]], %[[VAL_73]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_74:.*]] = llvm.extractvalue %[[VAL_15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_75:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_76:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_77:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_78:.*]] = llvm.getelementptr %[[VAL_77]][%[[VAL_75]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_79:.*]] = llvm.ptrtoint %[[VAL_78]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_80:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_81:.*]] = llvm.add %[[VAL_79]], %[[VAL_80]] : i64
// CHECK-NEXT: %[[VAL_82:.*]] = llvm.call @malloc(%[[VAL_81]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_83:.*]] = llvm.ptrtoint %[[VAL_82]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_84:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_85:.*]] = llvm.sub %[[VAL_80]], %[[VAL_84]] : i64
// CHECK-NEXT: %[[VAL_86:.*]] = llvm.add %[[VAL_83]], %[[VAL_85]] : i64
// CHECK-NEXT: %[[VAL_87:.*]] = llvm.urem %[[VAL_86]], %[[VAL_80]] : i64
// CHECK-NEXT: %[[VAL_88:.*]] = llvm.sub %[[VAL_86]], %[[VAL_87]] : i64
// CHECK-NEXT: %[[VAL_89:.*]] = llvm.inttoptr %[[VAL_88]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_90:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_91:.*]] = llvm.insertvalue %[[VAL_82]], %[[VAL_90]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_92:.*]] = llvm.insertvalue %[[VAL_89]], %[[VAL_91]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_93:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_94:.*]] = llvm.insertvalue %[[VAL_93]], %[[VAL_92]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_95:.*]] = llvm.insertvalue %[[VAL_75]], %[[VAL_94]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_96:.*]] = llvm.insertvalue %[[VAL_76]], %[[VAL_95]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_97:.*]] = llvm.extractvalue %[[VAL_96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_98:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_97]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_74]], %[[VAL_98]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_99:.*]] = llvm.extractvalue %[[VAL_96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_100:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_99]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_24]], %[[VAL_100]] : i64, !llvm.ptr
// CHECK-NEXT: llvm.br ^bb1(%[[VAL_48]], %[[VAL_47]] : i64, i64)
// CHECK-NEXT: ^bb1(%[[VAL_101:.*]]: i64, %[[VAL_102:.*]]: i64):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT: %[[VAL_103:.*]] = llvm.icmp "slt" %[[VAL_101]], %[[VAL_23]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_103]], ^bb2, ^bb3
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %[[VAL_104:.*]] = llvm.extractvalue %[[VAL_96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_105:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_104]][%[[VAL_101]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_106:.*]] = llvm.load %[[VAL_105]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_107:.*]] = llvm.mul %[[VAL_106]], %[[VAL_102]] : i64
// CHECK-NEXT: %[[VAL_108:.*]] = llvm.add %[[VAL_101]], %[[VAL_47]] : i64
// CHECK-NEXT: llvm.br ^bb1(%[[VAL_108]], %[[VAL_107]] : i64, i64)
// CHECK-NEXT: ^bb3:  // pred: ^bb1
// CHECK-NEXT: %[[VAL_109:.*]] = llvm.extractvalue %[[VAL_96]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_109]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_110:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_111:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_112:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_113:.*]] = llvm.getelementptr %[[VAL_112]][%[[VAL_110]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_114:.*]] = llvm.ptrtoint %[[VAL_113]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_115:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_116:.*]] = llvm.add %[[VAL_114]], %[[VAL_115]] : i64
// CHECK-NEXT: %[[VAL_117:.*]] = llvm.call @malloc(%[[VAL_116]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_118:.*]] = llvm.ptrtoint %[[VAL_117]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_119:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_120:.*]] = llvm.sub %[[VAL_115]], %[[VAL_119]] : i64
// CHECK-NEXT: %[[VAL_121:.*]] = llvm.add %[[VAL_118]], %[[VAL_120]] : i64
// CHECK-NEXT: %[[VAL_122:.*]] = llvm.urem %[[VAL_121]], %[[VAL_115]] : i64
// CHECK-NEXT: %[[VAL_123:.*]] = llvm.sub %[[VAL_121]], %[[VAL_122]] : i64
// CHECK-NEXT: %[[VAL_124:.*]] = llvm.inttoptr %[[VAL_123]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_125:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_126:.*]] = llvm.insertvalue %[[VAL_117]], %[[VAL_125]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_127:.*]] = llvm.insertvalue %[[VAL_124]], %[[VAL_126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_128:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_129:.*]] = llvm.insertvalue %[[VAL_128]], %[[VAL_127]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_130:.*]] = llvm.insertvalue %[[VAL_110]], %[[VAL_129]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_131:.*]] = llvm.insertvalue %[[VAL_111]], %[[VAL_130]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_132:.*]] = llvm.extractvalue %[[VAL_131]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_133:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_132]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_102]], %[[VAL_133]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_134:.*]] = llvm.extractvalue %[[VAL_7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_135:.*]] = llvm.extractvalue %[[VAL_7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_136:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_137:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_138:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_139:.*]] = llvm.getelementptr %[[VAL_138]][%[[VAL_136]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_140:.*]] = llvm.ptrtoint %[[VAL_139]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_141:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_142:.*]] = llvm.add %[[VAL_140]], %[[VAL_141]] : i64
// CHECK-NEXT: %[[VAL_143:.*]] = llvm.call @malloc(%[[VAL_142]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_144:.*]] = llvm.ptrtoint %[[VAL_143]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_145:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_146:.*]] = llvm.sub %[[VAL_141]], %[[VAL_145]] : i64
// CHECK-NEXT: %[[VAL_147:.*]] = llvm.add %[[VAL_144]], %[[VAL_146]] : i64
// CHECK-NEXT: %[[VAL_148:.*]] = llvm.urem %[[VAL_147]], %[[VAL_141]] : i64
// CHECK-NEXT: %[[VAL_149:.*]] = llvm.sub %[[VAL_147]], %[[VAL_148]] : i64
// CHECK-NEXT: %[[VAL_150:.*]] = llvm.inttoptr %[[VAL_149]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_151:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_152:.*]] = llvm.insertvalue %[[VAL_143]], %[[VAL_151]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_153:.*]] = llvm.insertvalue %[[VAL_150]], %[[VAL_152]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_154:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_155:.*]] = llvm.insertvalue %[[VAL_154]], %[[VAL_153]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_156:.*]] = llvm.insertvalue %[[VAL_136]], %[[VAL_155]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_157:.*]] = llvm.insertvalue %[[VAL_137]], %[[VAL_156]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_158:.*]] = llvm.extractvalue %[[VAL_157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_159:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_158]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_134]], %[[VAL_159]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_160:.*]] = llvm.extractvalue %[[VAL_157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_161:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_160]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_135]], %[[VAL_161]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_162:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_163:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_164:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_165:.*]] = llvm.getelementptr %[[VAL_164]][%[[VAL_162]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_166:.*]] = llvm.ptrtoint %[[VAL_165]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_167:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_168:.*]] = llvm.add %[[VAL_166]], %[[VAL_167]] : i64
// CHECK-NEXT: %[[VAL_169:.*]] = llvm.call @malloc(%[[VAL_168]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_170:.*]] = llvm.ptrtoint %[[VAL_169]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_171:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_172:.*]] = llvm.sub %[[VAL_167]], %[[VAL_171]] : i64
// CHECK-NEXT: %[[VAL_173:.*]] = llvm.add %[[VAL_170]], %[[VAL_172]] : i64
// CHECK-NEXT: %[[VAL_174:.*]] = llvm.urem %[[VAL_173]], %[[VAL_167]] : i64
// CHECK-NEXT: %[[VAL_175:.*]] = llvm.sub %[[VAL_173]], %[[VAL_174]] : i64
// CHECK-NEXT: %[[VAL_176:.*]] = llvm.inttoptr %[[VAL_175]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_177:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_178:.*]] = llvm.insertvalue %[[VAL_169]], %[[VAL_177]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_179:.*]] = llvm.insertvalue %[[VAL_176]], %[[VAL_178]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_180:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_181:.*]] = llvm.insertvalue %[[VAL_180]], %[[VAL_179]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_182:.*]] = llvm.insertvalue %[[VAL_162]], %[[VAL_181]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_183:.*]] = llvm.insertvalue %[[VAL_163]], %[[VAL_182]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_184:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_185:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_186:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_187:.*]] = llvm.getelementptr %[[VAL_186]][%[[VAL_184]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_188:.*]] = llvm.ptrtoint %[[VAL_187]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_189:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_190:.*]] = llvm.add %[[VAL_188]], %[[VAL_189]] : i64
// CHECK-NEXT: %[[VAL_191:.*]] = llvm.call @malloc(%[[VAL_190]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_192:.*]] = llvm.ptrtoint %[[VAL_191]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_193:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_194:.*]] = llvm.sub %[[VAL_189]], %[[VAL_193]] : i64
// CHECK-NEXT: %[[VAL_195:.*]] = llvm.add %[[VAL_192]], %[[VAL_194]] : i64
// CHECK-NEXT: %[[VAL_196:.*]] = llvm.urem %[[VAL_195]], %[[VAL_189]] : i64
// CHECK-NEXT: %[[VAL_197:.*]] = llvm.sub %[[VAL_195]], %[[VAL_196]] : i64
// CHECK-NEXT: %[[VAL_198:.*]] = llvm.inttoptr %[[VAL_197]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_199:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_200:.*]] = llvm.insertvalue %[[VAL_191]], %[[VAL_199]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_201:.*]] = llvm.insertvalue %[[VAL_198]], %[[VAL_200]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_202:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_203:.*]] = llvm.insertvalue %[[VAL_202]], %[[VAL_201]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_204:.*]] = llvm.insertvalue %[[VAL_184]], %[[VAL_203]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_205:.*]] = llvm.insertvalue %[[VAL_185]], %[[VAL_204]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.br ^bb4(%[[VAL_48]] : i64)
// CHECK-NEXT: ^bb4(%[[VAL_206:.*]]: i64):  // 2 preds: ^bb3, ^bb13
// CHECK-NEXT: %[[VAL_207:.*]] = llvm.icmp "slt" %[[VAL_206]], %[[VAL_23]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_207]], ^bb5, ^bb14
// CHECK-NEXT: ^bb5:  // pred: ^bb4
// CHECK-NEXT: %[[VAL_208:.*]] = llvm.icmp "ult" %[[VAL_206]], %[[VAL_48]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_208]], ^bb6, ^bb7
// CHECK-NEXT: ^bb6:  // pred: ^bb5
// CHECK-NEXT: llvm.br ^bb8(%[[VAL_47]] : i64)
// CHECK-NEXT: ^bb7:  // pred: ^bb5
// CHECK-NEXT: %[[VAL_209:.*]] = llvm.extractvalue %[[VAL_157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_210:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_209]][%[[VAL_206]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_211:.*]] = llvm.load %[[VAL_210]] : !llvm.ptr -> i64
// CHECK-NEXT: llvm.br ^bb8(%[[VAL_211]] : i64)
// CHECK-NEXT: ^bb8(%[[VAL_212:.*]]: i64):  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT: llvm.br ^bb9
// CHECK-NEXT: ^bb9:  // pred: ^bb8
// CHECK-NEXT: %[[VAL_213:.*]] = llvm.icmp "ult" %[[VAL_206]], %[[VAL_23]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_213]], ^bb10, ^bb11
// CHECK-NEXT: ^bb10:  // pred: ^bb9
// CHECK-NEXT: llvm.br ^bb12(%[[VAL_212]] : i64)
// CHECK-NEXT: ^bb11:  // pred: ^bb9
// CHECK-NEXT: %[[VAL_214:.*]] = llvm.sub %[[VAL_206]], %[[VAL_23]] : i64
// CHECK-NEXT: %[[VAL_215:.*]] = llvm.extractvalue %[[VAL_183]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_216:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_215]][%[[VAL_214]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_217:.*]] = llvm.load %[[VAL_216]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_218:.*]] = llvm.icmp "eq" %[[VAL_217]], %[[VAL_47]] : i64
// CHECK-NEXT: %[[VAL_219:.*]] = llvm.select %[[VAL_218]], %[[VAL_212]], %[[VAL_217]] : i1, i64
// CHECK-NEXT: llvm.br ^bb12(%[[VAL_219]] : i64)
// CHECK-NEXT: ^bb12(%[[VAL_220:.*]]: i64):  // 2 preds: ^bb10, ^bb11
// CHECK-NEXT: llvm.br ^bb13
// CHECK-NEXT: ^bb13:  // pred: ^bb12
// CHECK-NEXT: %[[VAL_221:.*]] = llvm.extractvalue %[[VAL_205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_222:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_221]][%[[VAL_206]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_220]], %[[VAL_222]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_223:.*]] = llvm.add %[[VAL_206]], %[[VAL_47]] : i64
// CHECK-NEXT: llvm.br ^bb4(%[[VAL_223]] : i64)
// CHECK-NEXT: ^bb14:  // pred: ^bb4
// CHECK-NEXT: %[[VAL_224:.*]] = llvm.extractvalue %[[VAL_183]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_224]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_225:.*]] = llvm.extractvalue %[[VAL_157]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_225]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_226:.*]] = llvm.extractvalue %[[VAL_205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_227:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_226]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_228:.*]] = llvm.load %[[VAL_227]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_229:.*]] = llvm.extractvalue %[[VAL_205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_230:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_229]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_231:.*]] = llvm.load %[[VAL_230]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_232:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_233:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_234:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_235:.*]] = llvm.getelementptr %[[VAL_234]][%[[VAL_232]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_236:.*]] = llvm.ptrtoint %[[VAL_235]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_237:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_238:.*]] = llvm.add %[[VAL_236]], %[[VAL_237]] : i64
// CHECK-NEXT: %[[VAL_239:.*]] = llvm.call @malloc(%[[VAL_238]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_240:.*]] = llvm.ptrtoint %[[VAL_239]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_241:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_242:.*]] = llvm.sub %[[VAL_237]], %[[VAL_241]] : i64
// CHECK-NEXT: %[[VAL_243:.*]] = llvm.add %[[VAL_240]], %[[VAL_242]] : i64
// CHECK-NEXT: %[[VAL_244:.*]] = llvm.urem %[[VAL_243]], %[[VAL_237]] : i64
// CHECK-NEXT: %[[VAL_245:.*]] = llvm.sub %[[VAL_243]], %[[VAL_244]] : i64
// CHECK-NEXT: %[[VAL_246:.*]] = llvm.inttoptr %[[VAL_245]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_247:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_248:.*]] = llvm.insertvalue %[[VAL_239]], %[[VAL_247]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_249:.*]] = llvm.insertvalue %[[VAL_246]], %[[VAL_248]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_250:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_251:.*]] = llvm.insertvalue %[[VAL_250]], %[[VAL_249]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_252:.*]] = llvm.insertvalue %[[VAL_232]], %[[VAL_251]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_253:.*]] = llvm.insertvalue %[[VAL_233]], %[[VAL_252]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_254:.*]] = llvm.extractvalue %[[VAL_253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_255:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_254]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_228]], %[[VAL_255]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_256:.*]] = llvm.extractvalue %[[VAL_253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_257:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_256]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_231]], %[[VAL_257]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_258:.*]] = llvm.extractvalue %[[VAL_253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_259:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_258]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_47]], %[[VAL_259]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_260:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_261:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_262:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_263:.*]] = llvm.getelementptr %[[VAL_262]][%[[VAL_260]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_264:.*]] = llvm.ptrtoint %[[VAL_263]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_265:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_266:.*]] = llvm.add %[[VAL_264]], %[[VAL_265]] : i64
// CHECK-NEXT: %[[VAL_267:.*]] = llvm.call @malloc(%[[VAL_266]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_268:.*]] = llvm.ptrtoint %[[VAL_267]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_269:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_270:.*]] = llvm.sub %[[VAL_265]], %[[VAL_269]] : i64
// CHECK-NEXT: %[[VAL_271:.*]] = llvm.add %[[VAL_268]], %[[VAL_270]] : i64
// CHECK-NEXT: %[[VAL_272:.*]] = llvm.urem %[[VAL_271]], %[[VAL_265]] : i64
// CHECK-NEXT: %[[VAL_273:.*]] = llvm.sub %[[VAL_271]], %[[VAL_272]] : i64
// CHECK-NEXT: %[[VAL_274:.*]] = llvm.inttoptr %[[VAL_273]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_275:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_276:.*]] = llvm.insertvalue %[[VAL_267]], %[[VAL_275]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_277:.*]] = llvm.insertvalue %[[VAL_274]], %[[VAL_276]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_278:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_279:.*]] = llvm.insertvalue %[[VAL_278]], %[[VAL_277]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_280:.*]] = llvm.insertvalue %[[VAL_260]], %[[VAL_279]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_281:.*]] = llvm.insertvalue %[[VAL_261]], %[[VAL_280]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_282:.*]] = llvm.extractvalue %[[VAL_281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_283:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_282]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_22]], %[[VAL_283]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_284:.*]] = llvm.extractvalue %[[VAL_281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_285:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_284]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_24]], %[[VAL_285]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_286:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_287:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_288:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_289:.*]] = llvm.getelementptr %[[VAL_288]][%[[VAL_286]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_290:.*]] = llvm.ptrtoint %[[VAL_289]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_291:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_292:.*]] = llvm.add %[[VAL_290]], %[[VAL_291]] : i64
// CHECK-NEXT: %[[VAL_293:.*]] = llvm.call @malloc(%[[VAL_292]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_294:.*]] = llvm.ptrtoint %[[VAL_293]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_295:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_296:.*]] = llvm.sub %[[VAL_291]], %[[VAL_295]] : i64
// CHECK-NEXT: %[[VAL_297:.*]] = llvm.add %[[VAL_294]], %[[VAL_296]] : i64
// CHECK-NEXT: %[[VAL_298:.*]] = llvm.urem %[[VAL_297]], %[[VAL_291]] : i64
// CHECK-NEXT: %[[VAL_299:.*]] = llvm.sub %[[VAL_297]], %[[VAL_298]] : i64
// CHECK-NEXT: %[[VAL_300:.*]] = llvm.inttoptr %[[VAL_299]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_301:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_302:.*]] = llvm.insertvalue %[[VAL_293]], %[[VAL_301]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_303:.*]] = llvm.insertvalue %[[VAL_300]], %[[VAL_302]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_304:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_305:.*]] = llvm.insertvalue %[[VAL_304]], %[[VAL_303]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_306:.*]] = llvm.insertvalue %[[VAL_286]], %[[VAL_305]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_307:.*]] = llvm.insertvalue %[[VAL_287]], %[[VAL_306]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_308:.*]] = llvm.extractvalue %[[VAL_307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_309:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_308]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_134]], %[[VAL_309]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_310:.*]] = llvm.extractvalue %[[VAL_307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_311:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_310]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_135]], %[[VAL_311]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_312:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_313:.*]] = llvm.extractvalue %[[VAL_281]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_314:.*]] = llvm.extractvalue %[[VAL_281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_315:.*]] = llvm.insertvalue %[[VAL_313]], %[[VAL_312]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_316:.*]] = llvm.insertvalue %[[VAL_314]], %[[VAL_315]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_317:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_318:.*]] = llvm.insertvalue %[[VAL_317]], %[[VAL_316]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_319:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_320:.*]] = llvm.insertvalue %[[VAL_319]], %[[VAL_318]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_321:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_322:.*]] = llvm.insertvalue %[[VAL_321]], %[[VAL_320]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_323:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_324:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_325:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_326:.*]] = llvm.getelementptr %[[VAL_325]][%[[VAL_323]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_327:.*]] = llvm.ptrtoint %[[VAL_326]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_328:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_329:.*]] = llvm.add %[[VAL_327]], %[[VAL_328]] : i64
// CHECK-NEXT: %[[VAL_330:.*]] = llvm.call @malloc(%[[VAL_329]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_331:.*]] = llvm.ptrtoint %[[VAL_330]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_332:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_333:.*]] = llvm.sub %[[VAL_328]], %[[VAL_332]] : i64
// CHECK-NEXT: %[[VAL_334:.*]] = llvm.add %[[VAL_331]], %[[VAL_333]] : i64
// CHECK-NEXT: %[[VAL_335:.*]] = llvm.urem %[[VAL_334]], %[[VAL_328]] : i64
// CHECK-NEXT: %[[VAL_336:.*]] = llvm.sub %[[VAL_334]], %[[VAL_335]] : i64
// CHECK-NEXT: %[[VAL_337:.*]] = llvm.inttoptr %[[VAL_336]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_338:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_339:.*]] = llvm.insertvalue %[[VAL_330]], %[[VAL_338]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_340:.*]] = llvm.insertvalue %[[VAL_337]], %[[VAL_339]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_341:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_342:.*]] = llvm.insertvalue %[[VAL_341]], %[[VAL_340]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_343:.*]] = llvm.insertvalue %[[VAL_323]], %[[VAL_342]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_344:.*]] = llvm.insertvalue %[[VAL_324]], %[[VAL_343]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_345:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_346:.*]] = llvm.extractvalue %[[VAL_307]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_347:.*]] = llvm.mul %[[VAL_345]], %[[VAL_346]] : i64
// CHECK-NEXT: %[[VAL_348:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_349:.*]] = llvm.getelementptr %[[VAL_348]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_350:.*]] = llvm.ptrtoint %[[VAL_349]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_351:.*]] = llvm.mul %[[VAL_347]], %[[VAL_350]] : i64
// CHECK-NEXT: %[[VAL_352:.*]] = llvm.extractvalue %[[VAL_307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_353:.*]] = llvm.extractvalue %[[VAL_307]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_354:.*]] = llvm.getelementptr %[[VAL_352]][%[[VAL_353]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_355:.*]] = llvm.extractvalue %[[VAL_344]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_356:.*]] = llvm.extractvalue %[[VAL_344]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_357:.*]] = llvm.getelementptr %[[VAL_355]][%[[VAL_356]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[VAL_357]], %[[VAL_354]], %[[VAL_351]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[VAL_358:.*]] = llvm.extractvalue %[[VAL_307]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_358]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_359:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_360:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_361:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_362:.*]] = llvm.getelementptr %[[VAL_361]][%[[VAL_359]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_363:.*]] = llvm.ptrtoint %[[VAL_362]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_364:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_365:.*]] = llvm.add %[[VAL_363]], %[[VAL_364]] : i64
// CHECK-NEXT: %[[VAL_366:.*]] = llvm.call @malloc(%[[VAL_365]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_367:.*]] = llvm.ptrtoint %[[VAL_366]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_368:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_369:.*]] = llvm.sub %[[VAL_364]], %[[VAL_368]] : i64
// CHECK-NEXT: %[[VAL_370:.*]] = llvm.add %[[VAL_367]], %[[VAL_369]] : i64
// CHECK-NEXT: %[[VAL_371:.*]] = llvm.urem %[[VAL_370]], %[[VAL_364]] : i64
// CHECK-NEXT: %[[VAL_372:.*]] = llvm.sub %[[VAL_370]], %[[VAL_371]] : i64
// CHECK-NEXT: %[[VAL_373:.*]] = llvm.inttoptr %[[VAL_372]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_374:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_375:.*]] = llvm.insertvalue %[[VAL_366]], %[[VAL_374]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_376:.*]] = llvm.insertvalue %[[VAL_373]], %[[VAL_375]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_377:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_378:.*]] = llvm.insertvalue %[[VAL_377]], %[[VAL_376]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_379:.*]] = llvm.insertvalue %[[VAL_359]], %[[VAL_378]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_380:.*]] = llvm.insertvalue %[[VAL_360]], %[[VAL_379]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_381:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_382:.*]] = llvm.extractvalue %[[VAL_380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_383:.*]] = llvm.extractvalue %[[VAL_380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_384:.*]] = llvm.insertvalue %[[VAL_382]], %[[VAL_381]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_385:.*]] = llvm.insertvalue %[[VAL_383]], %[[VAL_384]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_386:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_387:.*]] = llvm.insertvalue %[[VAL_386]], %[[VAL_385]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_388:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_389:.*]] = llvm.insertvalue %[[VAL_388]], %[[VAL_387]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_390:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_391:.*]] = llvm.insertvalue %[[VAL_390]], %[[VAL_389]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_392:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_393:.*]] = llvm.extractvalue %[[VAL_344]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_394:.*]] = llvm.mul %[[VAL_392]], %[[VAL_393]] : i64
// CHECK-NEXT: %[[VAL_395:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_396:.*]] = llvm.getelementptr %[[VAL_395]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_397:.*]] = llvm.ptrtoint %[[VAL_396]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_398:.*]] = llvm.mul %[[VAL_394]], %[[VAL_397]] : i64
// CHECK-NEXT: %[[VAL_399:.*]] = llvm.extractvalue %[[VAL_344]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_400:.*]] = llvm.extractvalue %[[VAL_344]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_401:.*]] = llvm.getelementptr %[[VAL_399]][%[[VAL_400]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_402:.*]] = llvm.extractvalue %[[VAL_391]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_403:.*]] = llvm.extractvalue %[[VAL_391]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_404:.*]] = llvm.getelementptr %[[VAL_402]][%[[VAL_403]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[VAL_404]], %[[VAL_401]], %[[VAL_398]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[VAL_405:.*]] = llvm.extractvalue %[[VAL_344]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_405]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_406:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_407:.*]] = llvm.extractvalue %[[VAL_380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_408:.*]] = llvm.extractvalue %[[VAL_380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_409:.*]] = llvm.insertvalue %[[VAL_407]], %[[VAL_406]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_410:.*]] = llvm.insertvalue %[[VAL_408]], %[[VAL_409]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_411:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_412:.*]] = llvm.insertvalue %[[VAL_411]], %[[VAL_410]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_413:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_414:.*]] = llvm.insertvalue %[[VAL_413]], %[[VAL_412]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_415:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_416:.*]] = llvm.insertvalue %[[VAL_415]], %[[VAL_414]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_417:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_418:.*]] = llvm.extractvalue %[[VAL_322]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_419:.*]] = llvm.mul %[[VAL_417]], %[[VAL_418]] : i64
// CHECK-NEXT: %[[VAL_420:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_421:.*]] = llvm.getelementptr %[[VAL_420]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_422:.*]] = llvm.ptrtoint %[[VAL_421]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_423:.*]] = llvm.mul %[[VAL_419]], %[[VAL_422]] : i64
// CHECK-NEXT: %[[VAL_424:.*]] = llvm.extractvalue %[[VAL_322]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_425:.*]] = llvm.extractvalue %[[VAL_322]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_426:.*]] = llvm.getelementptr %[[VAL_424]][%[[VAL_425]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_427:.*]] = llvm.extractvalue %[[VAL_416]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_428:.*]] = llvm.extractvalue %[[VAL_416]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_429:.*]] = llvm.getelementptr %[[VAL_427]][%[[VAL_428]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[VAL_429]], %[[VAL_426]], %[[VAL_423]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[VAL_430:.*]] = llvm.extractvalue %[[VAL_281]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_430]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_431:.*]] = llvm.extractvalue %[[VAL_205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_432:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_431]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_433:.*]] = llvm.load %[[VAL_432]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_434:.*]] = llvm.extractvalue %[[VAL_205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_435:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_434]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_436:.*]] = llvm.load %[[VAL_435]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_437:.*]] = llvm.extractvalue %[[VAL_380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_438:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_437]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_439:.*]] = llvm.load %[[VAL_438]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_440:.*]] = llvm.extractvalue %[[VAL_380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_441:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_440]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_442:.*]] = llvm.load %[[VAL_441]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_443:.*]] = llvm.mul %[[VAL_433]], %[[VAL_436]] : i64
// CHECK-NEXT: %[[VAL_444:.*]] = llvm.add %[[VAL_443]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_445:.*]] = llvm.udiv %[[VAL_444]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_446:.*]] = llvm.mul %[[VAL_445]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_447:.*]] = llvm.mul %[[VAL_439]], %[[VAL_19]] : i64
// CHECK-NEXT: %[[VAL_448:.*]] = llvm.mul %[[VAL_447]], %[[VAL_442]] : i64
// CHECK-NEXT: %[[VAL_449:.*]] = llvm.add %[[VAL_448]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_450:.*]] = llvm.udiv %[[VAL_449]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_451:.*]] = llvm.mul %[[VAL_450]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_452:.*]] = llvm.add %[[VAL_446]], %[[VAL_451]] : i64
// CHECK-NEXT: %[[VAL_453:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_454:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_453]], %[[VAL_452]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_455:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_456:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_457:.*]] = llvm.insertvalue %[[VAL_454]], %[[VAL_456]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_458:.*]] = llvm.insertvalue %[[VAL_454]], %[[VAL_457]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_459:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_460:.*]] = llvm.insertvalue %[[VAL_459]], %[[VAL_458]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_461:.*]] = llvm.insertvalue %[[VAL_452]], %[[VAL_460]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_462:.*]] = llvm.insertvalue %[[VAL_455]], %[[VAL_461]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_463:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_464:.*]] = llvm.extractvalue %[[VAL_462]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_465:.*]] = llvm.insertvalue %[[VAL_464]], %[[VAL_463]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_466:.*]] = llvm.extractvalue %[[VAL_462]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_467:.*]] = llvm.getelementptr %[[VAL_466]][%[[VAL_48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_468:.*]] = llvm.insertvalue %[[VAL_467]], %[[VAL_465]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_469:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_470:.*]] = llvm.insertvalue %[[VAL_469]], %[[VAL_468]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_471:.*]] = llvm.insertvalue %[[VAL_436]], %[[VAL_470]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_472:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_473:.*]] = llvm.insertvalue %[[VAL_472]], %[[VAL_471]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_474:.*]] = llvm.insertvalue %[[VAL_433]], %[[VAL_473]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_475:.*]] = llvm.mul %[[VAL_472]], %[[VAL_436]] : i64
// CHECK-NEXT: %[[VAL_476:.*]] = llvm.insertvalue %[[VAL_475]], %[[VAL_474]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_477:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_478:.*]] = llvm.extractvalue %[[VAL_462]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_479:.*]] = llvm.insertvalue %[[VAL_478]], %[[VAL_477]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_480:.*]] = llvm.extractvalue %[[VAL_462]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_481:.*]] = llvm.getelementptr %[[VAL_480]][%[[VAL_446]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_482:.*]] = llvm.insertvalue %[[VAL_481]], %[[VAL_479]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_483:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_484:.*]] = llvm.insertvalue %[[VAL_483]], %[[VAL_482]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_485:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT: %[[VAL_486:.*]] = llvm.insertvalue %[[VAL_485]], %[[VAL_484]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_487:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_488:.*]] = llvm.insertvalue %[[VAL_487]], %[[VAL_486]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_489:.*]] = llvm.insertvalue %[[VAL_442]], %[[VAL_488]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_490:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT: %[[VAL_491:.*]] = llvm.insertvalue %[[VAL_490]], %[[VAL_489]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_492:.*]] = llvm.insertvalue %[[VAL_439]], %[[VAL_491]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_493:.*]] = llvm.mul %[[VAL_490]], %[[VAL_442]] : i64
// CHECK-NEXT: %[[VAL_494:.*]] = llvm.insertvalue %[[VAL_493]], %[[VAL_492]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_495:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_496:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_497:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_498:.*]] = llvm.getelementptr %[[VAL_497]][%[[VAL_495]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_499:.*]] = llvm.ptrtoint %[[VAL_498]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_500:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_501:.*]] = llvm.add %[[VAL_499]], %[[VAL_500]] : i64
// CHECK-NEXT: %[[VAL_502:.*]] = llvm.call @malloc(%[[VAL_501]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_503:.*]] = llvm.ptrtoint %[[VAL_502]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_504:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_505:.*]] = llvm.sub %[[VAL_500]], %[[VAL_504]] : i64
// CHECK-NEXT: %[[VAL_506:.*]] = llvm.add %[[VAL_503]], %[[VAL_505]] : i64
// CHECK-NEXT: %[[VAL_507:.*]] = llvm.urem %[[VAL_506]], %[[VAL_500]] : i64
// CHECK-NEXT: %[[VAL_508:.*]] = llvm.sub %[[VAL_506]], %[[VAL_507]] : i64
// CHECK-NEXT: %[[VAL_509:.*]] = llvm.inttoptr %[[VAL_508]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_510:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_511:.*]] = llvm.insertvalue %[[VAL_502]], %[[VAL_510]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_512:.*]] = llvm.insertvalue %[[VAL_509]], %[[VAL_511]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_513:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_514:.*]] = llvm.insertvalue %[[VAL_513]], %[[VAL_512]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_515:.*]] = llvm.insertvalue %[[VAL_495]], %[[VAL_514]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_516:.*]] = llvm.insertvalue %[[VAL_496]], %[[VAL_515]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_517:.*]] = llvm.extractvalue %[[VAL_15]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_518:.*]] = llvm.extractvalue %[[VAL_15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_519:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_520:.*]] = llvm.insertvalue %[[VAL_517]], %[[VAL_519]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_521:.*]] = llvm.insertvalue %[[VAL_518]], %[[VAL_520]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_522:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_523:.*]] = llvm.insertvalue %[[VAL_522]], %[[VAL_521]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_524:.*]] = llvm.extractvalue %[[VAL_15]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_525:.*]] = llvm.extractvalue %[[VAL_15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_526:.*]] = llvm.extractvalue %[[VAL_15]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_527:.*]] = llvm.extractvalue %[[VAL_15]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_528:.*]] = llvm.extractvalue %[[VAL_15]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_529:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT: %[[VAL_530:.*]] = llvm.mul %[[VAL_525]], %[[VAL_529]] overflow<nsw> : i64
// CHECK-NEXT: %[[VAL_531:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_532:.*]] = llvm.extractvalue %[[VAL_523]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_533:.*]] = llvm.extractvalue %[[VAL_523]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_534:.*]] = llvm.insertvalue %[[VAL_532]], %[[VAL_531]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_535:.*]] = llvm.insertvalue %[[VAL_533]], %[[VAL_534]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_536:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_537:.*]] = llvm.insertvalue %[[VAL_536]], %[[VAL_535]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_538:.*]] = llvm.insertvalue %[[VAL_530]], %[[VAL_537]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_539:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_540:.*]] = llvm.insertvalue %[[VAL_539]], %[[VAL_538]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_541:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_542:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_543:.*]] = llvm.extractvalue %[[VAL_7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_544:.*]] = llvm.extractvalue %[[VAL_7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_545:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_546:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_547:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_548:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_549:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_550:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_551:.*]] = llvm.extractvalue %[[VAL_476]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_552:.*]] = llvm.extractvalue %[[VAL_476]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_553:.*]] = llvm.extractvalue %[[VAL_7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_554:.*]] = llvm.extractvalue %[[VAL_31]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_555:.*]] = llvm.extractvalue %[[VAL_476]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_556:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_557:.*]] = llvm.call @wrap_equal(%arg0, %[[VAL_553]], %[[VAL_554]], %[[VAL_555]], %[[VAL_541]], %[[VAL_542]], %[[VAL_543]], %[[VAL_544]], %[[VAL_545]], %[[VAL_546]], %[[VAL_547]], %[[VAL_548]], %[[VAL_549]], %[[VAL_550]], %[[VAL_551]], %[[VAL_552]], %[[VAL_556]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_558:.*]] = llvm.extractvalue %[[VAL_476]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_559:.*]] = llvm.extractvalue %[[VAL_476]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_560:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_561:.*]] = llvm.insertvalue %[[VAL_558]], %[[VAL_560]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_562:.*]] = llvm.insertvalue %[[VAL_559]], %[[VAL_561]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_563:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_564:.*]] = llvm.insertvalue %[[VAL_563]], %[[VAL_562]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_565:.*]] = llvm.extractvalue %[[VAL_476]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_566:.*]] = llvm.extractvalue %[[VAL_476]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_567:.*]] = llvm.extractvalue %[[VAL_476]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_568:.*]] = llvm.extractvalue %[[VAL_476]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_569:.*]] = llvm.extractvalue %[[VAL_476]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_570:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_571:.*]] = llvm.extractvalue %[[VAL_564]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_572:.*]] = llvm.extractvalue %[[VAL_564]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_573:.*]] = llvm.insertvalue %[[VAL_571]], %[[VAL_570]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_574:.*]] = llvm.insertvalue %[[VAL_572]], %[[VAL_573]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_575:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_576:.*]] = llvm.insertvalue %[[VAL_575]], %[[VAL_574]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_577:.*]] = llvm.insertvalue %[[VAL_566]], %[[VAL_576]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_578:.*]] = llvm.insertvalue %[[VAL_568]], %[[VAL_577]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_579:.*]] = llvm.insertvalue %[[VAL_567]], %[[VAL_578]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_580:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_581:.*]] = llvm.insertvalue %[[VAL_580]], %[[VAL_579]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_582:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_583:.*]] = llvm.insertvalue %[[VAL_582]], %[[VAL_581]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_584:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_585:.*]] = llvm.insertvalue %[[VAL_584]], %[[VAL_583]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_586:.*]] = llvm.mlir.constant(248320 : i64) : i64
// CHECK-NEXT: %[[VAL_587:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT: %[[VAL_588:.*]] = llvm.extractvalue %[[VAL_7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_589:.*]] = llvm.extractvalue %[[VAL_7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_590:.*]] = llvm.extractvalue %[[VAL_494]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_591:.*]] = llvm.extractvalue %[[VAL_494]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_592:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT: %[[VAL_593:.*]] = llvm.mul %[[VAL_586]], %[[VAL_587]] : i64
// CHECK-NEXT: %[[VAL_594:.*]] = llvm.mul %[[VAL_590]], %[[VAL_591]] : i64
// CHECK-NEXT: %[[VAL_595:.*]] = llvm.mul %[[VAL_594]], %[[VAL_592]] : i64
// CHECK-NEXT: %[[VAL_596:.*]] = llvm.mul %[[VAL_588]], %[[VAL_589]] : i64
// CHECK-NEXT: %[[VAL_597:.*]] = llvm.extractvalue %[[VAL_46]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_598:.*]] = llvm.extractvalue %[[VAL_7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_599:.*]] = llvm.extractvalue %[[VAL_494]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_600:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_601:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_602:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT: %[[VAL_603:.*]] = llvm.call @wrap_gather(%arg0, %[[VAL_597]], %[[VAL_598]], %[[VAL_599]], %[[VAL_600]], %[[VAL_593]], %[[VAL_596]], %[[VAL_595]], %[[VAL_586]], %[[VAL_587]], %[[VAL_601]], %[[VAL_602]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_604:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_605:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_604]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_439]], %[[VAL_605]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_606:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_607:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_606]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_442]], %[[VAL_607]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_608:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_609:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_608]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_18]], %[[VAL_609]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_610:.*]] = llvm.extractvalue %[[VAL_131]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_610]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_611:.*]] = llvm.extractvalue %[[VAL_205]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_611]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_612:.*]] = llvm.extractvalue %[[VAL_253]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_612]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_613:.*]] = llvm.extractvalue %[[VAL_380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_613]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_614:.*]] = llvm.extractvalue %[[VAL_71]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_614]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_615:.*]] = llvm.extractvalue %[[VAL_585]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_616:.*]] = llvm.extractvalue %[[VAL_585]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_617:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_618:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_619:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_620:.*]] = llvm.getelementptr %[[VAL_619]][%[[VAL_617]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_621:.*]] = llvm.ptrtoint %[[VAL_620]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_622:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_623:.*]] = llvm.add %[[VAL_621]], %[[VAL_622]] : i64
// CHECK-NEXT: %[[VAL_624:.*]] = llvm.call @malloc(%[[VAL_623]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_625:.*]] = llvm.ptrtoint %[[VAL_624]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_626:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_627:.*]] = llvm.sub %[[VAL_622]], %[[VAL_626]] : i64
// CHECK-NEXT: %[[VAL_628:.*]] = llvm.add %[[VAL_625]], %[[VAL_627]] : i64
// CHECK-NEXT: %[[VAL_629:.*]] = llvm.urem %[[VAL_628]], %[[VAL_622]] : i64
// CHECK-NEXT: %[[VAL_630:.*]] = llvm.sub %[[VAL_628]], %[[VAL_629]] : i64
// CHECK-NEXT: %[[VAL_631:.*]] = llvm.inttoptr %[[VAL_630]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_632:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_633:.*]] = llvm.insertvalue %[[VAL_624]], %[[VAL_632]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_634:.*]] = llvm.insertvalue %[[VAL_631]], %[[VAL_633]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_635:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_636:.*]] = llvm.insertvalue %[[VAL_635]], %[[VAL_634]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_637:.*]] = llvm.insertvalue %[[VAL_617]], %[[VAL_636]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_638:.*]] = llvm.insertvalue %[[VAL_618]], %[[VAL_637]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_639:.*]] = llvm.extractvalue %[[VAL_638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_640:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_639]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_615]], %[[VAL_640]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_641:.*]] = llvm.extractvalue %[[VAL_638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_642:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_641]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_616]], %[[VAL_642]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_643:.*]] = llvm.extractvalue %[[VAL_638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_644:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_643]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_47]], %[[VAL_644]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_645:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_646:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_645]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_647:.*]] = llvm.load %[[VAL_646]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_648:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_649:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_648]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_650:.*]] = llvm.load %[[VAL_649]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_651:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_652:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_651]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_653:.*]] = llvm.load %[[VAL_652]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_654:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_655:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_656:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_657:.*]] = llvm.getelementptr %[[VAL_656]][%[[VAL_654]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_658:.*]] = llvm.ptrtoint %[[VAL_657]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_659:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_660:.*]] = llvm.add %[[VAL_658]], %[[VAL_659]] : i64
// CHECK-NEXT: %[[VAL_661:.*]] = llvm.call @malloc(%[[VAL_660]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_662:.*]] = llvm.ptrtoint %[[VAL_661]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_663:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_664:.*]] = llvm.sub %[[VAL_659]], %[[VAL_663]] : i64
// CHECK-NEXT: %[[VAL_665:.*]] = llvm.add %[[VAL_662]], %[[VAL_664]] : i64
// CHECK-NEXT: %[[VAL_666:.*]] = llvm.urem %[[VAL_665]], %[[VAL_659]] : i64
// CHECK-NEXT: %[[VAL_667:.*]] = llvm.sub %[[VAL_665]], %[[VAL_666]] : i64
// CHECK-NEXT: %[[VAL_668:.*]] = llvm.inttoptr %[[VAL_667]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_669:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_670:.*]] = llvm.insertvalue %[[VAL_661]], %[[VAL_669]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_671:.*]] = llvm.insertvalue %[[VAL_668]], %[[VAL_670]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_672:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_673:.*]] = llvm.insertvalue %[[VAL_672]], %[[VAL_671]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_674:.*]] = llvm.insertvalue %[[VAL_654]], %[[VAL_673]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_675:.*]] = llvm.insertvalue %[[VAL_655]], %[[VAL_674]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_676:.*]] = llvm.extractvalue %[[VAL_675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_677:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_676]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_647]], %[[VAL_677]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_678:.*]] = llvm.extractvalue %[[VAL_675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_679:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_678]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_650]], %[[VAL_679]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_680:.*]] = llvm.extractvalue %[[VAL_675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_681:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_680]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_653]], %[[VAL_681]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_682:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_683:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_684:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_685:.*]] = llvm.getelementptr %[[VAL_684]][%[[VAL_682]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_686:.*]] = llvm.ptrtoint %[[VAL_685]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_687:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_688:.*]] = llvm.add %[[VAL_686]], %[[VAL_687]] : i64
// CHECK-NEXT: %[[VAL_689:.*]] = llvm.call @malloc(%[[VAL_688]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_690:.*]] = llvm.ptrtoint %[[VAL_689]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_691:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_692:.*]] = llvm.sub %[[VAL_687]], %[[VAL_691]] : i64
// CHECK-NEXT: %[[VAL_693:.*]] = llvm.add %[[VAL_690]], %[[VAL_692]] : i64
// CHECK-NEXT: %[[VAL_694:.*]] = llvm.urem %[[VAL_693]], %[[VAL_687]] : i64
// CHECK-NEXT: %[[VAL_695:.*]] = llvm.sub %[[VAL_693]], %[[VAL_694]] : i64
// CHECK-NEXT: %[[VAL_696:.*]] = llvm.inttoptr %[[VAL_695]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_697:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_698:.*]] = llvm.insertvalue %[[VAL_689]], %[[VAL_697]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_699:.*]] = llvm.insertvalue %[[VAL_696]], %[[VAL_698]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_700:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_701:.*]] = llvm.insertvalue %[[VAL_700]], %[[VAL_699]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_702:.*]] = llvm.insertvalue %[[VAL_682]], %[[VAL_701]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_703:.*]] = llvm.insertvalue %[[VAL_683]], %[[VAL_702]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.br ^bb15(%[[VAL_48]] : i64)
// CHECK-NEXT: ^bb15(%[[VAL_704:.*]]: i64):  // 2 preds: ^bb14, ^bb20
// CHECK-NEXT: %[[VAL_705:.*]] = llvm.icmp "slt" %[[VAL_704]], %[[VAL_49]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_705]], ^bb16, ^bb21
// CHECK-NEXT: ^bb16:  // pred: ^bb15
// CHECK-NEXT: %[[VAL_706:.*]] = llvm.icmp "ult" %[[VAL_704]], %[[VAL_48]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_706]], ^bb17, ^bb18
// CHECK-NEXT: ^bb17:  // pred: ^bb16
// CHECK-NEXT: llvm.br ^bb19(%[[VAL_47]] : i64)
// CHECK-NEXT: ^bb18:  // pred: ^bb16
// CHECK-NEXT: %[[VAL_707:.*]] = llvm.extractvalue %[[VAL_638]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_708:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_707]][%[[VAL_704]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_709:.*]] = llvm.load %[[VAL_708]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_710:.*]] = llvm.extractvalue %[[VAL_675]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_711:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_710]][%[[VAL_704]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_712:.*]] = llvm.load %[[VAL_711]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_713:.*]] = llvm.icmp "eq" %[[VAL_712]], %[[VAL_47]] : i64
// CHECK-NEXT: %[[VAL_714:.*]] = llvm.select %[[VAL_713]], %[[VAL_709]], %[[VAL_712]] : i1, i64
// CHECK-NEXT: llvm.br ^bb19(%[[VAL_714]] : i64)
// CHECK-NEXT: ^bb19(%[[VAL_715:.*]]: i64):  // 2 preds: ^bb17, ^bb18
// CHECK-NEXT: llvm.br ^bb20
// CHECK-NEXT: ^bb20:  // pred: ^bb19
// CHECK-NEXT: %[[VAL_716:.*]] = llvm.extractvalue %[[VAL_703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_717:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_716]][%[[VAL_704]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_715]], %[[VAL_717]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_718:.*]] = llvm.add %[[VAL_704]], %[[VAL_47]] : i64
// CHECK-NEXT: llvm.br ^bb15(%[[VAL_718]] : i64)
// CHECK-NEXT: ^bb21:  // pred: ^bb15
// CHECK-NEXT: %[[VAL_719:.*]] = llvm.extractvalue %[[VAL_675]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_719]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_720:.*]] = llvm.extractvalue %[[VAL_638]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_720]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_721:.*]] = llvm.extractvalue %[[VAL_703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_722:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_721]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_723:.*]] = llvm.load %[[VAL_722]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_724:.*]] = llvm.extractvalue %[[VAL_703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_725:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_724]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_726:.*]] = llvm.load %[[VAL_725]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_727:.*]] = llvm.extractvalue %[[VAL_703]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_728:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_727]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_729:.*]] = llvm.load %[[VAL_728]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_730:.*]] = llvm.mul %[[VAL_723]], %[[VAL_726]] : i64
// CHECK-NEXT: %[[VAL_731:.*]] = llvm.mul %[[VAL_730]], %[[VAL_729]] : i64
// CHECK-NEXT: %[[VAL_732:.*]] = llvm.add %[[VAL_731]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_733:.*]] = llvm.udiv %[[VAL_732]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_734:.*]] = llvm.mul %[[VAL_733]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_735:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_736:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_735]], %[[VAL_734]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_737:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_738:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_739:.*]] = llvm.insertvalue %[[VAL_736]], %[[VAL_738]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_740:.*]] = llvm.insertvalue %[[VAL_736]], %[[VAL_739]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_741:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_742:.*]] = llvm.insertvalue %[[VAL_741]], %[[VAL_740]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_743:.*]] = llvm.insertvalue %[[VAL_734]], %[[VAL_742]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_744:.*]] = llvm.insertvalue %[[VAL_737]], %[[VAL_743]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_745:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_746:.*]] = llvm.extractvalue %[[VAL_744]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_747:.*]] = llvm.insertvalue %[[VAL_746]], %[[VAL_745]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_748:.*]] = llvm.extractvalue %[[VAL_744]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_749:.*]] = llvm.getelementptr %[[VAL_748]][%[[VAL_48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_750:.*]] = llvm.insertvalue %[[VAL_749]], %[[VAL_747]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_751:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_752:.*]] = llvm.insertvalue %[[VAL_751]], %[[VAL_750]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_753:.*]] = llvm.insertvalue %[[VAL_729]], %[[VAL_752]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_754:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_755:.*]] = llvm.insertvalue %[[VAL_754]], %[[VAL_753]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_756:.*]] = llvm.insertvalue %[[VAL_726]], %[[VAL_755]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_757:.*]] = llvm.mul %[[VAL_754]], %[[VAL_729]] : i64
// CHECK-NEXT: %[[VAL_758:.*]] = llvm.insertvalue %[[VAL_757]], %[[VAL_756]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_759:.*]] = llvm.insertvalue %[[VAL_723]], %[[VAL_758]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_760:.*]] = llvm.mul %[[VAL_757]], %[[VAL_726]] : i64
// CHECK-NEXT: %[[VAL_761:.*]] = llvm.insertvalue %[[VAL_760]], %[[VAL_759]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_762:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_763:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_764:.*]] = llvm.alloca %[[VAL_762]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_765:.*]] = llvm.extractvalue %[[VAL_585]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_766:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_767:.*]] = llvm.getelementptr %[[VAL_764]][%[[VAL_766]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_765]], %[[VAL_767]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_768:.*]] = llvm.extractvalue %[[VAL_585]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_769:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_770:.*]] = llvm.getelementptr %[[VAL_764]][%[[VAL_769]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_768]], %[[VAL_770]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_771:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_772:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[VAL_773:.*]] = llvm.getelementptr %[[VAL_764]][%[[VAL_772]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_771]], %[[VAL_773]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_774:.*]] = llvm.alloca %[[VAL_762]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_775:.*]] = llvm.extractvalue %[[VAL_761]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_776:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_777:.*]] = llvm.getelementptr %[[VAL_774]][%[[VAL_776]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_775]], %[[VAL_777]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_778:.*]] = llvm.extractvalue %[[VAL_761]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_779:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_780:.*]] = llvm.getelementptr %[[VAL_774]][%[[VAL_779]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_778]], %[[VAL_780]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_781:.*]] = llvm.extractvalue %[[VAL_761]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_782:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[VAL_783:.*]] = llvm.getelementptr %[[VAL_774]][%[[VAL_782]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_781]], %[[VAL_783]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_784:.*]] = llvm.extractvalue %[[VAL_585]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_785:.*]] = llvm.extractvalue %[[VAL_761]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_786:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_787:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_788:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT: %[[VAL_789:.*]] = llvm.call @wrap_expand(%arg0, %[[VAL_784]], %[[VAL_763]], %[[VAL_785]], %[[VAL_764]], %[[VAL_786]], %[[VAL_774]], %[[VAL_787]], %[[VAL_788]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_790:.*]] = llvm.extractvalue %[[VAL_703]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_790]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_791:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_792:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_793:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_794:.*]] = llvm.getelementptr %[[VAL_793]][%[[VAL_791]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_795:.*]] = llvm.ptrtoint %[[VAL_794]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_796:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_797:.*]] = llvm.add %[[VAL_795]], %[[VAL_796]] : i64
// CHECK-NEXT: %[[VAL_798:.*]] = llvm.call @malloc(%[[VAL_797]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_799:.*]] = llvm.ptrtoint %[[VAL_798]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_800:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_801:.*]] = llvm.sub %[[VAL_796]], %[[VAL_800]] : i64
// CHECK-NEXT: %[[VAL_802:.*]] = llvm.add %[[VAL_799]], %[[VAL_801]] : i64
// CHECK-NEXT: %[[VAL_803:.*]] = llvm.urem %[[VAL_802]], %[[VAL_796]] : i64
// CHECK-NEXT: %[[VAL_804:.*]] = llvm.sub %[[VAL_802]], %[[VAL_803]] : i64
// CHECK-NEXT: %[[VAL_805:.*]] = llvm.inttoptr %[[VAL_804]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_806:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_807:.*]] = llvm.insertvalue %[[VAL_798]], %[[VAL_806]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_808:.*]] = llvm.insertvalue %[[VAL_805]], %[[VAL_807]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_809:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_810:.*]] = llvm.insertvalue %[[VAL_809]], %[[VAL_808]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_811:.*]] = llvm.insertvalue %[[VAL_791]], %[[VAL_810]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_812:.*]] = llvm.insertvalue %[[VAL_792]], %[[VAL_811]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_813:.*]] = llvm.extractvalue %[[VAL_812]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_814:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_813]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_47]], %[[VAL_814]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_815:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_816:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_817:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_818:.*]] = llvm.getelementptr %[[VAL_817]][%[[VAL_815]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_819:.*]] = llvm.ptrtoint %[[VAL_818]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_820:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_821:.*]] = llvm.add %[[VAL_819]], %[[VAL_820]] : i64
// CHECK-NEXT: %[[VAL_822:.*]] = llvm.call @malloc(%[[VAL_821]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_823:.*]] = llvm.ptrtoint %[[VAL_822]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_824:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_825:.*]] = llvm.sub %[[VAL_820]], %[[VAL_824]] : i64
// CHECK-NEXT: %[[VAL_826:.*]] = llvm.add %[[VAL_823]], %[[VAL_825]] : i64
// CHECK-NEXT: %[[VAL_827:.*]] = llvm.urem %[[VAL_826]], %[[VAL_820]] : i64
// CHECK-NEXT: %[[VAL_828:.*]] = llvm.sub %[[VAL_826]], %[[VAL_827]] : i64
// CHECK-NEXT: %[[VAL_829:.*]] = llvm.inttoptr %[[VAL_828]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_830:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_831:.*]] = llvm.insertvalue %[[VAL_822]], %[[VAL_830]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_832:.*]] = llvm.insertvalue %[[VAL_829]], %[[VAL_831]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_833:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_834:.*]] = llvm.insertvalue %[[VAL_833]], %[[VAL_832]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_835:.*]] = llvm.insertvalue %[[VAL_815]], %[[VAL_834]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_836:.*]] = llvm.insertvalue %[[VAL_816]], %[[VAL_835]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_837:.*]] = llvm.extractvalue %[[VAL_836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_838:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_837]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_723]], %[[VAL_838]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_839:.*]] = llvm.extractvalue %[[VAL_836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_840:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_839]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_726]], %[[VAL_840]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_841:.*]] = llvm.extractvalue %[[VAL_836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_842:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_841]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_729]], %[[VAL_842]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_843:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_844:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_843]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_845:.*]] = llvm.load %[[VAL_844]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_846:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_847:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_846]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_848:.*]] = llvm.load %[[VAL_847]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_849:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_850:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_849]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_851:.*]] = llvm.load %[[VAL_850]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_852:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_853:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_854:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_855:.*]] = llvm.getelementptr %[[VAL_854]][%[[VAL_852]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_856:.*]] = llvm.ptrtoint %[[VAL_855]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_857:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_858:.*]] = llvm.add %[[VAL_856]], %[[VAL_857]] : i64
// CHECK-NEXT: %[[VAL_859:.*]] = llvm.call @malloc(%[[VAL_858]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_860:.*]] = llvm.ptrtoint %[[VAL_859]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_861:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_862:.*]] = llvm.sub %[[VAL_857]], %[[VAL_861]] : i64
// CHECK-NEXT: %[[VAL_863:.*]] = llvm.add %[[VAL_860]], %[[VAL_862]] : i64
// CHECK-NEXT: %[[VAL_864:.*]] = llvm.urem %[[VAL_863]], %[[VAL_857]] : i64
// CHECK-NEXT: %[[VAL_865:.*]] = llvm.sub %[[VAL_863]], %[[VAL_864]] : i64
// CHECK-NEXT: %[[VAL_866:.*]] = llvm.inttoptr %[[VAL_865]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_867:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_868:.*]] = llvm.insertvalue %[[VAL_859]], %[[VAL_867]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_869:.*]] = llvm.insertvalue %[[VAL_866]], %[[VAL_868]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_870:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_871:.*]] = llvm.insertvalue %[[VAL_870]], %[[VAL_869]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_872:.*]] = llvm.insertvalue %[[VAL_852]], %[[VAL_871]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_873:.*]] = llvm.insertvalue %[[VAL_853]], %[[VAL_872]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_874:.*]] = llvm.extractvalue %[[VAL_873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_875:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_874]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_845]], %[[VAL_875]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_876:.*]] = llvm.extractvalue %[[VAL_873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_877:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_876]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_848]], %[[VAL_877]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_878:.*]] = llvm.extractvalue %[[VAL_873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_879:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_878]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_851]], %[[VAL_879]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_880:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_881:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_882:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_883:.*]] = llvm.getelementptr %[[VAL_882]][%[[VAL_880]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_884:.*]] = llvm.ptrtoint %[[VAL_883]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_885:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_886:.*]] = llvm.add %[[VAL_884]], %[[VAL_885]] : i64
// CHECK-NEXT: %[[VAL_887:.*]] = llvm.call @malloc(%[[VAL_886]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_888:.*]] = llvm.ptrtoint %[[VAL_887]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_889:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_890:.*]] = llvm.sub %[[VAL_885]], %[[VAL_889]] : i64
// CHECK-NEXT: %[[VAL_891:.*]] = llvm.add %[[VAL_888]], %[[VAL_890]] : i64
// CHECK-NEXT: %[[VAL_892:.*]] = llvm.urem %[[VAL_891]], %[[VAL_885]] : i64
// CHECK-NEXT: %[[VAL_893:.*]] = llvm.sub %[[VAL_891]], %[[VAL_892]] : i64
// CHECK-NEXT: %[[VAL_894:.*]] = llvm.inttoptr %[[VAL_893]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_895:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_896:.*]] = llvm.insertvalue %[[VAL_887]], %[[VAL_895]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_897:.*]] = llvm.insertvalue %[[VAL_894]], %[[VAL_896]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_898:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_899:.*]] = llvm.insertvalue %[[VAL_898]], %[[VAL_897]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_900:.*]] = llvm.insertvalue %[[VAL_880]], %[[VAL_899]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_901:.*]] = llvm.insertvalue %[[VAL_881]], %[[VAL_900]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.br ^bb22(%[[VAL_48]] : i64)
// CHECK-NEXT: ^bb22(%[[VAL_902:.*]]: i64):  // 2 preds: ^bb21, ^bb27
// CHECK-NEXT: %[[VAL_903:.*]] = llvm.icmp "slt" %[[VAL_902]], %[[VAL_49]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_903]], ^bb23, ^bb28
// CHECK-NEXT: ^bb23:  // pred: ^bb22
// CHECK-NEXT: %[[VAL_904:.*]] = llvm.icmp "ult" %[[VAL_902]], %[[VAL_48]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_904]], ^bb24, ^bb25
// CHECK-NEXT: ^bb24:  // pred: ^bb23
// CHECK-NEXT: llvm.br ^bb26(%[[VAL_47]] : i64)
// CHECK-NEXT: ^bb25:  // pred: ^bb23
// CHECK-NEXT: %[[VAL_905:.*]] = llvm.extractvalue %[[VAL_836]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_906:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_905]][%[[VAL_902]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_907:.*]] = llvm.load %[[VAL_906]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_908:.*]] = llvm.extractvalue %[[VAL_873]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_909:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_908]][%[[VAL_902]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_910:.*]] = llvm.load %[[VAL_909]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_911:.*]] = llvm.icmp "eq" %[[VAL_910]], %[[VAL_47]] : i64
// CHECK-NEXT: %[[VAL_912:.*]] = llvm.select %[[VAL_911]], %[[VAL_907]], %[[VAL_910]] : i1, i64
// CHECK-NEXT: llvm.br ^bb26(%[[VAL_912]] : i64)
// CHECK-NEXT: ^bb26(%[[VAL_913:.*]]: i64):  // 2 preds: ^bb24, ^bb25
// CHECK-NEXT: llvm.br ^bb27
// CHECK-NEXT: ^bb27:  // pred: ^bb26
// CHECK-NEXT: %[[VAL_914:.*]] = llvm.extractvalue %[[VAL_901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_915:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_914]][%[[VAL_902]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_913]], %[[VAL_915]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_916:.*]] = llvm.add %[[VAL_902]], %[[VAL_47]] : i64
// CHECK-NEXT: llvm.br ^bb22(%[[VAL_916]] : i64)
// CHECK-NEXT: ^bb28:  // pred: ^bb22
// CHECK-NEXT: %[[VAL_917:.*]] = llvm.extractvalue %[[VAL_873]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_917]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_918:.*]] = llvm.extractvalue %[[VAL_836]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_918]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: llvm.br ^bb29(%[[VAL_48]], %[[VAL_47]] : i64, i64)
// CHECK-NEXT: ^bb29(%[[VAL_919:.*]]: i64, %[[VAL_920:.*]]: i64):  // 2 preds: ^bb28, ^bb30
// CHECK-NEXT: %[[VAL_921:.*]] = llvm.icmp "slt" %[[VAL_919]], %[[VAL_49]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_921]], ^bb30, ^bb31
// CHECK-NEXT: ^bb30:  // pred: ^bb29
// CHECK-NEXT: %[[VAL_922:.*]] = llvm.extractvalue %[[VAL_901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_923:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_922]][%[[VAL_919]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_924:.*]] = llvm.load %[[VAL_923]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_925:.*]] = llvm.mul %[[VAL_924]], %[[VAL_920]] : i64
// CHECK-NEXT: %[[VAL_926:.*]] = llvm.add %[[VAL_919]], %[[VAL_47]] : i64
// CHECK-NEXT: llvm.br ^bb29(%[[VAL_926]], %[[VAL_925]] : i64, i64)
// CHECK-NEXT: ^bb31:  // pred: ^bb29
// CHECK-NEXT: %[[VAL_927:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_928:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_929:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_930:.*]] = llvm.getelementptr %[[VAL_929]][%[[VAL_927]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_931:.*]] = llvm.ptrtoint %[[VAL_930]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_932:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_933:.*]] = llvm.add %[[VAL_931]], %[[VAL_932]] : i64
// CHECK-NEXT: %[[VAL_934:.*]] = llvm.call @malloc(%[[VAL_933]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_935:.*]] = llvm.ptrtoint %[[VAL_934]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_936:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_937:.*]] = llvm.sub %[[VAL_932]], %[[VAL_936]] : i64
// CHECK-NEXT: %[[VAL_938:.*]] = llvm.add %[[VAL_935]], %[[VAL_937]] : i64
// CHECK-NEXT: %[[VAL_939:.*]] = llvm.urem %[[VAL_938]], %[[VAL_932]] : i64
// CHECK-NEXT: %[[VAL_940:.*]] = llvm.sub %[[VAL_938]], %[[VAL_939]] : i64
// CHECK-NEXT: %[[VAL_941:.*]] = llvm.inttoptr %[[VAL_940]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_942:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_943:.*]] = llvm.insertvalue %[[VAL_934]], %[[VAL_942]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_944:.*]] = llvm.insertvalue %[[VAL_941]], %[[VAL_943]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_945:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_946:.*]] = llvm.insertvalue %[[VAL_945]], %[[VAL_944]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_947:.*]] = llvm.insertvalue %[[VAL_927]], %[[VAL_946]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_948:.*]] = llvm.insertvalue %[[VAL_928]], %[[VAL_947]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_949:.*]] = llvm.extractvalue %[[VAL_948]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_950:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_949]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_49]], %[[VAL_950]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_951:.*]] = llvm.extractvalue %[[VAL_948]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_952:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_951]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_920]], %[[VAL_952]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_953:.*]] = llvm.extractvalue %[[VAL_901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_954:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_953]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_955:.*]] = llvm.load %[[VAL_954]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_956:.*]] = llvm.extractvalue %[[VAL_901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_957:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_956]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_958:.*]] = llvm.load %[[VAL_957]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_959:.*]] = llvm.extractvalue %[[VAL_901]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_960:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_959]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_961:.*]] = llvm.load %[[VAL_960]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_962:.*]] = llvm.extractvalue %[[VAL_948]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_963:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_962]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_964:.*]] = llvm.load %[[VAL_963]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_965:.*]] = llvm.mul %[[VAL_955]], %[[VAL_958]] : i64
// CHECK-NEXT: %[[VAL_966:.*]] = llvm.mul %[[VAL_965]], %[[VAL_961]] : i64
// CHECK-NEXT: %[[VAL_967:.*]] = llvm.add %[[VAL_966]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_968:.*]] = llvm.udiv %[[VAL_967]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_969:.*]] = llvm.mul %[[VAL_968]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_970:.*]] = llvm.mul %[[VAL_964]], %[[VAL_17]] : i64
// CHECK-NEXT: %[[VAL_971:.*]] = llvm.add %[[VAL_970]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_972:.*]] = llvm.udiv %[[VAL_971]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_973:.*]] = llvm.mul %[[VAL_972]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_974:.*]] = llvm.add %[[VAL_969]], %[[VAL_973]] : i64
// CHECK-NEXT: %[[VAL_975:.*]] = llvm.add %[[VAL_974]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_976:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[VAL_977:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_976]], %[[VAL_975]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_978:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_979:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_980:.*]] = llvm.insertvalue %[[VAL_977]], %[[VAL_979]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_981:.*]] = llvm.insertvalue %[[VAL_977]], %[[VAL_980]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_982:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_983:.*]] = llvm.insertvalue %[[VAL_982]], %[[VAL_981]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_984:.*]] = llvm.insertvalue %[[VAL_975]], %[[VAL_983]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_985:.*]] = llvm.insertvalue %[[VAL_978]], %[[VAL_984]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_986:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_987:.*]] = llvm.extractvalue %[[VAL_985]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_988:.*]] = llvm.insertvalue %[[VAL_987]], %[[VAL_986]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_989:.*]] = llvm.extractvalue %[[VAL_985]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_990:.*]] = llvm.getelementptr %[[VAL_989]][%[[VAL_48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_991:.*]] = llvm.insertvalue %[[VAL_990]], %[[VAL_988]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_992:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_993:.*]] = llvm.insertvalue %[[VAL_992]], %[[VAL_991]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_994:.*]] = llvm.insertvalue %[[VAL_961]], %[[VAL_993]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_995:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_996:.*]] = llvm.insertvalue %[[VAL_995]], %[[VAL_994]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_997:.*]] = llvm.insertvalue %[[VAL_958]], %[[VAL_996]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_998:.*]] = llvm.mul %[[VAL_995]], %[[VAL_961]] : i64
// CHECK-NEXT: %[[VAL_999:.*]] = llvm.insertvalue %[[VAL_998]], %[[VAL_997]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1000:.*]] = llvm.insertvalue %[[VAL_955]], %[[VAL_999]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1001:.*]] = llvm.mul %[[VAL_998]], %[[VAL_958]] : i64
// CHECK-NEXT: %[[VAL_1002:.*]] = llvm.insertvalue %[[VAL_1001]], %[[VAL_1000]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1003:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1004:.*]] = llvm.extractvalue %[[VAL_985]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1005:.*]] = llvm.insertvalue %[[VAL_1004]], %[[VAL_1003]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1006:.*]] = llvm.extractvalue %[[VAL_985]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1007:.*]] = llvm.getelementptr %[[VAL_1006]][%[[VAL_969]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_1008:.*]] = llvm.insertvalue %[[VAL_1007]], %[[VAL_1005]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1009:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1010:.*]] = llvm.insertvalue %[[VAL_1009]], %[[VAL_1008]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1011:.*]] = llvm.insertvalue %[[VAL_964]], %[[VAL_1010]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1012:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1013:.*]] = llvm.insertvalue %[[VAL_1012]], %[[VAL_1011]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1014:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_1015:.*]] = llvm.insertvalue %[[VAL_1014]], %[[VAL_1013]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1016:.*]] = llvm.mul %[[VAL_1012]], %[[VAL_964]] : i64
// CHECK-NEXT: %[[VAL_1017:.*]] = llvm.insertvalue %[[VAL_1016]], %[[VAL_1015]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1018:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1019:.*]] = llvm.extractvalue %[[VAL_985]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1020:.*]] = llvm.insertvalue %[[VAL_1019]], %[[VAL_1018]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1021:.*]] = llvm.extractvalue %[[VAL_985]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1022:.*]] = llvm.getelementptr %[[VAL_1021]][%[[VAL_974]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_1023:.*]] = llvm.insertvalue %[[VAL_1022]], %[[VAL_1020]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1024:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1025:.*]] = llvm.insertvalue %[[VAL_1024]], %[[VAL_1023]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1026:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1027:.*]] = llvm.insertvalue %[[VAL_1026]], %[[VAL_1025]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1028:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1029:.*]] = llvm.insertvalue %[[VAL_1028]], %[[VAL_1027]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1030:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1031:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1032:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1033:.*]] = llvm.getelementptr %[[VAL_1032]][%[[VAL_1030]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT: %[[VAL_1034:.*]] = llvm.ptrtoint %[[VAL_1033]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1035:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1036:.*]] = llvm.add %[[VAL_1034]], %[[VAL_1035]] : i64
// CHECK-NEXT: %[[VAL_1037:.*]] = llvm.call @malloc(%[[VAL_1036]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1038:.*]] = llvm.ptrtoint %[[VAL_1037]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1039:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1040:.*]] = llvm.sub %[[VAL_1035]], %[[VAL_1039]] : i64
// CHECK-NEXT: %[[VAL_1041:.*]] = llvm.add %[[VAL_1038]], %[[VAL_1040]] : i64
// CHECK-NEXT: %[[VAL_1042:.*]] = llvm.urem %[[VAL_1041]], %[[VAL_1035]] : i64
// CHECK-NEXT: %[[VAL_1043:.*]] = llvm.sub %[[VAL_1041]], %[[VAL_1042]] : i64
// CHECK-NEXT: %[[VAL_1044:.*]] = llvm.inttoptr %[[VAL_1043]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1045:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1046:.*]] = llvm.insertvalue %[[VAL_1037]], %[[VAL_1045]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1047:.*]] = llvm.insertvalue %[[VAL_1044]], %[[VAL_1046]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1048:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1049:.*]] = llvm.insertvalue %[[VAL_1048]], %[[VAL_1047]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1050:.*]] = llvm.insertvalue %[[VAL_1030]], %[[VAL_1049]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1051:.*]] = llvm.insertvalue %[[VAL_1031]], %[[VAL_1050]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1052:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1053:.*]] = llvm.extractvalue %[[VAL_516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1054:.*]] = llvm.alloca %[[VAL_1052]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1055:.*]] = llvm.extractvalue %[[VAL_761]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1056:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_1057:.*]] = llvm.getelementptr %[[VAL_1054]][%[[VAL_1056]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1055]], %[[VAL_1057]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1058:.*]] = llvm.extractvalue %[[VAL_761]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1059:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_1060:.*]] = llvm.getelementptr %[[VAL_1054]][%[[VAL_1059]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1058]], %[[VAL_1060]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1061:.*]] = llvm.extractvalue %[[VAL_761]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1062:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[VAL_1063:.*]] = llvm.getelementptr %[[VAL_1054]][%[[VAL_1062]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1061]], %[[VAL_1063]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1064:.*]] = llvm.alloca %[[VAL_1052]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1065:.*]] = llvm.extractvalue %[[VAL_1002]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1066:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_1067:.*]] = llvm.getelementptr %[[VAL_1064]][%[[VAL_1066]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1065]], %[[VAL_1067]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1068:.*]] = llvm.extractvalue %[[VAL_1002]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1069:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_1070:.*]] = llvm.getelementptr %[[VAL_1064]][%[[VAL_1069]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1068]], %[[VAL_1070]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1071:.*]] = llvm.extractvalue %[[VAL_1002]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1072:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[VAL_1073:.*]] = llvm.getelementptr %[[VAL_1064]][%[[VAL_1072]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1071]], %[[VAL_1073]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1074:.*]] = llvm.extractvalue %[[VAL_761]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1075:.*]] = llvm.extractvalue %[[VAL_1002]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1076:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1077:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1078:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT: %[[VAL_1079:.*]] = llvm.call @wrap_expand(%arg0, %[[VAL_1074]], %[[VAL_1053]], %[[VAL_1075]], %[[VAL_1054]], %[[VAL_1076]], %[[VAL_1064]], %[[VAL_1077]], %[[VAL_1078]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_1080:.*]] = llvm.extractvalue %[[VAL_516]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1080]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1081:.*]] = llvm.extractvalue %[[VAL_1002]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1082:.*]] = llvm.extractvalue %[[VAL_1002]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1083:.*]] = llvm.extractvalue %[[VAL_1002]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1084:.*]] = llvm.mul %[[VAL_1081]], %[[VAL_1082]] : i64
// CHECK-NEXT: %[[VAL_1085:.*]] = llvm.mul %[[VAL_1084]], %[[VAL_1083]] : i64
// CHECK-NEXT: %[[VAL_1086:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1087:.*]] = llvm.extractvalue %[[VAL_1017]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1088:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1089:.*]] = llvm.alloca %[[VAL_1088]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1090:.*]] = llvm.getelementptr %[[VAL_1089]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1081]], %[[VAL_1090]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1091:.*]] = llvm.getelementptr %[[VAL_1089]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1082]], %[[VAL_1091]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1092:.*]] = llvm.getelementptr %[[VAL_1089]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1083]], %[[VAL_1092]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1093:.*]] = llvm.extractvalue %[[VAL_1002]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1094:.*]] = llvm.extractvalue %[[VAL_1017]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1095:.*]] = llvm.extractvalue %[[VAL_1029]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1096:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1097:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT: %[[VAL_1098:.*]] = llvm.call @wrap_nonzero(%arg0, %[[VAL_1093]], %[[VAL_1094]], %[[VAL_1095]], %[[VAL_1085]], %[[VAL_1096]], %[[VAL_1089]], %[[VAL_1087]], %[[VAL_1097]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_1099:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1100:.*]] = llvm.getelementptr %[[VAL_1099]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK-NEXT: %[[VAL_1101:.*]] = llvm.ptrtoint %[[VAL_1100]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1102:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1103:.*]] = llvm.mul %[[VAL_1101]], %[[VAL_1102]] : i64
// CHECK-NEXT: %[[VAL_1104:.*]] = llvm.extractvalue %[[VAL_1051]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1105:.*]] = llvm.extractvalue %[[VAL_1029]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1106:.*]] = llvm.call @wrap_copy_d2h(%arg0, %[[VAL_1104]], %[[VAL_1105]], %[[VAL_1103]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT: %[[VAL_1107:.*]] = llvm.extractvalue %[[VAL_901]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1107]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1108:.*]] = llvm.extractvalue %[[VAL_948]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1108]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1109:.*]] = llvm.extractvalue %[[VAL_812]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1109]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1110:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1111:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1112:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1113:.*]] = llvm.getelementptr %[[VAL_1112]][%[[VAL_1110]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1114:.*]] = llvm.ptrtoint %[[VAL_1113]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1115:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1116:.*]] = llvm.add %[[VAL_1114]], %[[VAL_1115]] : i64
// CHECK-NEXT: %[[VAL_1117:.*]] = llvm.call @malloc(%[[VAL_1116]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1118:.*]] = llvm.ptrtoint %[[VAL_1117]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1119:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1120:.*]] = llvm.sub %[[VAL_1115]], %[[VAL_1119]] : i64
// CHECK-NEXT: %[[VAL_1121:.*]] = llvm.add %[[VAL_1118]], %[[VAL_1120]] : i64
// CHECK-NEXT: %[[VAL_1122:.*]] = llvm.urem %[[VAL_1121]], %[[VAL_1115]] : i64
// CHECK-NEXT: %[[VAL_1123:.*]] = llvm.sub %[[VAL_1121]], %[[VAL_1122]] : i64
// CHECK-NEXT: %[[VAL_1124:.*]] = llvm.inttoptr %[[VAL_1123]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1125:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1126:.*]] = llvm.insertvalue %[[VAL_1117]], %[[VAL_1125]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1127:.*]] = llvm.insertvalue %[[VAL_1124]], %[[VAL_1126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1128:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1129:.*]] = llvm.insertvalue %[[VAL_1128]], %[[VAL_1127]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1130:.*]] = llvm.insertvalue %[[VAL_1110]], %[[VAL_1129]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1131:.*]] = llvm.insertvalue %[[VAL_1111]], %[[VAL_1130]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1132:.*]] = llvm.extractvalue %[[VAL_1131]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1133:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1132]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_47]], %[[VAL_1133]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1134:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1135:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1136:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1137:.*]] = llvm.getelementptr %[[VAL_1136]][%[[VAL_1134]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1138:.*]] = llvm.ptrtoint %[[VAL_1137]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1139:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1140:.*]] = llvm.add %[[VAL_1138]], %[[VAL_1139]] : i64
// CHECK-NEXT: %[[VAL_1141:.*]] = llvm.call @malloc(%[[VAL_1140]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1142:.*]] = llvm.ptrtoint %[[VAL_1141]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1143:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1144:.*]] = llvm.sub %[[VAL_1139]], %[[VAL_1143]] : i64
// CHECK-NEXT: %[[VAL_1145:.*]] = llvm.add %[[VAL_1142]], %[[VAL_1144]] : i64
// CHECK-NEXT: %[[VAL_1146:.*]] = llvm.urem %[[VAL_1145]], %[[VAL_1139]] : i64
// CHECK-NEXT: %[[VAL_1147:.*]] = llvm.sub %[[VAL_1145]], %[[VAL_1146]] : i64
// CHECK-NEXT: %[[VAL_1148:.*]] = llvm.inttoptr %[[VAL_1147]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1149:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1150:.*]] = llvm.insertvalue %[[VAL_1141]], %[[VAL_1149]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1151:.*]] = llvm.insertvalue %[[VAL_1148]], %[[VAL_1150]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1152:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1153:.*]] = llvm.insertvalue %[[VAL_1152]], %[[VAL_1151]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1154:.*]] = llvm.insertvalue %[[VAL_1134]], %[[VAL_1153]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1155:.*]] = llvm.insertvalue %[[VAL_1135]], %[[VAL_1154]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1156:.*]] = llvm.extractvalue %[[VAL_1155]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1157:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1156]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_23]], %[[VAL_1157]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1158:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1159:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1160:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1161:.*]] = llvm.getelementptr %[[VAL_1160]][%[[VAL_1158]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1162:.*]] = llvm.ptrtoint %[[VAL_1161]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1163:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1164:.*]] = llvm.add %[[VAL_1162]], %[[VAL_1163]] : i64
// CHECK-NEXT: %[[VAL_1165:.*]] = llvm.call @malloc(%[[VAL_1164]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1166:.*]] = llvm.ptrtoint %[[VAL_1165]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1167:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1168:.*]] = llvm.sub %[[VAL_1163]], %[[VAL_1167]] : i64
// CHECK-NEXT: %[[VAL_1169:.*]] = llvm.add %[[VAL_1166]], %[[VAL_1168]] : i64
// CHECK-NEXT: %[[VAL_1170:.*]] = llvm.urem %[[VAL_1169]], %[[VAL_1163]] : i64
// CHECK-NEXT: %[[VAL_1171:.*]] = llvm.sub %[[VAL_1169]], %[[VAL_1170]] : i64
// CHECK-NEXT: %[[VAL_1172:.*]] = llvm.inttoptr %[[VAL_1171]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1173:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1174:.*]] = llvm.insertvalue %[[VAL_1165]], %[[VAL_1173]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1175:.*]] = llvm.insertvalue %[[VAL_1172]], %[[VAL_1174]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1176:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1177:.*]] = llvm.insertvalue %[[VAL_1176]], %[[VAL_1175]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1178:.*]] = llvm.insertvalue %[[VAL_1158]], %[[VAL_1177]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1179:.*]] = llvm.insertvalue %[[VAL_1159]], %[[VAL_1178]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1180:.*]] = llvm.extractvalue %[[VAL_1051]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1181:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1180]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT: %[[VAL_1182:.*]] = llvm.load %[[VAL_1181]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[VAL_1183:.*]] = llvm.extractvalue %[[VAL_1051]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1183]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1184:.*]] = llvm.sext %[[VAL_1182]] : i32 to i64
// CHECK-NEXT: %[[VAL_1185:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_1186:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1187:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1188:.*]] = llvm.getelementptr %[[VAL_1187]][%[[VAL_1185]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1189:.*]] = llvm.ptrtoint %[[VAL_1188]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1190:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1191:.*]] = llvm.add %[[VAL_1189]], %[[VAL_1190]] : i64
// CHECK-NEXT: %[[VAL_1192:.*]] = llvm.call @malloc(%[[VAL_1191]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1193:.*]] = llvm.ptrtoint %[[VAL_1192]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1194:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1195:.*]] = llvm.sub %[[VAL_1190]], %[[VAL_1194]] : i64
// CHECK-NEXT: %[[VAL_1196:.*]] = llvm.add %[[VAL_1193]], %[[VAL_1195]] : i64
// CHECK-NEXT: %[[VAL_1197:.*]] = llvm.urem %[[VAL_1196]], %[[VAL_1190]] : i64
// CHECK-NEXT: %[[VAL_1198:.*]] = llvm.sub %[[VAL_1196]], %[[VAL_1197]] : i64
// CHECK-NEXT: %[[VAL_1199:.*]] = llvm.inttoptr %[[VAL_1198]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1200:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1201:.*]] = llvm.insertvalue %[[VAL_1192]], %[[VAL_1200]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1202:.*]] = llvm.insertvalue %[[VAL_1199]], %[[VAL_1201]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1203:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1204:.*]] = llvm.insertvalue %[[VAL_1203]], %[[VAL_1202]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1205:.*]] = llvm.insertvalue %[[VAL_1185]], %[[VAL_1204]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1206:.*]] = llvm.insertvalue %[[VAL_1186]], %[[VAL_1205]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1207:.*]] = llvm.extractvalue %[[VAL_1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1208:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1207]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_49]], %[[VAL_1208]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1209:.*]] = llvm.extractvalue %[[VAL_1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1210:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1209]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1184]], %[[VAL_1210]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1211:.*]] = llvm.extractvalue %[[VAL_1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1212:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1211]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1213:.*]] = llvm.load %[[VAL_1212]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1214:.*]] = llvm.extractvalue %[[VAL_1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1215:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1214]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1216:.*]] = llvm.load %[[VAL_1215]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1217:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_1218:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1219:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1220:.*]] = llvm.getelementptr %[[VAL_1219]][%[[VAL_1217]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1221:.*]] = llvm.ptrtoint %[[VAL_1220]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1222:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1223:.*]] = llvm.add %[[VAL_1221]], %[[VAL_1222]] : i64
// CHECK-NEXT: %[[VAL_1224:.*]] = llvm.call @malloc(%[[VAL_1223]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1225:.*]] = llvm.ptrtoint %[[VAL_1224]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1226:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1227:.*]] = llvm.sub %[[VAL_1222]], %[[VAL_1226]] : i64
// CHECK-NEXT: %[[VAL_1228:.*]] = llvm.add %[[VAL_1225]], %[[VAL_1227]] : i64
// CHECK-NEXT: %[[VAL_1229:.*]] = llvm.urem %[[VAL_1228]], %[[VAL_1222]] : i64
// CHECK-NEXT: %[[VAL_1230:.*]] = llvm.sub %[[VAL_1228]], %[[VAL_1229]] : i64
// CHECK-NEXT: %[[VAL_1231:.*]] = llvm.inttoptr %[[VAL_1230]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1232:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1233:.*]] = llvm.insertvalue %[[VAL_1224]], %[[VAL_1232]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1234:.*]] = llvm.insertvalue %[[VAL_1231]], %[[VAL_1233]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1235:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1236:.*]] = llvm.insertvalue %[[VAL_1235]], %[[VAL_1234]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1237:.*]] = llvm.insertvalue %[[VAL_1217]], %[[VAL_1236]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1238:.*]] = llvm.insertvalue %[[VAL_1218]], %[[VAL_1237]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1239:.*]] = llvm.extractvalue %[[VAL_1238]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1240:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1239]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1213]], %[[VAL_1240]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1241:.*]] = llvm.extractvalue %[[VAL_1238]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1242:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1241]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1216]], %[[VAL_1242]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1243:.*]] = llvm.extractvalue %[[VAL_1206]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1244:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1243]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1245:.*]] = llvm.load %[[VAL_1244]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1246:.*]] = llvm.extractvalue %[[VAL_1238]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1247:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1246]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1248:.*]] = llvm.load %[[VAL_1247]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1249:.*]] = llvm.mul %[[VAL_1245]], %[[VAL_17]] : i64
// CHECK-NEXT: %[[VAL_1250:.*]] = llvm.mul %[[VAL_1248]], %[[VAL_17]] : i64
// CHECK-NEXT: %[[VAL_1251:.*]] = llvm.intr.umax(%[[VAL_1249]], %[[VAL_1250]]) : (i64, i64) -> i64
// CHECK-NEXT: %[[VAL_1252:.*]] = llvm.add %[[VAL_1251]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_1253:.*]] = llvm.udiv %[[VAL_1252]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_1254:.*]] = llvm.mul %[[VAL_1253]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_1255:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[VAL_1256:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_1255]], %[[VAL_1254]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_1257:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1258:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1259:.*]] = llvm.insertvalue %[[VAL_1256]], %[[VAL_1258]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1260:.*]] = llvm.insertvalue %[[VAL_1256]], %[[VAL_1259]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1261:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1262:.*]] = llvm.insertvalue %[[VAL_1261]], %[[VAL_1260]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1263:.*]] = llvm.insertvalue %[[VAL_1254]], %[[VAL_1262]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1264:.*]] = llvm.insertvalue %[[VAL_1257]], %[[VAL_1263]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1265:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1266:.*]] = llvm.extractvalue %[[VAL_1264]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1267:.*]] = llvm.insertvalue %[[VAL_1266]], %[[VAL_1265]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1268:.*]] = llvm.extractvalue %[[VAL_1264]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1269:.*]] = llvm.getelementptr %[[VAL_1268]][%[[VAL_48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_1270:.*]] = llvm.insertvalue %[[VAL_1269]], %[[VAL_1267]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1271:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1272:.*]] = llvm.insertvalue %[[VAL_1271]], %[[VAL_1270]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1273:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_1274:.*]] = llvm.insertvalue %[[VAL_1273]], %[[VAL_1272]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1275:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1276:.*]] = llvm.insertvalue %[[VAL_1275]], %[[VAL_1274]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1277:.*]] = llvm.insertvalue %[[VAL_1248]], %[[VAL_1276]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1278:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_1279:.*]] = llvm.insertvalue %[[VAL_1278]], %[[VAL_1277]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1280:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_1281:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1282:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1283:.*]] = llvm.getelementptr %[[VAL_1282]][%[[VAL_1280]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1284:.*]] = llvm.ptrtoint %[[VAL_1283]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1285:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1286:.*]] = llvm.add %[[VAL_1284]], %[[VAL_1285]] : i64
// CHECK-NEXT: %[[VAL_1287:.*]] = llvm.call @malloc(%[[VAL_1286]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1288:.*]] = llvm.ptrtoint %[[VAL_1287]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1289:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1290:.*]] = llvm.sub %[[VAL_1285]], %[[VAL_1289]] : i64
// CHECK-NEXT: %[[VAL_1291:.*]] = llvm.add %[[VAL_1288]], %[[VAL_1290]] : i64
// CHECK-NEXT: %[[VAL_1292:.*]] = llvm.urem %[[VAL_1291]], %[[VAL_1285]] : i64
// CHECK-NEXT: %[[VAL_1293:.*]] = llvm.sub %[[VAL_1291]], %[[VAL_1292]] : i64
// CHECK-NEXT: %[[VAL_1294:.*]] = llvm.inttoptr %[[VAL_1293]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1295:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1296:.*]] = llvm.insertvalue %[[VAL_1287]], %[[VAL_1295]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1297:.*]] = llvm.insertvalue %[[VAL_1294]], %[[VAL_1296]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1298:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1299:.*]] = llvm.insertvalue %[[VAL_1298]], %[[VAL_1297]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1300:.*]] = llvm.insertvalue %[[VAL_1280]], %[[VAL_1299]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1301:.*]] = llvm.insertvalue %[[VAL_1281]], %[[VAL_1300]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1302:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1303:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1304:.*]] = llvm.getelementptr %[[VAL_1303]][%[[VAL_1302]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1305:.*]] = llvm.ptrtoint %[[VAL_1304]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1306:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1307:.*]] = llvm.add %[[VAL_1305]], %[[VAL_1306]] : i64
// CHECK-NEXT: %[[VAL_1308:.*]] = llvm.call @malloc(%[[VAL_1307]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1309:.*]] = llvm.ptrtoint %[[VAL_1308]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1310:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1311:.*]] = llvm.sub %[[VAL_1306]], %[[VAL_1310]] : i64
// CHECK-NEXT: %[[VAL_1312:.*]] = llvm.add %[[VAL_1309]], %[[VAL_1311]] : i64
// CHECK-NEXT: %[[VAL_1313:.*]] = llvm.urem %[[VAL_1312]], %[[VAL_1306]] : i64
// CHECK-NEXT: %[[VAL_1314:.*]] = llvm.sub %[[VAL_1312]], %[[VAL_1313]] : i64
// CHECK-NEXT: %[[VAL_1315:.*]] = llvm.inttoptr %[[VAL_1314]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1316:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: %[[VAL_1317:.*]] = llvm.insertvalue %[[VAL_1308]], %[[VAL_1316]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: %[[VAL_1318:.*]] = llvm.insertvalue %[[VAL_1315]], %[[VAL_1317]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: %[[VAL_1319:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1320:.*]] = llvm.insertvalue %[[VAL_1319]], %[[VAL_1318]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: %[[VAL_1321:.*]] = llvm.extractvalue %[[VAL_1017]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1322:.*]] = llvm.extractvalue %[[VAL_1017]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1323:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_1324:.*]] = llvm.insertvalue %[[VAL_1321]], %[[VAL_1323]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_1325:.*]] = llvm.insertvalue %[[VAL_1322]], %[[VAL_1324]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_1326:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1327:.*]] = llvm.insertvalue %[[VAL_1326]], %[[VAL_1325]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_1328:.*]] = llvm.extractvalue %[[VAL_1017]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1329:.*]] = llvm.extractvalue %[[VAL_1017]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1330:.*]] = llvm.extractvalue %[[VAL_1017]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1331:.*]] = llvm.extractvalue %[[VAL_1017]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1332:.*]] = llvm.extractvalue %[[VAL_1017]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1333:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1334:.*]] = llvm.extractvalue %[[VAL_1327]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_1335:.*]] = llvm.extractvalue %[[VAL_1327]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT: %[[VAL_1336:.*]] = llvm.insertvalue %[[VAL_1334]], %[[VAL_1333]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1337:.*]] = llvm.insertvalue %[[VAL_1335]], %[[VAL_1336]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1338:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1339:.*]] = llvm.insertvalue %[[VAL_1338]], %[[VAL_1337]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1340:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_1341:.*]] = llvm.insertvalue %[[VAL_1340]], %[[VAL_1339]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1342:.*]] = llvm.insertvalue %[[VAL_1331]], %[[VAL_1341]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1343:.*]] = llvm.insertvalue %[[VAL_1245]], %[[VAL_1342]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1344:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1345:.*]] = llvm.insertvalue %[[VAL_1344]], %[[VAL_1343]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1346:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1347:.*]] = llvm.extractvalue %[[VAL_1345]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1348:.*]] = llvm.mul %[[VAL_1346]], %[[VAL_1347]] : i64
// CHECK-NEXT: %[[VAL_1349:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1350:.*]] = llvm.alloca %[[VAL_1349]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1351:.*]] = llvm.getelementptr %[[VAL_1350]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1346]], %[[VAL_1351]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1352:.*]] = llvm.getelementptr %[[VAL_1350]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1347]], %[[VAL_1352]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1353:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1354:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_1355:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1356:.*]] = llvm.alloca %[[VAL_1355]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1357:.*]] = llvm.getelementptr %[[VAL_1356]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1353]], %[[VAL_1357]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1358:.*]] = llvm.getelementptr %[[VAL_1356]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1354]], %[[VAL_1358]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1359:.*]] = llvm.extractvalue %[[VAL_1345]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1360:.*]] = llvm.extractvalue %[[VAL_1279]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1361:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_1362:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT: %[[VAL_1363:.*]] = llvm.call @wrap_transpose(%arg0, %[[VAL_1359]], %[[VAL_1360]], %[[VAL_1361]], %[[VAL_1350]], %[[VAL_1356]], %[[VAL_1348]], %[[VAL_1362]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_1364:.*]] = llvm.extractvalue %[[VAL_1301]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1365:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1364]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1248]], %[[VAL_1365]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1366:.*]] = llvm.extractvalue %[[VAL_1301]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1367:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1366]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_16]], %[[VAL_1367]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1368:.*]] = llvm.extractvalue %[[VAL_1301]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1369:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1368]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1370:.*]] = llvm.load %[[VAL_1369]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1371:.*]] = llvm.extractvalue %[[VAL_1320]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: llvm.store %[[VAL_1370]], %[[VAL_1371]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1372:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1373:.*]] = llvm.extractvalue %[[VAL_1320]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: %[[VAL_1374:.*]] = llvm.extractvalue %[[VAL_1320]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: %[[VAL_1375:.*]] = llvm.insertvalue %[[VAL_1373]], %[[VAL_1372]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1376:.*]] = llvm.insertvalue %[[VAL_1374]], %[[VAL_1375]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1377:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1378:.*]] = llvm.insertvalue %[[VAL_1377]], %[[VAL_1376]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1379:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1380:.*]] = llvm.insertvalue %[[VAL_1379]], %[[VAL_1378]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1381:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1382:.*]] = llvm.insertvalue %[[VAL_1381]], %[[VAL_1380]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1383:.*]] = llvm.extractvalue %[[VAL_1206]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1383]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1384:.*]] = llvm.extractvalue %[[VAL_1238]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1384]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1385:.*]] = llvm.extractvalue %[[VAL_1301]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1385]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1386:.*]] = llvm.extractvalue %[[VAL_1155]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1386]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1387:.*]] = llvm.extractvalue %[[VAL_1179]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1387]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1388:.*]] = llvm.extractvalue %[[VAL_1131]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1388]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1389:.*]] = llvm.extractvalue %[[VAL_540]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1390:.*]] = llvm.extractvalue %[[VAL_1382]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1391:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1390]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1392:.*]] = llvm.load %[[VAL_1391]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1393:.*]] = llvm.icmp "slt" %[[VAL_1392]], %[[VAL_48]] : i64
// CHECK-NEXT: %[[VAL_1394:.*]] = llvm.add %[[VAL_1392]], %[[VAL_1389]] : i64
// CHECK-NEXT: %[[VAL_1395:.*]] = llvm.select %[[VAL_1393]], %[[VAL_1394]], %[[VAL_1392]] : i1, i64
// CHECK-NEXT: %[[VAL_1396:.*]] = llvm.intr.smin(%[[VAL_1389]], %[[VAL_48]]) : (i64, i64) -> i64
// CHECK-NEXT: %[[VAL_1397:.*]] = llvm.intr.smax(%[[VAL_1395]], %[[VAL_48]]) : (i64, i64) -> i64
// CHECK-NEXT: %[[VAL_1398:.*]] = llvm.intr.smin(%[[VAL_1397]], %[[VAL_1389]]) : (i64, i64) -> i64
// CHECK-NEXT: %[[VAL_1399:.*]] = llvm.sub %[[VAL_1398]], %[[VAL_1396]] : i64
// CHECK-NEXT: %[[VAL_1400:.*]] = llvm.intr.smax(%[[VAL_1399]], %[[VAL_48]]) : (i64, i64) -> i64
// CHECK-NEXT: %[[VAL_1401:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1402:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1403:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1404:.*]] = llvm.getelementptr %[[VAL_1403]][%[[VAL_1401]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1405:.*]] = llvm.ptrtoint %[[VAL_1404]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1406:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1407:.*]] = llvm.add %[[VAL_1405]], %[[VAL_1406]] : i64
// CHECK-NEXT: %[[VAL_1408:.*]] = llvm.call @malloc(%[[VAL_1407]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1409:.*]] = llvm.ptrtoint %[[VAL_1408]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1410:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1411:.*]] = llvm.sub %[[VAL_1406]], %[[VAL_1410]] : i64
// CHECK-NEXT: %[[VAL_1412:.*]] = llvm.add %[[VAL_1409]], %[[VAL_1411]] : i64
// CHECK-NEXT: %[[VAL_1413:.*]] = llvm.urem %[[VAL_1412]], %[[VAL_1406]] : i64
// CHECK-NEXT: %[[VAL_1414:.*]] = llvm.sub %[[VAL_1412]], %[[VAL_1413]] : i64
// CHECK-NEXT: %[[VAL_1415:.*]] = llvm.inttoptr %[[VAL_1414]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1416:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1417:.*]] = llvm.insertvalue %[[VAL_1408]], %[[VAL_1416]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1418:.*]] = llvm.insertvalue %[[VAL_1415]], %[[VAL_1417]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1419:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1420:.*]] = llvm.insertvalue %[[VAL_1419]], %[[VAL_1418]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1421:.*]] = llvm.insertvalue %[[VAL_1401]], %[[VAL_1420]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1422:.*]] = llvm.insertvalue %[[VAL_1402]], %[[VAL_1421]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1423:.*]] = llvm.extractvalue %[[VAL_1422]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1424:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1423]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1400]], %[[VAL_1424]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1425:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT: %[[VAL_1426:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1427:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1428:.*]] = llvm.getelementptr %[[VAL_1427]][%[[VAL_1425]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1429:.*]] = llvm.ptrtoint %[[VAL_1428]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1430:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_1431:.*]] = llvm.add %[[VAL_1429]], %[[VAL_1430]] : i64
// CHECK-NEXT: %[[VAL_1432:.*]] = llvm.call @malloc(%[[VAL_1431]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1433:.*]] = llvm.ptrtoint %[[VAL_1432]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1434:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1435:.*]] = llvm.sub %[[VAL_1430]], %[[VAL_1434]] : i64
// CHECK-NEXT: %[[VAL_1436:.*]] = llvm.add %[[VAL_1433]], %[[VAL_1435]] : i64
// CHECK-NEXT: %[[VAL_1437:.*]] = llvm.urem %[[VAL_1436]], %[[VAL_1430]] : i64
// CHECK-NEXT: %[[VAL_1438:.*]] = llvm.sub %[[VAL_1436]], %[[VAL_1437]] : i64
// CHECK-NEXT: %[[VAL_1439:.*]] = llvm.inttoptr %[[VAL_1438]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_1440:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1441:.*]] = llvm.insertvalue %[[VAL_1432]], %[[VAL_1440]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1442:.*]] = llvm.insertvalue %[[VAL_1439]], %[[VAL_1441]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1443:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1444:.*]] = llvm.insertvalue %[[VAL_1443]], %[[VAL_1442]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1445:.*]] = llvm.insertvalue %[[VAL_1425]], %[[VAL_1444]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1446:.*]] = llvm.insertvalue %[[VAL_1426]], %[[VAL_1445]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1447:.*]] = llvm.extractvalue %[[VAL_1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1448:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1447]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_439]], %[[VAL_1448]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1449:.*]] = llvm.extractvalue %[[VAL_1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1450:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1449]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_442]], %[[VAL_1450]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1451:.*]] = llvm.extractvalue %[[VAL_1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1452:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1451]][%[[VAL_23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_24]], %[[VAL_1452]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1453:.*]] = llvm.extractvalue %[[VAL_1422]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1454:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1453]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1455:.*]] = llvm.load %[[VAL_1454]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1456:.*]] = llvm.mul %[[VAL_1455]], %[[VAL_23]] : i64
// CHECK-NEXT: %[[VAL_1457:.*]] = llvm.add %[[VAL_1456]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_1458:.*]] = llvm.udiv %[[VAL_1457]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_1459:.*]] = llvm.mul %[[VAL_1458]], %[[VAL_21]] : i64
// CHECK-NEXT: %[[VAL_1460:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %[[VAL_1461:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_1460]], %[[VAL_1459]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_1462:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1463:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1464:.*]] = llvm.insertvalue %[[VAL_1461]], %[[VAL_1463]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1465:.*]] = llvm.insertvalue %[[VAL_1461]], %[[VAL_1464]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1466:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1467:.*]] = llvm.insertvalue %[[VAL_1466]], %[[VAL_1465]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1468:.*]] = llvm.insertvalue %[[VAL_1459]], %[[VAL_1467]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1469:.*]] = llvm.insertvalue %[[VAL_1462]], %[[VAL_1468]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1470:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1471:.*]] = llvm.extractvalue %[[VAL_1469]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1472:.*]] = llvm.insertvalue %[[VAL_1471]], %[[VAL_1470]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1473:.*]] = llvm.extractvalue %[[VAL_1469]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1474:.*]] = llvm.getelementptr %[[VAL_1473]][%[[VAL_48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_1475:.*]] = llvm.insertvalue %[[VAL_1474]], %[[VAL_1472]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1476:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1477:.*]] = llvm.insertvalue %[[VAL_1476]], %[[VAL_1475]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1478:.*]] = llvm.insertvalue %[[VAL_1455]], %[[VAL_1477]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1479:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1480:.*]] = llvm.insertvalue %[[VAL_1479]], %[[VAL_1478]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1481:.*]] = llvm.extractvalue %[[VAL_1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1482:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1481]][%[[VAL_48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1483:.*]] = llvm.load %[[VAL_1482]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1484:.*]] = llvm.extractvalue %[[VAL_1446]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1485:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_1484]][%[[VAL_47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_1486:.*]] = llvm.load %[[VAL_1485]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_1487:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT: %[[VAL_1488:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_1489:.*]] = llvm.mul %[[VAL_1487]], %[[VAL_1486]] : i64
// CHECK-NEXT: %[[VAL_1490:.*]] = llvm.mul %[[VAL_1489]], %[[VAL_1483]] : i64
// CHECK-NEXT: %[[VAL_1491:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_1492:.*]] = llvm.getelementptr %[[VAL_1491]][%[[VAL_1490]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-NEXT: %[[VAL_1493:.*]] = llvm.ptrtoint %[[VAL_1492]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_1494:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1495:.*]] = llvm.alloca %[[VAL_1494]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1496:.*]] = llvm.getelementptr %[[VAL_1495]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1483]], %[[VAL_1496]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1497:.*]] = llvm.getelementptr %[[VAL_1495]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1486]], %[[VAL_1497]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1498:.*]] = llvm.getelementptr %[[VAL_1495]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1487]], %[[VAL_1498]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1499:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_1500:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1501:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_1502:.*]] = llvm.call @hipdnn_ep_alloc_output(%arg0, %[[VAL_1499]], %[[VAL_1495]], %[[VAL_1500]], %[[VAL_1501]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1503:.*]] = llvm.addrspacecast %[[VAL_1502]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_1504:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1505:.*]] = llvm.insertvalue %[[VAL_1503]], %[[VAL_1504]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1506:.*]] = llvm.insertvalue %[[VAL_1503]], %[[VAL_1505]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1507:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_1508:.*]] = llvm.insertvalue %[[VAL_1507]], %[[VAL_1506]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1509:.*]] = llvm.insertvalue %[[VAL_1483]], %[[VAL_1508]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1510:.*]] = llvm.insertvalue %[[VAL_1486]], %[[VAL_1509]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1511:.*]] = llvm.insertvalue %[[VAL_1487]], %[[VAL_1510]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1512:.*]] = llvm.insertvalue %[[VAL_1489]], %[[VAL_1511]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1513:.*]] = llvm.insertvalue %[[VAL_1487]], %[[VAL_1512]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1514:.*]] = llvm.insertvalue %[[VAL_1488]], %[[VAL_1513]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1515:.*]] = llvm.extractvalue %[[VAL_540]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1516:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1517:.*]] = llvm.alloca %[[VAL_1516]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1518:.*]] = llvm.getelementptr %[[VAL_1517]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1515]], %[[VAL_1518]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1519:.*]] = llvm.extractvalue %[[VAL_1480]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1520:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1521:.*]] = llvm.alloca %[[VAL_1520]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1522:.*]] = llvm.getelementptr %[[VAL_1521]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1519]], %[[VAL_1522]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1523:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_1524:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1525:.*]] = llvm.alloca %[[VAL_1524]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1526:.*]] = llvm.getelementptr %[[VAL_1525]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1523]], %[[VAL_1526]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1527:.*]] = llvm.extractvalue %[[VAL_1382]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1528:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_1529:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1530:.*]] = llvm.alloca %[[VAL_1529]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1531:.*]] = llvm.getelementptr %[[VAL_1530]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1528]], %[[VAL_1531]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1532:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1533:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1534:.*]] = llvm.alloca %[[VAL_1533]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1535:.*]] = llvm.getelementptr %[[VAL_1534]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1532]], %[[VAL_1535]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1536:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1537:.*]] = llvm.extractvalue %[[VAL_540]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1538:.*]] = llvm.extractvalue %[[VAL_1480]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1539:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1540:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1541:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1542:.*]] = llvm.call @wrap_slice(%arg0, %[[VAL_1537]], %[[VAL_1525]], %[[VAL_1527]], %[[VAL_1530]], %[[VAL_1534]], %[[VAL_1538]], %[[VAL_1517]], %[[VAL_1539]], %[[VAL_1521]], %[[VAL_1540]], %[[VAL_1536]], %[[VAL_1536]], %[[VAL_1536]], %[[VAL_1541]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_1543:.*]] = llvm.extractvalue %[[VAL_1320]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1543]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1544:.*]] = llvm.extractvalue %[[VAL_494]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1545:.*]] = llvm.extractvalue %[[VAL_494]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1546:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT: %[[VAL_1547:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1548:.*]] = llvm.alloca %[[VAL_1547]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1549:.*]] = llvm.getelementptr %[[VAL_1548]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1544]], %[[VAL_1549]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1550:.*]] = llvm.getelementptr %[[VAL_1548]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1545]], %[[VAL_1550]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1551:.*]] = llvm.getelementptr %[[VAL_1548]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1546]], %[[VAL_1551]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1552:.*]] = llvm.extractvalue %[[VAL_1279]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1553:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1554:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1555:.*]] = llvm.alloca %[[VAL_1554]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1556:.*]] = llvm.getelementptr %[[VAL_1555]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1552]], %[[VAL_1556]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1557:.*]] = llvm.getelementptr %[[VAL_1555]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1553]], %[[VAL_1557]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1558:.*]] = llvm.extractvalue %[[VAL_1480]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1559:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1560:.*]] = llvm.alloca %[[VAL_1559]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1561:.*]] = llvm.getelementptr %[[VAL_1560]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1558]], %[[VAL_1561]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1562:.*]] = llvm.extractvalue %[[VAL_1514]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1563:.*]] = llvm.extractvalue %[[VAL_1514]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1564:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT: %[[VAL_1565:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1566:.*]] = llvm.alloca %[[VAL_1565]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_1567:.*]] = llvm.getelementptr %[[VAL_1566]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1562]], %[[VAL_1567]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1568:.*]] = llvm.getelementptr %[[VAL_1566]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1563]], %[[VAL_1568]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1569:.*]] = llvm.getelementptr %[[VAL_1566]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_1564]], %[[VAL_1569]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_1570:.*]] = llvm.extractvalue %[[VAL_494]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1571:.*]] = llvm.extractvalue %[[VAL_1279]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1572:.*]] = llvm.extractvalue %[[VAL_1480]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_1573:.*]] = llvm.extractvalue %[[VAL_1514]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[VAL_1574:.*]] = llvm.mlir.zero : !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_1575:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1576:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_1577:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1578:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_1579:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_1580:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_1581:.*]] = llvm.call @wrap_scatter_nd(%arg0, %[[VAL_1570]], %[[VAL_1571]], %[[VAL_1572]], %[[VAL_1573]], %[[VAL_1574]], %[[VAL_1548]], %[[VAL_1575]], %[[VAL_1555]], %[[VAL_1576]], %[[VAL_1560]], %[[VAL_1577]], %[[VAL_1566]], %[[VAL_1578]], %[[VAL_1579]], %[[VAL_1580]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_1582:.*]] = llvm.extractvalue %[[VAL_1422]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1582]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_1583:.*]] = llvm.extractvalue %[[VAL_1446]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_1583]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: llvm.return %[[VAL_1514]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_init(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(216 : i64) : i64
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.call @hipdnn_ep_inference_init(%arg0, %arg1, %[[VAL_0]], %[[VAL_1]], %arg2, %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_3]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_compute(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__hipsr_input_ranks : !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.addressof @main_graph : !llvm.ptr
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.call @hipdnn_ep_inference_compute(%arg0, %arg1, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_3]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_cleanup(%arg0: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.call @hipdnn_ep_state_cleanup(%arg0) : (!llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_0]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_get_metadata_json() -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__metadata_json : !llvm.ptr
// CHECK-NEXT: llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }

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
