// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The hipsr pipeline on a graph where every extent is static. Each extent is a
// constant, so no allocation reads its size back out of a shape buffer.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// CHECK-LABEL: module attributes {hip.constants_file = "constants.bin", hipdnn.constant_offsets = array<i64: 0, 64>, hipdnn.constant_sizes = array<i64: 32, 6>, hipdnn.input_ranks = array<i64: 2, 2>, hipdnn.num_op_state_slots = 2 : i32} {
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_json(
// CHECK-NEXT: llvm.func @hipdnn_ep_state_cleanup(!llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__hipsr_input_ranks(dense<2> : tensor<2xi64>) {addr_space = 0 : i32} : !llvm.array<2 x i64>
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_compute(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_blob(
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.func @wrap_expand(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT: llvm.func @wrap_cast(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.func @wrap_hipblasLtMatmul(!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: llvm.func @free(!llvm.ptr)
// CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-NEXT: llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: llvm.func @hipdnn_ep_op_state_construct_matmul(!llvm.ptr, i32) -> i8
// CHECK-NEXT: llvm.func @hipdnn_ep_op_states_alloc(!llvm.ptr, i64) -> i8
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
// CHECK-NEXT: %[[VAL_20:.*]] = llvm.call @main_graph_internal(%arg0, %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_21:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %[[VAL_21]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func private @main_graph_internal(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: !llvm.ptr<1>, %arg9: !llvm.ptr<1>, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64) -> (!llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {onnx.name = "y"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.insertvalue %arg1, %[[VAL_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.insertvalue %arg2, %[[VAL_1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.insertvalue %arg3, %[[VAL_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.insertvalue %arg4, %[[VAL_3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_5:.*]] = llvm.insertvalue %arg6, %[[VAL_4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_6:.*]] = llvm.insertvalue %arg5, %[[VAL_5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_7:.*]] = llvm.insertvalue %arg7, %[[VAL_6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_8:.*]] = llvm.mlir.constant(512 : index) : i64
// CHECK-NEXT: %[[VAL_9:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT: %[[VAL_10:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_11:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_12:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_13:.*]] = llvm.call @hipdnn_ep_constant_get(%arg0, %[[VAL_12]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_14:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_15:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_16:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_17:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_18:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_18]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_20:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_19]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_21:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_21]], %[[VAL_20]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_22]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_23]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_24]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_25]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_27:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_28:.*]] = llvm.call @hipdnn_ep_constant_get(%arg0, %[[VAL_27]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_29:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_30:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_31:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_32:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_33:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_34:.*]] = llvm.insertvalue %[[VAL_28]], %[[VAL_33]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_28]], %[[VAL_34]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_36:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_35]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_38:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_37]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_39:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_38]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_32]], %[[VAL_39]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_41:.*]] = llvm.insertvalue %[[VAL_31]], %[[VAL_40]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_42:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_43:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_44:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_45:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_46:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_47:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_48:.*]] = llvm.getelementptr %[[VAL_47]][%[[VAL_45]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_49:.*]] = llvm.ptrtoint %[[VAL_48]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_50:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_51:.*]] = llvm.add %[[VAL_49]], %[[VAL_50]] : i64
// CHECK-NEXT: %[[VAL_52:.*]] = llvm.call @malloc(%[[VAL_51]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_53:.*]] = llvm.ptrtoint %[[VAL_52]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_54:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_55:.*]] = llvm.sub %[[VAL_50]], %[[VAL_54]] : i64
// CHECK-NEXT: %[[VAL_56:.*]] = llvm.add %[[VAL_53]], %[[VAL_55]] : i64
// CHECK-NEXT: %[[VAL_57:.*]] = llvm.urem %[[VAL_56]], %[[VAL_50]] : i64
// CHECK-NEXT: %[[VAL_58:.*]] = llvm.sub %[[VAL_56]], %[[VAL_57]] : i64
// CHECK-NEXT: %[[VAL_59:.*]] = llvm.inttoptr %[[VAL_58]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_60:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_61:.*]] = llvm.insertvalue %[[VAL_52]], %[[VAL_60]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_62:.*]] = llvm.insertvalue %[[VAL_59]], %[[VAL_61]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_63:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_64:.*]] = llvm.insertvalue %[[VAL_63]], %[[VAL_62]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_65:.*]] = llvm.insertvalue %[[VAL_45]], %[[VAL_64]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_66:.*]] = llvm.insertvalue %[[VAL_46]], %[[VAL_65]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_67:.*]] = llvm.extractvalue %[[VAL_66]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_68:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_67]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_44]], %[[VAL_68]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_69:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_70:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_71:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_72:.*]] = llvm.getelementptr %[[VAL_71]][%[[VAL_69]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_73:.*]] = llvm.ptrtoint %[[VAL_72]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_74:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_75:.*]] = llvm.add %[[VAL_73]], %[[VAL_74]] : i64
// CHECK-NEXT: %[[VAL_76:.*]] = llvm.call @malloc(%[[VAL_75]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_77:.*]] = llvm.ptrtoint %[[VAL_76]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_78:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_79:.*]] = llvm.sub %[[VAL_74]], %[[VAL_78]] : i64
// CHECK-NEXT: %[[VAL_80:.*]] = llvm.add %[[VAL_77]], %[[VAL_79]] : i64
// CHECK-NEXT: %[[VAL_81:.*]] = llvm.urem %[[VAL_80]], %[[VAL_74]] : i64
// CHECK-NEXT: %[[VAL_82:.*]] = llvm.sub %[[VAL_80]], %[[VAL_81]] : i64
// CHECK-NEXT: %[[VAL_83:.*]] = llvm.inttoptr %[[VAL_82]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_84:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_85:.*]] = llvm.insertvalue %[[VAL_76]], %[[VAL_84]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_86:.*]] = llvm.insertvalue %[[VAL_83]], %[[VAL_85]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_87:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_88:.*]] = llvm.insertvalue %[[VAL_87]], %[[VAL_86]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_89:.*]] = llvm.insertvalue %[[VAL_69]], %[[VAL_88]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_90:.*]] = llvm.insertvalue %[[VAL_70]], %[[VAL_89]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_91:.*]] = llvm.extractvalue %[[VAL_90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_92:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_91]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_44]], %[[VAL_92]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_93:.*]] = llvm.extractvalue %[[VAL_90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_94:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_93]][%[[VAL_42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_42]], %[[VAL_94]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_95:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_96:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_97:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_98:.*]] = llvm.getelementptr %[[VAL_97]][%[[VAL_95]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_99:.*]] = llvm.ptrtoint %[[VAL_98]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_100:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_101:.*]] = llvm.add %[[VAL_99]], %[[VAL_100]] : i64
// CHECK-NEXT: %[[VAL_102:.*]] = llvm.call @malloc(%[[VAL_101]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_103:.*]] = llvm.ptrtoint %[[VAL_102]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_104:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_105:.*]] = llvm.sub %[[VAL_100]], %[[VAL_104]] : i64
// CHECK-NEXT: %[[VAL_106:.*]] = llvm.add %[[VAL_103]], %[[VAL_105]] : i64
// CHECK-NEXT: %[[VAL_107:.*]] = llvm.urem %[[VAL_106]], %[[VAL_100]] : i64
// CHECK-NEXT: %[[VAL_108:.*]] = llvm.sub %[[VAL_106]], %[[VAL_107]] : i64
// CHECK-NEXT: %[[VAL_109:.*]] = llvm.inttoptr %[[VAL_108]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_110:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_111:.*]] = llvm.insertvalue %[[VAL_102]], %[[VAL_110]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_112:.*]] = llvm.insertvalue %[[VAL_109]], %[[VAL_111]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_113:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_114:.*]] = llvm.insertvalue %[[VAL_113]], %[[VAL_112]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_115:.*]] = llvm.insertvalue %[[VAL_95]], %[[VAL_114]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_116:.*]] = llvm.insertvalue %[[VAL_96]], %[[VAL_115]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_117:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_118:.*]] = llvm.extractvalue %[[VAL_90]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_119:.*]] = llvm.mul %[[VAL_117]], %[[VAL_118]] : i64
// CHECK-NEXT: %[[VAL_120:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_121:.*]] = llvm.getelementptr %[[VAL_120]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_122:.*]] = llvm.ptrtoint %[[VAL_121]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_123:.*]] = llvm.mul %[[VAL_119]], %[[VAL_122]] : i64
// CHECK-NEXT: %[[VAL_124:.*]] = llvm.extractvalue %[[VAL_90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_125:.*]] = llvm.extractvalue %[[VAL_90]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_126:.*]] = llvm.getelementptr %[[VAL_124]][%[[VAL_125]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_127:.*]] = llvm.extractvalue %[[VAL_116]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_128:.*]] = llvm.extractvalue %[[VAL_116]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_129:.*]] = llvm.getelementptr %[[VAL_127]][%[[VAL_128]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[VAL_129]], %[[VAL_126]], %[[VAL_123]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[VAL_130:.*]] = llvm.extractvalue %[[VAL_90]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_130]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_131:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_132:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_131]], %[[VAL_8]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_133:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_134:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_135:.*]] = llvm.insertvalue %[[VAL_132]], %[[VAL_134]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_136:.*]] = llvm.insertvalue %[[VAL_132]], %[[VAL_135]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_137:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_138:.*]] = llvm.insertvalue %[[VAL_137]], %[[VAL_136]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_139:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_138]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_140:.*]] = llvm.insertvalue %[[VAL_133]], %[[VAL_139]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_141:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_142:.*]] = llvm.extractvalue %[[VAL_140]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_143:.*]] = llvm.insertvalue %[[VAL_142]], %[[VAL_141]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_144:.*]] = llvm.extractvalue %[[VAL_140]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_145:.*]] = llvm.getelementptr %[[VAL_144]][%[[VAL_43]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_146:.*]] = llvm.insertvalue %[[VAL_145]], %[[VAL_143]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_147:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_148:.*]] = llvm.insertvalue %[[VAL_147]], %[[VAL_146]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_149:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_150:.*]] = llvm.insertvalue %[[VAL_149]], %[[VAL_148]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_151:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_152:.*]] = llvm.insertvalue %[[VAL_151]], %[[VAL_150]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_153:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_154:.*]] = llvm.insertvalue %[[VAL_153]], %[[VAL_152]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_155:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_156:.*]] = llvm.insertvalue %[[VAL_155]], %[[VAL_154]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_157:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_158:.*]] = llvm.extractvalue %[[VAL_140]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_159:.*]] = llvm.insertvalue %[[VAL_158]], %[[VAL_157]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_160:.*]] = llvm.extractvalue %[[VAL_140]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_161:.*]] = llvm.getelementptr %[[VAL_160]][%[[VAL_9]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_162:.*]] = llvm.insertvalue %[[VAL_161]], %[[VAL_159]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_163:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_164:.*]] = llvm.insertvalue %[[VAL_163]], %[[VAL_162]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_165:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_166:.*]] = llvm.insertvalue %[[VAL_165]], %[[VAL_164]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_167:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_168:.*]] = llvm.insertvalue %[[VAL_167]], %[[VAL_166]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_169:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_170:.*]] = llvm.insertvalue %[[VAL_169]], %[[VAL_168]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_171:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_172:.*]] = llvm.insertvalue %[[VAL_171]], %[[VAL_170]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_173:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_174:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_175:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_176:.*]] = llvm.getelementptr %[[VAL_175]][%[[VAL_173]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_177:.*]] = llvm.ptrtoint %[[VAL_176]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_178:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_179:.*]] = llvm.add %[[VAL_177]], %[[VAL_178]] : i64
// CHECK-NEXT: %[[VAL_180:.*]] = llvm.call @malloc(%[[VAL_179]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_181:.*]] = llvm.ptrtoint %[[VAL_180]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_182:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_183:.*]] = llvm.sub %[[VAL_178]], %[[VAL_182]] : i64
// CHECK-NEXT: %[[VAL_184:.*]] = llvm.add %[[VAL_181]], %[[VAL_183]] : i64
// CHECK-NEXT: %[[VAL_185:.*]] = llvm.urem %[[VAL_184]], %[[VAL_178]] : i64
// CHECK-NEXT: %[[VAL_186:.*]] = llvm.sub %[[VAL_184]], %[[VAL_185]] : i64
// CHECK-NEXT: %[[VAL_187:.*]] = llvm.inttoptr %[[VAL_186]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_188:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_189:.*]] = llvm.insertvalue %[[VAL_180]], %[[VAL_188]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_190:.*]] = llvm.insertvalue %[[VAL_187]], %[[VAL_189]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_191:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_192:.*]] = llvm.insertvalue %[[VAL_191]], %[[VAL_190]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_193:.*]] = llvm.insertvalue %[[VAL_173]], %[[VAL_192]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_194:.*]] = llvm.insertvalue %[[VAL_174]], %[[VAL_193]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_195:.*]] = llvm.extractvalue %[[VAL_7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_196:.*]] = llvm.extractvalue %[[VAL_7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_197:.*]] = llvm.extractvalue %[[VAL_41]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_198:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_199:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_200:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_201:.*]] = llvm.extractvalue %[[VAL_7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_202:.*]] = llvm.extractvalue %[[VAL_41]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_203:.*]] = llvm.extractvalue %[[VAL_156]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_204:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_205:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_206:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_207:.*]] = llvm.call @wrap_hipblasLtMatmul(%arg0, %[[VAL_200]], %[[VAL_201]], %[[VAL_202]], %[[VAL_203]], %[[VAL_195]], %[[VAL_197]], %[[VAL_196]], %[[VAL_198]], %[[VAL_204]], %[[VAL_199]], %[[VAL_205]], %[[VAL_206]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_208:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_209:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_210:.*]] = llvm.mul %[[VAL_208]], %[[VAL_209]] : i64
// CHECK-NEXT: %[[VAL_211:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_212:.*]] = llvm.mul %[[VAL_210]], %[[VAL_211]] : i64
// CHECK-NEXT: %[[VAL_213:.*]] = llvm.extractvalue %[[VAL_156]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_214:.*]] = llvm.extractvalue %[[VAL_172]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_215:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_216:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_217:.*]] = llvm.call @wrap_cast(%arg0, %[[VAL_213]], %[[VAL_214]], %[[VAL_212]], %[[VAL_215]], %[[VAL_216]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_218:.*]] = llvm.extractvalue %[[VAL_194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_219:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_218]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_10]], %[[VAL_219]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_220:.*]] = llvm.extractvalue %[[VAL_194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_221:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_220]][%[[VAL_42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_11]], %[[VAL_221]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_222:.*]] = llvm.extractvalue %[[VAL_116]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_222]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_223:.*]] = llvm.extractvalue %[[VAL_66]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_223]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_224:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_225:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_226:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_227:.*]] = llvm.getelementptr %[[VAL_226]][%[[VAL_224]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_228:.*]] = llvm.ptrtoint %[[VAL_227]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_229:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_230:.*]] = llvm.add %[[VAL_228]], %[[VAL_229]] : i64
// CHECK-NEXT: %[[VAL_231:.*]] = llvm.call @malloc(%[[VAL_230]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_232:.*]] = llvm.ptrtoint %[[VAL_231]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_233:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_234:.*]] = llvm.sub %[[VAL_229]], %[[VAL_233]] : i64
// CHECK-NEXT: %[[VAL_235:.*]] = llvm.add %[[VAL_232]], %[[VAL_234]] : i64
// CHECK-NEXT: %[[VAL_236:.*]] = llvm.urem %[[VAL_235]], %[[VAL_229]] : i64
// CHECK-NEXT: %[[VAL_237:.*]] = llvm.sub %[[VAL_235]], %[[VAL_236]] : i64
// CHECK-NEXT: %[[VAL_238:.*]] = llvm.inttoptr %[[VAL_237]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_239:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_240:.*]] = llvm.insertvalue %[[VAL_231]], %[[VAL_239]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_241:.*]] = llvm.insertvalue %[[VAL_238]], %[[VAL_240]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_242:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_243:.*]] = llvm.insertvalue %[[VAL_242]], %[[VAL_241]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_244:.*]] = llvm.insertvalue %[[VAL_224]], %[[VAL_243]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_245:.*]] = llvm.insertvalue %[[VAL_225]], %[[VAL_244]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_246:.*]] = llvm.extractvalue %[[VAL_245]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_247:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_246]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_44]], %[[VAL_247]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_248:.*]] = llvm.extractvalue %[[VAL_245]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_249:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_248]][%[[VAL_42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_42]], %[[VAL_249]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_250:.*]] = llvm.extractvalue %[[VAL_194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_251:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_250]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_252:.*]] = llvm.load %[[VAL_251]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_253:.*]] = llvm.extractvalue %[[VAL_194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_254:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_253]][%[[VAL_42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_255:.*]] = llvm.load %[[VAL_254]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_256:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_257:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_258:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_259:.*]] = llvm.getelementptr %[[VAL_258]][%[[VAL_256]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_260:.*]] = llvm.ptrtoint %[[VAL_259]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_261:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_262:.*]] = llvm.add %[[VAL_260]], %[[VAL_261]] : i64
// CHECK-NEXT: %[[VAL_263:.*]] = llvm.call @malloc(%[[VAL_262]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_264:.*]] = llvm.ptrtoint %[[VAL_263]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_265:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_266:.*]] = llvm.sub %[[VAL_261]], %[[VAL_265]] : i64
// CHECK-NEXT: %[[VAL_267:.*]] = llvm.add %[[VAL_264]], %[[VAL_266]] : i64
// CHECK-NEXT: %[[VAL_268:.*]] = llvm.urem %[[VAL_267]], %[[VAL_261]] : i64
// CHECK-NEXT: %[[VAL_269:.*]] = llvm.sub %[[VAL_267]], %[[VAL_268]] : i64
// CHECK-NEXT: %[[VAL_270:.*]] = llvm.inttoptr %[[VAL_269]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_271:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_272:.*]] = llvm.insertvalue %[[VAL_263]], %[[VAL_271]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_273:.*]] = llvm.insertvalue %[[VAL_270]], %[[VAL_272]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_274:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_275:.*]] = llvm.insertvalue %[[VAL_274]], %[[VAL_273]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_276:.*]] = llvm.insertvalue %[[VAL_256]], %[[VAL_275]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_277:.*]] = llvm.insertvalue %[[VAL_257]], %[[VAL_276]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_278:.*]] = llvm.extractvalue %[[VAL_277]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_279:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_278]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_252]], %[[VAL_279]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_280:.*]] = llvm.extractvalue %[[VAL_277]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_281:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_280]][%[[VAL_42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_255]], %[[VAL_281]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_282:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_283:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_284:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_285:.*]] = llvm.getelementptr %[[VAL_284]][%[[VAL_282]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_286:.*]] = llvm.ptrtoint %[[VAL_285]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_287:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_288:.*]] = llvm.add %[[VAL_286]], %[[VAL_287]] : i64
// CHECK-NEXT: %[[VAL_289:.*]] = llvm.call @malloc(%[[VAL_288]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_290:.*]] = llvm.ptrtoint %[[VAL_289]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_291:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_292:.*]] = llvm.sub %[[VAL_287]], %[[VAL_291]] : i64
// CHECK-NEXT: %[[VAL_293:.*]] = llvm.add %[[VAL_290]], %[[VAL_292]] : i64
// CHECK-NEXT: %[[VAL_294:.*]] = llvm.urem %[[VAL_293]], %[[VAL_287]] : i64
// CHECK-NEXT: %[[VAL_295:.*]] = llvm.sub %[[VAL_293]], %[[VAL_294]] : i64
// CHECK-NEXT: %[[VAL_296:.*]] = llvm.inttoptr %[[VAL_295]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_297:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_298:.*]] = llvm.insertvalue %[[VAL_289]], %[[VAL_297]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_299:.*]] = llvm.insertvalue %[[VAL_296]], %[[VAL_298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_300:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_301:.*]] = llvm.insertvalue %[[VAL_300]], %[[VAL_299]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_302:.*]] = llvm.insertvalue %[[VAL_282]], %[[VAL_301]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_303:.*]] = llvm.insertvalue %[[VAL_283]], %[[VAL_302]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.br ^bb1(%[[VAL_43]] : i64)
// CHECK-NEXT: ^bb1(%[[VAL_304:.*]]: i64):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT: %[[VAL_305:.*]] = llvm.icmp "slt" %[[VAL_304]], %[[VAL_44]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_305]], ^bb2, ^bb7
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %[[VAL_306:.*]] = llvm.icmp "ult" %[[VAL_304]], %[[VAL_43]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_306]], ^bb3, ^bb4
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT: llvm.br ^bb5(%[[VAL_42]] : i64)
// CHECK-NEXT: ^bb4:  // pred: ^bb2
// CHECK-NEXT: %[[VAL_307:.*]] = llvm.extractvalue %[[VAL_245]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_308:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_307]][%[[VAL_304]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_309:.*]] = llvm.load %[[VAL_308]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_310:.*]] = llvm.extractvalue %[[VAL_277]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_311:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_310]][%[[VAL_304]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_312:.*]] = llvm.load %[[VAL_311]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_313:.*]] = llvm.icmp "eq" %[[VAL_312]], %[[VAL_42]] : i64
// CHECK-NEXT: %[[VAL_314:.*]] = llvm.select %[[VAL_313]], %[[VAL_309]], %[[VAL_312]] : i1, i64
// CHECK-NEXT: llvm.br ^bb5(%[[VAL_314]] : i64)
// CHECK-NEXT: ^bb5(%[[VAL_315:.*]]: i64):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT: llvm.br ^bb6
// CHECK-NEXT: ^bb6:  // pred: ^bb5
// CHECK-NEXT: %[[VAL_316:.*]] = llvm.extractvalue %[[VAL_303]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_317:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_316]][%[[VAL_304]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_315]], %[[VAL_317]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_318:.*]] = llvm.add %[[VAL_304]], %[[VAL_42]] : i64
// CHECK-NEXT: llvm.br ^bb1(%[[VAL_318]] : i64)
// CHECK-NEXT: ^bb7:  // pred: ^bb1
// CHECK-NEXT: %[[VAL_319:.*]] = llvm.extractvalue %[[VAL_277]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_319]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_320:.*]] = llvm.extractvalue %[[VAL_245]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_320]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_321:.*]] = llvm.extractvalue %[[VAL_303]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_322:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_321]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_323:.*]] = llvm.load %[[VAL_322]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_324:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_325:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_326:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_327:.*]] = llvm.getelementptr %[[VAL_326]][%[[VAL_324]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_328:.*]] = llvm.ptrtoint %[[VAL_327]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_329:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_330:.*]] = llvm.add %[[VAL_328]], %[[VAL_329]] : i64
// CHECK-NEXT: %[[VAL_331:.*]] = llvm.call @malloc(%[[VAL_330]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_332:.*]] = llvm.ptrtoint %[[VAL_331]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_333:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_334:.*]] = llvm.sub %[[VAL_329]], %[[VAL_333]] : i64
// CHECK-NEXT: %[[VAL_335:.*]] = llvm.add %[[VAL_332]], %[[VAL_334]] : i64
// CHECK-NEXT: %[[VAL_336:.*]] = llvm.urem %[[VAL_335]], %[[VAL_329]] : i64
// CHECK-NEXT: %[[VAL_337:.*]] = llvm.sub %[[VAL_335]], %[[VAL_336]] : i64
// CHECK-NEXT: %[[VAL_338:.*]] = llvm.inttoptr %[[VAL_337]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_339:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_340:.*]] = llvm.insertvalue %[[VAL_331]], %[[VAL_339]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_341:.*]] = llvm.insertvalue %[[VAL_338]], %[[VAL_340]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_342:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_343:.*]] = llvm.insertvalue %[[VAL_342]], %[[VAL_341]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_344:.*]] = llvm.insertvalue %[[VAL_324]], %[[VAL_343]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_345:.*]] = llvm.insertvalue %[[VAL_325]], %[[VAL_344]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_346:.*]] = llvm.extractvalue %[[VAL_345]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_347:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_346]][%[[VAL_43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_323]], %[[VAL_347]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_348:.*]] = llvm.extractvalue %[[VAL_345]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_349:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_348]][%[[VAL_42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_44]], %[[VAL_349]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_350:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_351:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_352:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_353:.*]] = llvm.getelementptr %[[VAL_352]][%[[VAL_350]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_354:.*]] = llvm.ptrtoint %[[VAL_353]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_355:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_356:.*]] = llvm.add %[[VAL_354]], %[[VAL_355]] : i64
// CHECK-NEXT: %[[VAL_357:.*]] = llvm.call @malloc(%[[VAL_356]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_358:.*]] = llvm.ptrtoint %[[VAL_357]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_359:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_360:.*]] = llvm.sub %[[VAL_355]], %[[VAL_359]] : i64
// CHECK-NEXT: %[[VAL_361:.*]] = llvm.add %[[VAL_358]], %[[VAL_360]] : i64
// CHECK-NEXT: %[[VAL_362:.*]] = llvm.urem %[[VAL_361]], %[[VAL_355]] : i64
// CHECK-NEXT: %[[VAL_363:.*]] = llvm.sub %[[VAL_361]], %[[VAL_362]] : i64
// CHECK-NEXT: %[[VAL_364:.*]] = llvm.inttoptr %[[VAL_363]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_365:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_366:.*]] = llvm.insertvalue %[[VAL_357]], %[[VAL_365]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_367:.*]] = llvm.insertvalue %[[VAL_364]], %[[VAL_366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_368:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_369:.*]] = llvm.insertvalue %[[VAL_368]], %[[VAL_367]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_370:.*]] = llvm.insertvalue %[[VAL_350]], %[[VAL_369]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_371:.*]] = llvm.insertvalue %[[VAL_351]], %[[VAL_370]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_372:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_373:.*]] = llvm.extractvalue %[[VAL_345]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_374:.*]] = llvm.mul %[[VAL_372]], %[[VAL_373]] : i64
// CHECK-NEXT: %[[VAL_375:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_376:.*]] = llvm.getelementptr %[[VAL_375]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_377:.*]] = llvm.ptrtoint %[[VAL_376]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_378:.*]] = llvm.mul %[[VAL_374]], %[[VAL_377]] : i64
// CHECK-NEXT: %[[VAL_379:.*]] = llvm.extractvalue %[[VAL_345]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_380:.*]] = llvm.extractvalue %[[VAL_345]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_381:.*]] = llvm.getelementptr %[[VAL_379]][%[[VAL_380]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_382:.*]] = llvm.extractvalue %[[VAL_371]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_383:.*]] = llvm.extractvalue %[[VAL_371]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_384:.*]] = llvm.getelementptr %[[VAL_382]][%[[VAL_383]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[VAL_384]], %[[VAL_381]], %[[VAL_378]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[VAL_385:.*]] = llvm.extractvalue %[[VAL_345]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_385]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_386:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_387:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_386]], %[[VAL_9]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_388:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_389:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_390:.*]] = llvm.insertvalue %[[VAL_387]], %[[VAL_389]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_391:.*]] = llvm.insertvalue %[[VAL_387]], %[[VAL_390]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_392:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_393:.*]] = llvm.insertvalue %[[VAL_392]], %[[VAL_391]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_394:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_393]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_395:.*]] = llvm.insertvalue %[[VAL_388]], %[[VAL_394]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_396:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_397:.*]] = llvm.extractvalue %[[VAL_395]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_398:.*]] = llvm.insertvalue %[[VAL_397]], %[[VAL_396]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_399:.*]] = llvm.extractvalue %[[VAL_395]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_400:.*]] = llvm.getelementptr %[[VAL_399]][%[[VAL_43]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_401:.*]] = llvm.insertvalue %[[VAL_400]], %[[VAL_398]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_402:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_403:.*]] = llvm.insertvalue %[[VAL_402]], %[[VAL_401]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_404:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[VAL_405:.*]] = llvm.insertvalue %[[VAL_404]], %[[VAL_403]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_406:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_407:.*]] = llvm.insertvalue %[[VAL_406]], %[[VAL_405]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_408:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_409:.*]] = llvm.insertvalue %[[VAL_408]], %[[VAL_407]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_410:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[VAL_411:.*]] = llvm.insertvalue %[[VAL_410]], %[[VAL_409]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_412:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_413:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_414:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_415:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[VAL_416:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_417:.*]] = llvm.getelementptr %[[VAL_416]][%[[VAL_415]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT: %[[VAL_418:.*]] = llvm.ptrtoint %[[VAL_417]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_419:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_420:.*]] = llvm.alloca %[[VAL_419]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_421:.*]] = llvm.getelementptr %[[VAL_420]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_412]], %[[VAL_421]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_422:.*]] = llvm.getelementptr %[[VAL_420]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_413]], %[[VAL_422]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_423:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_424:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_425:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_426:.*]] = llvm.call @hipdnn_ep_alloc_output(%arg0, %[[VAL_423]], %[[VAL_420]], %[[VAL_424]], %[[VAL_425]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_427:.*]] = llvm.addrspacecast %[[VAL_426]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_428:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_429:.*]] = llvm.insertvalue %[[VAL_427]], %[[VAL_428]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_430:.*]] = llvm.insertvalue %[[VAL_427]], %[[VAL_429]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_431:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_432:.*]] = llvm.insertvalue %[[VAL_431]], %[[VAL_430]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_433:.*]] = llvm.insertvalue %[[VAL_412]], %[[VAL_432]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_434:.*]] = llvm.insertvalue %[[VAL_413]], %[[VAL_433]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_435:.*]] = llvm.insertvalue %[[VAL_413]], %[[VAL_434]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_436:.*]] = llvm.insertvalue %[[VAL_414]], %[[VAL_435]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_437:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_438:.*]] = llvm.extractvalue %[[VAL_194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_439:.*]] = llvm.alloca %[[VAL_437]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_440:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_441:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_442:.*]] = llvm.getelementptr %[[VAL_439]][%[[VAL_441]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_440]], %[[VAL_442]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_443:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_444:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_445:.*]] = llvm.getelementptr %[[VAL_439]][%[[VAL_444]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_443]], %[[VAL_445]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_446:.*]] = llvm.alloca %[[VAL_437]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_447:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_448:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_449:.*]] = llvm.getelementptr %[[VAL_446]][%[[VAL_448]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_447]], %[[VAL_449]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_450:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_451:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_452:.*]] = llvm.getelementptr %[[VAL_446]][%[[VAL_451]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_450]], %[[VAL_452]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_453:.*]] = llvm.extractvalue %[[VAL_172]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_454:.*]] = llvm.extractvalue %[[VAL_411]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_455:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_456:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_457:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_458:.*]] = llvm.call @wrap_expand(%arg0, %[[VAL_453]], %[[VAL_438]], %[[VAL_454]], %[[VAL_439]], %[[VAL_455]], %[[VAL_446]], %[[VAL_456]], %[[VAL_457]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_459:.*]] = llvm.extractvalue %[[VAL_194]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_459]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_460:.*]] = llvm.extractvalue %[[VAL_411]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_461:.*]] = llvm.extractvalue %[[VAL_411]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_462:.*]] = llvm.extractvalue %[[VAL_26]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_463:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_464:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_465:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_466:.*]] = llvm.extractvalue %[[VAL_411]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_467:.*]] = llvm.extractvalue %[[VAL_26]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_468:.*]] = llvm.extractvalue %[[VAL_436]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_469:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_470:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_471:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_472:.*]] = llvm.call @wrap_hipblasLtMatmul(%arg0, %[[VAL_465]], %[[VAL_466]], %[[VAL_467]], %[[VAL_468]], %[[VAL_460]], %[[VAL_462]], %[[VAL_461]], %[[VAL_463]], %[[VAL_469]], %[[VAL_464]], %[[VAL_470]], %[[VAL_471]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_473:.*]] = llvm.extractvalue %[[VAL_303]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_473]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_474:.*]] = llvm.extractvalue %[[VAL_371]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_474]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: llvm.return %[[VAL_436]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @hipdnn_ep_op_states_init_fn(%arg0: !llvm.ptr) -> i32 {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%arg0, %[[VAL_3]]) : (!llvm.ptr, i64) -> i8
// CHECK-NEXT: %[[VAL_5:.*]] = llvm.icmp "eq" %[[VAL_4]], %[[VAL_2]] : i8
// CHECK-NEXT: llvm.cond_br %[[VAL_5]], ^bb2, ^bb1
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT: %[[VAL_6:.*]] = llvm.call @hipdnn_ep_op_state_construct_matmul(%arg0, %[[VAL_1]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT: %[[VAL_7:.*]] = llvm.call @hipdnn_ep_op_state_construct_matmul(%arg0, %[[VAL_0]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT: llvm.return %[[VAL_1]] : i32
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT: llvm.return %[[VAL_0]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_init(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(216 : i64) : i64
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.addressof @hipdnn_ep_op_states_init_fn : !llvm.ptr
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
