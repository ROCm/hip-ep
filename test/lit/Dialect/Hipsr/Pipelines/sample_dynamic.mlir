// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The same graph as sample_static.mlir, but with a dynamic leading extent.

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
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.insertvalue %arg8, %[[VAL_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.insertvalue %arg9, %[[VAL_1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.insertvalue %arg10, %[[VAL_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.insertvalue %arg11, %[[VAL_3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_5:.*]] = llvm.insertvalue %arg13, %[[VAL_4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_6:.*]] = llvm.insertvalue %arg12, %[[VAL_5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_7:.*]] = llvm.insertvalue %arg14, %[[VAL_6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_8:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_9:.*]] = llvm.insertvalue %arg1, %[[VAL_8]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_10:.*]] = llvm.insertvalue %arg2, %[[VAL_9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_11:.*]] = llvm.insertvalue %arg3, %[[VAL_10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_12:.*]] = llvm.insertvalue %arg4, %[[VAL_11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_13:.*]] = llvm.insertvalue %arg6, %[[VAL_12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_14:.*]] = llvm.insertvalue %arg5, %[[VAL_13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_15:.*]] = llvm.insertvalue %arg7, %[[VAL_14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_16:.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK-NEXT: %[[VAL_17:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_18:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[VAL_19:.*]] = llvm.mlir.constant(255 : index) : i64
// CHECK-NEXT: %[[VAL_20:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT: %[[VAL_21:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_22:.*]] = llvm.call @hipdnn_ep_constant_get(%arg0, %[[VAL_21]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_23:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_24:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_25:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_26:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_27:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_22]], %[[VAL_27]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_22]], %[[VAL_28]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_30:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_31:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_29]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_31]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_32]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_34:.*]] = llvm.insertvalue %[[VAL_26]], %[[VAL_33]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_25]], %[[VAL_34]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_36:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_37:.*]] = llvm.call @hipdnn_ep_constant_get(%arg0, %[[VAL_36]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_38:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: %[[VAL_39:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_40:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_41:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_42:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_37]], %[[VAL_42]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_44:.*]] = llvm.insertvalue %[[VAL_37]], %[[VAL_43]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_45:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_46:.*]] = llvm.insertvalue %[[VAL_45]], %[[VAL_44]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_47:.*]] = llvm.insertvalue %[[VAL_38]], %[[VAL_46]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_48:.*]] = llvm.insertvalue %[[VAL_39]], %[[VAL_47]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_49:.*]] = llvm.insertvalue %[[VAL_41]], %[[VAL_48]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_50:.*]] = llvm.insertvalue %[[VAL_40]], %[[VAL_49]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_51:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_52:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_53:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_54:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_55:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_56:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_57:.*]] = llvm.getelementptr %[[VAL_56]][%[[VAL_54]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_58:.*]] = llvm.ptrtoint %[[VAL_57]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_59:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_60:.*]] = llvm.add %[[VAL_58]], %[[VAL_59]] : i64
// CHECK-NEXT: %[[VAL_61:.*]] = llvm.call @malloc(%[[VAL_60]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_62:.*]] = llvm.ptrtoint %[[VAL_61]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_63:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_64:.*]] = llvm.sub %[[VAL_59]], %[[VAL_63]] : i64
// CHECK-NEXT: %[[VAL_65:.*]] = llvm.add %[[VAL_62]], %[[VAL_64]] : i64
// CHECK-NEXT: %[[VAL_66:.*]] = llvm.urem %[[VAL_65]], %[[VAL_59]] : i64
// CHECK-NEXT: %[[VAL_67:.*]] = llvm.sub %[[VAL_65]], %[[VAL_66]] : i64
// CHECK-NEXT: %[[VAL_68:.*]] = llvm.inttoptr %[[VAL_67]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_69:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_70:.*]] = llvm.insertvalue %[[VAL_61]], %[[VAL_69]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_71:.*]] = llvm.insertvalue %[[VAL_68]], %[[VAL_70]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_72:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_73:.*]] = llvm.insertvalue %[[VAL_72]], %[[VAL_71]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_74:.*]] = llvm.insertvalue %[[VAL_54]], %[[VAL_73]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_75:.*]] = llvm.insertvalue %[[VAL_55]], %[[VAL_74]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_76:.*]] = llvm.extractvalue %[[VAL_75]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_77:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_76]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_53]], %[[VAL_77]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_78:.*]] = llvm.extractvalue %[[VAL_15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_79:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_80:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_81:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_82:.*]] = llvm.getelementptr %[[VAL_81]][%[[VAL_79]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_83:.*]] = llvm.ptrtoint %[[VAL_82]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_84:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_85:.*]] = llvm.add %[[VAL_83]], %[[VAL_84]] : i64
// CHECK-NEXT: %[[VAL_86:.*]] = llvm.call @malloc(%[[VAL_85]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_87:.*]] = llvm.ptrtoint %[[VAL_86]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_88:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_89:.*]] = llvm.sub %[[VAL_84]], %[[VAL_88]] : i64
// CHECK-NEXT: %[[VAL_90:.*]] = llvm.add %[[VAL_87]], %[[VAL_89]] : i64
// CHECK-NEXT: %[[VAL_91:.*]] = llvm.urem %[[VAL_90]], %[[VAL_84]] : i64
// CHECK-NEXT: %[[VAL_92:.*]] = llvm.sub %[[VAL_90]], %[[VAL_91]] : i64
// CHECK-NEXT: %[[VAL_93:.*]] = llvm.inttoptr %[[VAL_92]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_94:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_95:.*]] = llvm.insertvalue %[[VAL_86]], %[[VAL_94]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_96:.*]] = llvm.insertvalue %[[VAL_93]], %[[VAL_95]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_97:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_98:.*]] = llvm.insertvalue %[[VAL_97]], %[[VAL_96]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_99:.*]] = llvm.insertvalue %[[VAL_79]], %[[VAL_98]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_100:.*]] = llvm.insertvalue %[[VAL_80]], %[[VAL_99]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_101:.*]] = llvm.extractvalue %[[VAL_100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_102:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_101]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_78]], %[[VAL_102]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_103:.*]] = llvm.extractvalue %[[VAL_100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_104:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_103]][%[[VAL_51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_51]], %[[VAL_104]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_105:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_106:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_107:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_108:.*]] = llvm.getelementptr %[[VAL_107]][%[[VAL_105]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_109:.*]] = llvm.ptrtoint %[[VAL_108]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_110:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_111:.*]] = llvm.add %[[VAL_109]], %[[VAL_110]] : i64
// CHECK-NEXT: %[[VAL_112:.*]] = llvm.call @malloc(%[[VAL_111]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_113:.*]] = llvm.ptrtoint %[[VAL_112]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_114:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_115:.*]] = llvm.sub %[[VAL_110]], %[[VAL_114]] : i64
// CHECK-NEXT: %[[VAL_116:.*]] = llvm.add %[[VAL_113]], %[[VAL_115]] : i64
// CHECK-NEXT: %[[VAL_117:.*]] = llvm.urem %[[VAL_116]], %[[VAL_110]] : i64
// CHECK-NEXT: %[[VAL_118:.*]] = llvm.sub %[[VAL_116]], %[[VAL_117]] : i64
// CHECK-NEXT: %[[VAL_119:.*]] = llvm.inttoptr %[[VAL_118]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_120:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_121:.*]] = llvm.insertvalue %[[VAL_112]], %[[VAL_120]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_122:.*]] = llvm.insertvalue %[[VAL_119]], %[[VAL_121]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_123:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_124:.*]] = llvm.insertvalue %[[VAL_123]], %[[VAL_122]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_125:.*]] = llvm.insertvalue %[[VAL_105]], %[[VAL_124]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_126:.*]] = llvm.insertvalue %[[VAL_106]], %[[VAL_125]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_127:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_128:.*]] = llvm.extractvalue %[[VAL_100]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_129:.*]] = llvm.mul %[[VAL_127]], %[[VAL_128]] : i64
// CHECK-NEXT: %[[VAL_130:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_131:.*]] = llvm.getelementptr %[[VAL_130]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_132:.*]] = llvm.ptrtoint %[[VAL_131]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_133:.*]] = llvm.mul %[[VAL_129]], %[[VAL_132]] : i64
// CHECK-NEXT: %[[VAL_134:.*]] = llvm.extractvalue %[[VAL_100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_135:.*]] = llvm.extractvalue %[[VAL_100]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_136:.*]] = llvm.getelementptr %[[VAL_134]][%[[VAL_135]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_137:.*]] = llvm.extractvalue %[[VAL_126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_138:.*]] = llvm.extractvalue %[[VAL_126]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_139:.*]] = llvm.getelementptr %[[VAL_137]][%[[VAL_138]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[VAL_139]], %[[VAL_136]], %[[VAL_133]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[VAL_140:.*]] = llvm.extractvalue %[[VAL_100]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_140]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_141:.*]] = llvm.extractvalue %[[VAL_126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_142:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_141]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_143:.*]] = llvm.load %[[VAL_142]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_144:.*]] = llvm.mul %[[VAL_143]], %[[VAL_53]] : i64
// CHECK-NEXT: %[[VAL_145:.*]] = llvm.add %[[VAL_144]], %[[VAL_19]] : i64
// CHECK-NEXT: %[[VAL_146:.*]] = llvm.udiv %[[VAL_145]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_147:.*]] = llvm.mul %[[VAL_146]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_148:.*]] = llvm.mul %[[VAL_143]], %[[VAL_18]] : i64
// CHECK-NEXT: %[[VAL_149:.*]] = llvm.add %[[VAL_148]], %[[VAL_19]] : i64
// CHECK-NEXT: %[[VAL_150:.*]] = llvm.udiv %[[VAL_149]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_151:.*]] = llvm.mul %[[VAL_150]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_152:.*]] = llvm.add %[[VAL_147]], %[[VAL_151]] : i64
// CHECK-NEXT: %[[VAL_153:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_154:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_153]], %[[VAL_152]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_155:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_156:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_157:.*]] = llvm.insertvalue %[[VAL_154]], %[[VAL_156]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_158:.*]] = llvm.insertvalue %[[VAL_154]], %[[VAL_157]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_159:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_160:.*]] = llvm.insertvalue %[[VAL_159]], %[[VAL_158]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_161:.*]] = llvm.insertvalue %[[VAL_152]], %[[VAL_160]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_162:.*]] = llvm.insertvalue %[[VAL_155]], %[[VAL_161]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_163:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_164:.*]] = llvm.extractvalue %[[VAL_162]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_165:.*]] = llvm.insertvalue %[[VAL_164]], %[[VAL_163]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_166:.*]] = llvm.extractvalue %[[VAL_162]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_167:.*]] = llvm.getelementptr %[[VAL_166]][%[[VAL_52]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_168:.*]] = llvm.insertvalue %[[VAL_167]], %[[VAL_165]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_169:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_170:.*]] = llvm.insertvalue %[[VAL_169]], %[[VAL_168]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_171:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_172:.*]] = llvm.insertvalue %[[VAL_171]], %[[VAL_170]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_173:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_174:.*]] = llvm.insertvalue %[[VAL_173]], %[[VAL_172]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_175:.*]] = llvm.insertvalue %[[VAL_143]], %[[VAL_174]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_176:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_177:.*]] = llvm.insertvalue %[[VAL_176]], %[[VAL_175]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_178:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_179:.*]] = llvm.extractvalue %[[VAL_162]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_180:.*]] = llvm.insertvalue %[[VAL_179]], %[[VAL_178]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_181:.*]] = llvm.extractvalue %[[VAL_162]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_182:.*]] = llvm.getelementptr %[[VAL_181]][%[[VAL_147]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_183:.*]] = llvm.insertvalue %[[VAL_182]], %[[VAL_180]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_184:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_185:.*]] = llvm.insertvalue %[[VAL_184]], %[[VAL_183]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_186:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_187:.*]] = llvm.insertvalue %[[VAL_186]], %[[VAL_185]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_188:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_189:.*]] = llvm.insertvalue %[[VAL_188]], %[[VAL_187]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_190:.*]] = llvm.insertvalue %[[VAL_143]], %[[VAL_189]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_191:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_192:.*]] = llvm.insertvalue %[[VAL_191]], %[[VAL_190]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_193:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_194:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_195:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_196:.*]] = llvm.getelementptr %[[VAL_195]][%[[VAL_193]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_197:.*]] = llvm.ptrtoint %[[VAL_196]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_198:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_199:.*]] = llvm.add %[[VAL_197]], %[[VAL_198]] : i64
// CHECK-NEXT: %[[VAL_200:.*]] = llvm.call @malloc(%[[VAL_199]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_201:.*]] = llvm.ptrtoint %[[VAL_200]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_202:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_203:.*]] = llvm.sub %[[VAL_198]], %[[VAL_202]] : i64
// CHECK-NEXT: %[[VAL_204:.*]] = llvm.add %[[VAL_201]], %[[VAL_203]] : i64
// CHECK-NEXT: %[[VAL_205:.*]] = llvm.urem %[[VAL_204]], %[[VAL_198]] : i64
// CHECK-NEXT: %[[VAL_206:.*]] = llvm.sub %[[VAL_204]], %[[VAL_205]] : i64
// CHECK-NEXT: %[[VAL_207:.*]] = llvm.inttoptr %[[VAL_206]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_208:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_209:.*]] = llvm.insertvalue %[[VAL_200]], %[[VAL_208]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_210:.*]] = llvm.insertvalue %[[VAL_207]], %[[VAL_209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_211:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_212:.*]] = llvm.insertvalue %[[VAL_211]], %[[VAL_210]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_213:.*]] = llvm.insertvalue %[[VAL_193]], %[[VAL_212]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_214:.*]] = llvm.insertvalue %[[VAL_194]], %[[VAL_213]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_215:.*]] = llvm.extractvalue %[[VAL_15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_216:.*]] = llvm.extractvalue %[[VAL_15]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_217:.*]] = llvm.extractvalue %[[VAL_50]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_218:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_219:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_220:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_221:.*]] = llvm.extractvalue %[[VAL_15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_222:.*]] = llvm.extractvalue %[[VAL_50]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_223:.*]] = llvm.extractvalue %[[VAL_177]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_224:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_225:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_226:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_227:.*]] = llvm.call @wrap_hipblasLtMatmul(%arg0, %[[VAL_220]], %[[VAL_221]], %[[VAL_222]], %[[VAL_223]], %[[VAL_215]], %[[VAL_217]], %[[VAL_216]], %[[VAL_218]], %[[VAL_224]], %[[VAL_219]], %[[VAL_225]], %[[VAL_226]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_228:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_229:.*]] = llvm.extractvalue %[[VAL_192]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_230:.*]] = llvm.mul %[[VAL_228]], %[[VAL_229]] : i64
// CHECK-NEXT: %[[VAL_231:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_232:.*]] = llvm.mul %[[VAL_230]], %[[VAL_231]] : i64
// CHECK-NEXT: %[[VAL_233:.*]] = llvm.extractvalue %[[VAL_177]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_234:.*]] = llvm.extractvalue %[[VAL_192]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_235:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_236:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_237:.*]] = llvm.call @wrap_cast(%arg0, %[[VAL_233]], %[[VAL_234]], %[[VAL_232]], %[[VAL_235]], %[[VAL_236]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_238:.*]] = llvm.extractvalue %[[VAL_7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_239:.*]] = llvm.extractvalue %[[VAL_214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_240:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_239]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_238]], %[[VAL_240]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_241:.*]] = llvm.extractvalue %[[VAL_214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_242:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_241]][%[[VAL_51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_17]], %[[VAL_242]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_243:.*]] = llvm.extractvalue %[[VAL_126]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_243]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_244:.*]] = llvm.extractvalue %[[VAL_75]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_244]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_245:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_246:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_247:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_248:.*]] = llvm.getelementptr %[[VAL_247]][%[[VAL_245]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_249:.*]] = llvm.ptrtoint %[[VAL_248]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_250:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_251:.*]] = llvm.add %[[VAL_249]], %[[VAL_250]] : i64
// CHECK-NEXT: %[[VAL_252:.*]] = llvm.call @malloc(%[[VAL_251]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_253:.*]] = llvm.ptrtoint %[[VAL_252]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_254:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_255:.*]] = llvm.sub %[[VAL_250]], %[[VAL_254]] : i64
// CHECK-NEXT: %[[VAL_256:.*]] = llvm.add %[[VAL_253]], %[[VAL_255]] : i64
// CHECK-NEXT: %[[VAL_257:.*]] = llvm.urem %[[VAL_256]], %[[VAL_250]] : i64
// CHECK-NEXT: %[[VAL_258:.*]] = llvm.sub %[[VAL_256]], %[[VAL_257]] : i64
// CHECK-NEXT: %[[VAL_259:.*]] = llvm.inttoptr %[[VAL_258]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_260:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_261:.*]] = llvm.insertvalue %[[VAL_252]], %[[VAL_260]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_262:.*]] = llvm.insertvalue %[[VAL_259]], %[[VAL_261]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_263:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_264:.*]] = llvm.insertvalue %[[VAL_263]], %[[VAL_262]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_265:.*]] = llvm.insertvalue %[[VAL_245]], %[[VAL_264]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_266:.*]] = llvm.insertvalue %[[VAL_246]], %[[VAL_265]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_267:.*]] = llvm.extractvalue %[[VAL_266]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_268:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_267]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_143]], %[[VAL_268]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_269:.*]] = llvm.extractvalue %[[VAL_266]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_270:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_269]][%[[VAL_51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_51]], %[[VAL_270]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_271:.*]] = llvm.extractvalue %[[VAL_214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_272:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_271]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_273:.*]] = llvm.load %[[VAL_272]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_274:.*]] = llvm.extractvalue %[[VAL_214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_275:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_274]][%[[VAL_51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_276:.*]] = llvm.load %[[VAL_275]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_277:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_278:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_279:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_280:.*]] = llvm.getelementptr %[[VAL_279]][%[[VAL_277]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_281:.*]] = llvm.ptrtoint %[[VAL_280]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_282:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_283:.*]] = llvm.add %[[VAL_281]], %[[VAL_282]] : i64
// CHECK-NEXT: %[[VAL_284:.*]] = llvm.call @malloc(%[[VAL_283]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_285:.*]] = llvm.ptrtoint %[[VAL_284]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_286:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_287:.*]] = llvm.sub %[[VAL_282]], %[[VAL_286]] : i64
// CHECK-NEXT: %[[VAL_288:.*]] = llvm.add %[[VAL_285]], %[[VAL_287]] : i64
// CHECK-NEXT: %[[VAL_289:.*]] = llvm.urem %[[VAL_288]], %[[VAL_282]] : i64
// CHECK-NEXT: %[[VAL_290:.*]] = llvm.sub %[[VAL_288]], %[[VAL_289]] : i64
// CHECK-NEXT: %[[VAL_291:.*]] = llvm.inttoptr %[[VAL_290]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_292:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_293:.*]] = llvm.insertvalue %[[VAL_284]], %[[VAL_292]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_294:.*]] = llvm.insertvalue %[[VAL_291]], %[[VAL_293]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_295:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_296:.*]] = llvm.insertvalue %[[VAL_295]], %[[VAL_294]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_297:.*]] = llvm.insertvalue %[[VAL_277]], %[[VAL_296]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_298:.*]] = llvm.insertvalue %[[VAL_278]], %[[VAL_297]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_299:.*]] = llvm.extractvalue %[[VAL_298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_300:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_299]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_273]], %[[VAL_300]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_301:.*]] = llvm.extractvalue %[[VAL_298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_302:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_301]][%[[VAL_51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_276]], %[[VAL_302]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_303:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_304:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_305:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_306:.*]] = llvm.getelementptr %[[VAL_305]][%[[VAL_303]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_307:.*]] = llvm.ptrtoint %[[VAL_306]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_308:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_309:.*]] = llvm.add %[[VAL_307]], %[[VAL_308]] : i64
// CHECK-NEXT: %[[VAL_310:.*]] = llvm.call @malloc(%[[VAL_309]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_311:.*]] = llvm.ptrtoint %[[VAL_310]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_312:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_313:.*]] = llvm.sub %[[VAL_308]], %[[VAL_312]] : i64
// CHECK-NEXT: %[[VAL_314:.*]] = llvm.add %[[VAL_311]], %[[VAL_313]] : i64
// CHECK-NEXT: %[[VAL_315:.*]] = llvm.urem %[[VAL_314]], %[[VAL_308]] : i64
// CHECK-NEXT: %[[VAL_316:.*]] = llvm.sub %[[VAL_314]], %[[VAL_315]] : i64
// CHECK-NEXT: %[[VAL_317:.*]] = llvm.inttoptr %[[VAL_316]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_318:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_319:.*]] = llvm.insertvalue %[[VAL_310]], %[[VAL_318]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_320:.*]] = llvm.insertvalue %[[VAL_317]], %[[VAL_319]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_321:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_322:.*]] = llvm.insertvalue %[[VAL_321]], %[[VAL_320]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_323:.*]] = llvm.insertvalue %[[VAL_303]], %[[VAL_322]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_324:.*]] = llvm.insertvalue %[[VAL_304]], %[[VAL_323]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.br ^bb1(%[[VAL_52]] : i64)
// CHECK-NEXT: ^bb1(%[[VAL_325:.*]]: i64):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT: %[[VAL_326:.*]] = llvm.icmp "slt" %[[VAL_325]], %[[VAL_53]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_326]], ^bb2, ^bb7
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %[[VAL_327:.*]] = llvm.icmp "ult" %[[VAL_325]], %[[VAL_52]] : i64
// CHECK-NEXT: llvm.cond_br %[[VAL_327]], ^bb3, ^bb4
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT: llvm.br ^bb5(%[[VAL_51]] : i64)
// CHECK-NEXT: ^bb4:  // pred: ^bb2
// CHECK-NEXT: %[[VAL_328:.*]] = llvm.extractvalue %[[VAL_266]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_329:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_328]][%[[VAL_325]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_330:.*]] = llvm.load %[[VAL_329]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_331:.*]] = llvm.extractvalue %[[VAL_298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_332:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_331]][%[[VAL_325]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_333:.*]] = llvm.load %[[VAL_332]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_334:.*]] = llvm.icmp "eq" %[[VAL_333]], %[[VAL_51]] : i64
// CHECK-NEXT: %[[VAL_335:.*]] = llvm.select %[[VAL_334]], %[[VAL_330]], %[[VAL_333]] : i1, i64
// CHECK-NEXT: llvm.br ^bb5(%[[VAL_335]] : i64)
// CHECK-NEXT: ^bb5(%[[VAL_336:.*]]: i64):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT: llvm.br ^bb6
// CHECK-NEXT: ^bb6:  // pred: ^bb5
// CHECK-NEXT: %[[VAL_337:.*]] = llvm.extractvalue %[[VAL_324]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_338:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_337]][%[[VAL_325]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_336]], %[[VAL_338]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_339:.*]] = llvm.add %[[VAL_325]], %[[VAL_51]] : i64
// CHECK-NEXT: llvm.br ^bb1(%[[VAL_339]] : i64)
// CHECK-NEXT: ^bb7:  // pred: ^bb1
// CHECK-NEXT: %[[VAL_340:.*]] = llvm.extractvalue %[[VAL_298]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_340]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_341:.*]] = llvm.extractvalue %[[VAL_266]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_341]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_342:.*]] = llvm.extractvalue %[[VAL_324]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_343:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_342]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_344:.*]] = llvm.load %[[VAL_343]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_345:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_346:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_347:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_348:.*]] = llvm.getelementptr %[[VAL_347]][%[[VAL_345]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_349:.*]] = llvm.ptrtoint %[[VAL_348]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_350:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_351:.*]] = llvm.add %[[VAL_349]], %[[VAL_350]] : i64
// CHECK-NEXT: %[[VAL_352:.*]] = llvm.call @malloc(%[[VAL_351]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_353:.*]] = llvm.ptrtoint %[[VAL_352]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_354:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_355:.*]] = llvm.sub %[[VAL_350]], %[[VAL_354]] : i64
// CHECK-NEXT: %[[VAL_356:.*]] = llvm.add %[[VAL_353]], %[[VAL_355]] : i64
// CHECK-NEXT: %[[VAL_357:.*]] = llvm.urem %[[VAL_356]], %[[VAL_350]] : i64
// CHECK-NEXT: %[[VAL_358:.*]] = llvm.sub %[[VAL_356]], %[[VAL_357]] : i64
// CHECK-NEXT: %[[VAL_359:.*]] = llvm.inttoptr %[[VAL_358]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_360:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_361:.*]] = llvm.insertvalue %[[VAL_352]], %[[VAL_360]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_362:.*]] = llvm.insertvalue %[[VAL_359]], %[[VAL_361]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_363:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_364:.*]] = llvm.insertvalue %[[VAL_363]], %[[VAL_362]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_365:.*]] = llvm.insertvalue %[[VAL_345]], %[[VAL_364]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_366:.*]] = llvm.insertvalue %[[VAL_346]], %[[VAL_365]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_367:.*]] = llvm.extractvalue %[[VAL_366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_368:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_367]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_344]], %[[VAL_368]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_369:.*]] = llvm.extractvalue %[[VAL_366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_370:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_369]][%[[VAL_51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_53]], %[[VAL_370]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_371:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_372:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_373:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_374:.*]] = llvm.getelementptr %[[VAL_373]][%[[VAL_371]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_375:.*]] = llvm.ptrtoint %[[VAL_374]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_376:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %[[VAL_377:.*]] = llvm.add %[[VAL_375]], %[[VAL_376]] : i64
// CHECK-NEXT: %[[VAL_378:.*]] = llvm.call @malloc(%[[VAL_377]]) : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_379:.*]] = llvm.ptrtoint %[[VAL_378]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_380:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_381:.*]] = llvm.sub %[[VAL_376]], %[[VAL_380]] : i64
// CHECK-NEXT: %[[VAL_382:.*]] = llvm.add %[[VAL_379]], %[[VAL_381]] : i64
// CHECK-NEXT: %[[VAL_383:.*]] = llvm.urem %[[VAL_382]], %[[VAL_376]] : i64
// CHECK-NEXT: %[[VAL_384:.*]] = llvm.sub %[[VAL_382]], %[[VAL_383]] : i64
// CHECK-NEXT: %[[VAL_385:.*]] = llvm.inttoptr %[[VAL_384]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[VAL_386:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_387:.*]] = llvm.insertvalue %[[VAL_378]], %[[VAL_386]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_388:.*]] = llvm.insertvalue %[[VAL_385]], %[[VAL_387]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_389:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_390:.*]] = llvm.insertvalue %[[VAL_389]], %[[VAL_388]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_391:.*]] = llvm.insertvalue %[[VAL_371]], %[[VAL_390]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_392:.*]] = llvm.insertvalue %[[VAL_372]], %[[VAL_391]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_393:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_394:.*]] = llvm.extractvalue %[[VAL_366]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_395:.*]] = llvm.mul %[[VAL_393]], %[[VAL_394]] : i64
// CHECK-NEXT: %[[VAL_396:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_397:.*]] = llvm.getelementptr %[[VAL_396]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_398:.*]] = llvm.ptrtoint %[[VAL_397]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_399:.*]] = llvm.mul %[[VAL_395]], %[[VAL_398]] : i64
// CHECK-NEXT: %[[VAL_400:.*]] = llvm.extractvalue %[[VAL_366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_401:.*]] = llvm.extractvalue %[[VAL_366]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_402:.*]] = llvm.getelementptr %[[VAL_400]][%[[VAL_401]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_403:.*]] = llvm.extractvalue %[[VAL_392]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_404:.*]] = llvm.extractvalue %[[VAL_392]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_405:.*]] = llvm.getelementptr %[[VAL_403]][%[[VAL_404]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: "llvm.intr.memcpy"(%[[VAL_405]], %[[VAL_402]], %[[VAL_399]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT: %[[VAL_406:.*]] = llvm.extractvalue %[[VAL_366]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_406]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_407:.*]] = llvm.extractvalue %[[VAL_324]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_408:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_407]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_409:.*]] = llvm.load %[[VAL_408]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_410:.*]] = llvm.mul %[[VAL_409]], %[[VAL_16]] : i64
// CHECK-NEXT: %[[VAL_411:.*]] = llvm.add %[[VAL_410]], %[[VAL_19]] : i64
// CHECK-NEXT: %[[VAL_412:.*]] = llvm.udiv %[[VAL_411]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_413:.*]] = llvm.mul %[[VAL_412]], %[[VAL_20]] : i64
// CHECK-NEXT: %[[VAL_414:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_415:.*]] = llvm.call @hipdnn_ep_get_pool_base(%arg0, %[[VAL_414]], %[[VAL_413]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_416:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_417:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_418:.*]] = llvm.insertvalue %[[VAL_415]], %[[VAL_417]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_419:.*]] = llvm.insertvalue %[[VAL_415]], %[[VAL_418]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_420:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_421:.*]] = llvm.insertvalue %[[VAL_420]], %[[VAL_419]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_422:.*]] = llvm.insertvalue %[[VAL_413]], %[[VAL_421]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_423:.*]] = llvm.insertvalue %[[VAL_416]], %[[VAL_422]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_424:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_425:.*]] = llvm.extractvalue %[[VAL_423]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_426:.*]] = llvm.insertvalue %[[VAL_425]], %[[VAL_424]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_427:.*]] = llvm.extractvalue %[[VAL_423]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_428:.*]] = llvm.getelementptr %[[VAL_427]][%[[VAL_52]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT: %[[VAL_429:.*]] = llvm.insertvalue %[[VAL_428]], %[[VAL_426]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_430:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_431:.*]] = llvm.insertvalue %[[VAL_430]], %[[VAL_429]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_432:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[VAL_433:.*]] = llvm.insertvalue %[[VAL_432]], %[[VAL_431]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_434:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_435:.*]] = llvm.insertvalue %[[VAL_434]], %[[VAL_433]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_436:.*]] = llvm.insertvalue %[[VAL_409]], %[[VAL_435]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_437:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[VAL_438:.*]] = llvm.insertvalue %[[VAL_437]], %[[VAL_436]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_439:.*]] = llvm.extractvalue %[[VAL_392]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_440:.*]] = llvm.getelementptr inbounds|nuw %[[VAL_439]][%[[VAL_52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT: %[[VAL_441:.*]] = llvm.load %[[VAL_440]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[VAL_442:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[VAL_443:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[VAL_444:.*]] = llvm.mul %[[VAL_442]], %[[VAL_441]] : i64
// CHECK-NEXT: %[[VAL_445:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_446:.*]] = llvm.getelementptr %[[VAL_445]][%[[VAL_444]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT: %[[VAL_447:.*]] = llvm.ptrtoint %[[VAL_446]] : !llvm.ptr to i64
// CHECK-NEXT: %[[VAL_448:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_449:.*]] = llvm.alloca %[[VAL_448]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_450:.*]] = llvm.getelementptr %[[VAL_449]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_441]], %[[VAL_450]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_451:.*]] = llvm.getelementptr %[[VAL_449]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_442]], %[[VAL_451]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_452:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_453:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_454:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_455:.*]] = llvm.call @hipdnn_ep_alloc_output(%arg0, %[[VAL_452]], %[[VAL_449]], %[[VAL_453]], %[[VAL_454]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_456:.*]] = llvm.addrspacecast %[[VAL_455]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT: %[[VAL_457:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_458:.*]] = llvm.insertvalue %[[VAL_456]], %[[VAL_457]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_459:.*]] = llvm.insertvalue %[[VAL_456]], %[[VAL_458]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_460:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[VAL_461:.*]] = llvm.insertvalue %[[VAL_460]], %[[VAL_459]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_462:.*]] = llvm.insertvalue %[[VAL_441]], %[[VAL_461]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_463:.*]] = llvm.insertvalue %[[VAL_442]], %[[VAL_462]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_464:.*]] = llvm.insertvalue %[[VAL_442]], %[[VAL_463]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_465:.*]] = llvm.insertvalue %[[VAL_443]], %[[VAL_464]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_466:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_467:.*]] = llvm.extractvalue %[[VAL_214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_468:.*]] = llvm.alloca %[[VAL_466]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_469:.*]] = llvm.extractvalue %[[VAL_192]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_470:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_471:.*]] = llvm.getelementptr %[[VAL_468]][%[[VAL_470]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_469]], %[[VAL_471]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_472:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_473:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_474:.*]] = llvm.getelementptr %[[VAL_468]][%[[VAL_473]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_472]], %[[VAL_474]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_475:.*]] = llvm.alloca %[[VAL_466]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[VAL_476:.*]] = llvm.extractvalue %[[VAL_438]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_477:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VAL_478:.*]] = llvm.getelementptr %[[VAL_475]][%[[VAL_477]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_476]], %[[VAL_478]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_479:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_480:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_481:.*]] = llvm.getelementptr %[[VAL_475]][%[[VAL_480]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT: llvm.store %[[VAL_479]], %[[VAL_481]] : i64, !llvm.ptr
// CHECK-NEXT: %[[VAL_482:.*]] = llvm.extractvalue %[[VAL_192]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_483:.*]] = llvm.extractvalue %[[VAL_438]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_484:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_485:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_486:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_487:.*]] = llvm.call @wrap_expand(%arg0, %[[VAL_482]], %[[VAL_467]], %[[VAL_483]], %[[VAL_468]], %[[VAL_484]], %[[VAL_475]], %[[VAL_485]], %[[VAL_486]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_488:.*]] = llvm.extractvalue %[[VAL_214]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_488]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_489:.*]] = llvm.extractvalue %[[VAL_438]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_490:.*]] = llvm.extractvalue %[[VAL_438]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_491:.*]] = llvm.extractvalue %[[VAL_35]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_492:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_493:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_494:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VAL_495:.*]] = llvm.extractvalue %[[VAL_438]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_496:.*]] = llvm.extractvalue %[[VAL_35]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_497:.*]] = llvm.extractvalue %[[VAL_465]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_498:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VAL_499:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_500:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[VAL_501:.*]] = llvm.call @wrap_hipblasLtMatmul(%arg0, %[[VAL_494]], %[[VAL_495]], %[[VAL_496]], %[[VAL_497]], %[[VAL_489]], %[[VAL_491]], %[[VAL_490]], %[[VAL_492]], %[[VAL_498]], %[[VAL_493]], %[[VAL_499]], %[[VAL_500]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: %[[VAL_502:.*]] = llvm.extractvalue %[[VAL_324]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_502]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: %[[VAL_503:.*]] = llvm.extractvalue %[[VAL_392]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.call @free(%[[VAL_503]]) : (!llvm.ptr) -> ()
// CHECK-NEXT: llvm.return %[[VAL_465]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
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

func.func @main_graph(%a: tensor<?x3xf16> {onnx.name = "a"},
                      %b: tensor<?x4xf32> {onnx.name = "b"})
    -> (tensor<?x2xf32> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %w1 = "onnx.Constant"() {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : () -> tensor<3x1xf16>
  %mm1 = "onnx.MatMul"(%a, %w1) : (tensor<?x3xf16>, tensor<3x1xf16>)
      -> tensor<?x1xf16>
  %cast = "onnx.Cast"(%mm1) {to = f32} : (tensor<?x1xf16>) -> tensor<?x1xf32>
  %shape = "onnx.Shape"(%b) : (tensor<?x4xf32>) -> tensor<2xi64>
  %expand = "onnx.Expand"(%cast, %shape)
      : (tensor<?x1xf32>, tensor<2xi64>) -> tensor<?x4xf32>
  %w2 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0],
                                          [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>}
      : () -> tensor<4x2xf32>
  %y = "onnx.MatMul"(%expand, %w2) : (tensor<?x4xf32>, tensor<4x2xf32>)
      -> tensor<?x2xf32>
  "onnx.Return"(%y) : (tensor<?x2xf32>) -> ()
}
