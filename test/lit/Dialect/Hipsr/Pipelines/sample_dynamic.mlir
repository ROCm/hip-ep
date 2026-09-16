// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The same graph as sample_static.mlir, but with a dynamic leading extent. It
// enters the shape graph as a memref.dim. The checks cover the full LLVM IR
// after --hipsr-pipeline.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// CHECK-LABEL: module attributes {
// CHECK-SAME: hip.constants_file = "constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64: 0, 64>
// CHECK-SAME: hipdnn.constant_sizes = array<i64: 32, 6>
// CHECK-SAME: hipdnn.num_op_state_slots = 2 : i32
// CHECK-NEXT:  llvm.func @wrap_expand(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @wrap_cast(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_hipblasLtMatmul(!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:  llvm.func @free(!llvm.ptr)
// CHECK-NEXT:  llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:  llvm.func @hipdnn_ep_op_state_construct_matmul(!llvm.ptr, i32) -> i8
// CHECK-NEXT:  llvm.func @hipdnn_ep_op_states_alloc(!llvm.ptr, i64) -> i8
// CHECK-LABEL: llvm.func @main_graph(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr<1>, %[[ARG2:[^,]*]]: !llvm.ptr<1>, %[[ARG3:[^,]*]]: i64, %[[ARG4:[^,]*]]: i64, %[[ARG5:[^,]*]]: i64, %[[ARG6:[^,]*]]: i64, %[[ARG7:[^,]*]]: i64, %[[ARG8:[^,]*]]: !llvm.ptr<1>, %[[ARG9:[^,]*]]: !llvm.ptr<1>, %[[ARG10:[^,]*]]: i64, %[[ARG11:[^,]*]]: i64, %[[ARG12:[^,]*]]: i64, %[[ARG13:[^,]*]]: i64, %[[ARG14:[^,]*]]: i64) -> (!llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {onnx.name = "y"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1:.*]] = llvm.insertvalue %[[ARG8]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[ARG9]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[ARG10]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[ARG11]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[ARG13]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.insertvalue %[[ARG12]], %[[V5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[ARG14]], %[[V6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V9:.*]] = llvm.insertvalue %[[ARG1]], %[[V8]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V10:.*]] = llvm.insertvalue %[[ARG2]], %[[V9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V11:.*]] = llvm.insertvalue %[[ARG3]], %[[V10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V12:.*]] = llvm.insertvalue %[[ARG4]], %[[V11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V13:.*]] = llvm.insertvalue %[[ARG6]], %[[V12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V14:.*]] = llvm.insertvalue %[[ARG5]], %[[V13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.*]] = llvm.insertvalue %[[ARG7]], %[[V14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK-NEXT:    %[[V17:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V18:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V19:.*]] = llvm.mlir.constant(255 : index) : i64
// CHECK-NEXT:    %[[V20:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT:    %[[V21:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V22:.*]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V21]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V23:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V24:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V25:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V26:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V27:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V28:.*]] = llvm.insertvalue %[[V22]], %[[V27]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V29:.*]] = llvm.insertvalue %[[V22]], %[[V28]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V30:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V31:.*]] = llvm.insertvalue %[[V30]], %[[V29]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V32:.*]] = llvm.insertvalue %[[V23]], %[[V31]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V33:.*]] = llvm.insertvalue %[[V24]], %[[V32]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V34:.*]] = llvm.insertvalue %[[V26]], %[[V33]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V35:.*]] = llvm.insertvalue %[[V25]], %[[V34]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V36:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V37:.*]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V36]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V38:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V39:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V40:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V41:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V42:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V43:.*]] = llvm.insertvalue %[[V37]], %[[V42]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V44:.*]] = llvm.insertvalue %[[V37]], %[[V43]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V45:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V46:.*]] = llvm.insertvalue %[[V45]], %[[V44]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V47:.*]] = llvm.insertvalue %[[V38]], %[[V46]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V48:.*]] = llvm.insertvalue %[[V39]], %[[V47]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V49:.*]] = llvm.insertvalue %[[V41]], %[[V48]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V50:.*]] = llvm.insertvalue %[[V40]], %[[V49]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V51:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V52:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V53:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V54:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V55:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V56:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V57:.*]] = llvm.getelementptr %[[V56]][%[[V54]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V58:.*]] = llvm.ptrtoint %[[V57]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V59:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V60:.*]] = llvm.add %[[V58]], %[[V59]] : i64
// CHECK-NEXT:    %[[V61:.*]] = llvm.call @malloc(%[[V60]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V62:.*]] = llvm.ptrtoint %[[V61]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V63:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V64:.*]] = llvm.sub %[[V59]], %[[V63]] : i64
// CHECK-NEXT:    %[[V65:.*]] = llvm.add %[[V62]], %[[V64]] : i64
// CHECK-NEXT:    %[[V66:.*]] = llvm.urem %[[V65]], %[[V59]] : i64
// CHECK-NEXT:    %[[V67:.*]] = llvm.sub %[[V65]], %[[V66]] : i64
// CHECK-NEXT:    %[[V68:.*]] = llvm.inttoptr %[[V67]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V69:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V70:.*]] = llvm.insertvalue %[[V61]], %[[V69]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V71:.*]] = llvm.insertvalue %[[V68]], %[[V70]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V72:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V73:.*]] = llvm.insertvalue %[[V72]], %[[V71]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V74:.*]] = llvm.insertvalue %[[V54]], %[[V73]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V75:.*]] = llvm.insertvalue %[[V55]], %[[V74]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V76:.*]] = llvm.extractvalue %[[V75]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V77:.*]] = llvm.getelementptr inbounds|nuw %[[V76]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V53]], %[[V77]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V78:.*]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V79:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V80:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V81:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V82:.*]] = llvm.getelementptr %[[V81]][%[[V79]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V83:.*]] = llvm.ptrtoint %[[V82]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V84:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V85:.*]] = llvm.add %[[V83]], %[[V84]] : i64
// CHECK-NEXT:    %[[V86:.*]] = llvm.call @malloc(%[[V85]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V87:.*]] = llvm.ptrtoint %[[V86]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V88:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V89:.*]] = llvm.sub %[[V84]], %[[V88]] : i64
// CHECK-NEXT:    %[[V90:.*]] = llvm.add %[[V87]], %[[V89]] : i64
// CHECK-NEXT:    %[[V91:.*]] = llvm.urem %[[V90]], %[[V84]] : i64
// CHECK-NEXT:    %[[V92:.*]] = llvm.sub %[[V90]], %[[V91]] : i64
// CHECK-NEXT:    %[[V93:.*]] = llvm.inttoptr %[[V92]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V94:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V95:.*]] = llvm.insertvalue %[[V86]], %[[V94]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V96:.*]] = llvm.insertvalue %[[V93]], %[[V95]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V97:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V98:.*]] = llvm.insertvalue %[[V97]], %[[V96]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V99:.*]] = llvm.insertvalue %[[V79]], %[[V98]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V100:.*]] = llvm.insertvalue %[[V80]], %[[V99]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V101:.*]] = llvm.extractvalue %[[V100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V102:.*]] = llvm.getelementptr inbounds|nuw %[[V101]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V78]], %[[V102]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V103:.*]] = llvm.extractvalue %[[V100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V104:.*]] = llvm.getelementptr inbounds|nuw %[[V103]][%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V51]], %[[V104]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V105:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V106:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V107:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V108:.*]] = llvm.getelementptr %[[V107]][%[[V105]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V109:.*]] = llvm.ptrtoint %[[V108]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V110:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V111:.*]] = llvm.add %[[V109]], %[[V110]] : i64
// CHECK-NEXT:    %[[V112:.*]] = llvm.call @malloc(%[[V111]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V113:.*]] = llvm.ptrtoint %[[V112]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V114:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V115:.*]] = llvm.sub %[[V110]], %[[V114]] : i64
// CHECK-NEXT:    %[[V116:.*]] = llvm.add %[[V113]], %[[V115]] : i64
// CHECK-NEXT:    %[[V117:.*]] = llvm.urem %[[V116]], %[[V110]] : i64
// CHECK-NEXT:    %[[V118:.*]] = llvm.sub %[[V116]], %[[V117]] : i64
// CHECK-NEXT:    %[[V119:.*]] = llvm.inttoptr %[[V118]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V120:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V121:.*]] = llvm.insertvalue %[[V112]], %[[V120]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V122:.*]] = llvm.insertvalue %[[V119]], %[[V121]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V123:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V124:.*]] = llvm.insertvalue %[[V123]], %[[V122]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V125:.*]] = llvm.insertvalue %[[V105]], %[[V124]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V126:.*]] = llvm.insertvalue %[[V106]], %[[V125]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V127:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V128:.*]] = llvm.extractvalue %[[V100]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V129:.*]] = llvm.mul %[[V127]], %[[V128]] : i64
// CHECK-NEXT:    %[[V130:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V131:.*]] = llvm.getelementptr %[[V130]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V132:.*]] = llvm.ptrtoint %[[V131]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V133:.*]] = llvm.mul %[[V129]], %[[V132]] : i64
// CHECK-NEXT:    %[[V134:.*]] = llvm.extractvalue %[[V100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V135:.*]] = llvm.extractvalue %[[V100]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V136:.*]] = llvm.getelementptr %[[V134]][%[[V135]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V137:.*]] = llvm.extractvalue %[[V126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V138:.*]] = llvm.extractvalue %[[V126]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V139:.*]] = llvm.getelementptr %[[V137]][%[[V138]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V139]], %[[V136]], %[[V133]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V140:.*]] = llvm.extractvalue %[[V100]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V140]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V141:.*]] = llvm.extractvalue %[[V126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V142:.*]] = llvm.getelementptr inbounds|nuw %[[V141]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V143:.*]] = llvm.load %[[V142]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V144:.*]] = llvm.mul %[[V143]], %[[V53]] : i64
// CHECK-NEXT:    %[[V145:.*]] = llvm.add %[[V144]], %[[V19]] : i64
// CHECK-NEXT:    %[[V146:.*]] = llvm.udiv %[[V145]], %[[V20]] : i64
// CHECK-NEXT:    %[[V147:.*]] = llvm.mul %[[V146]], %[[V20]] : i64
// CHECK-NEXT:    %[[V148:.*]] = llvm.mul %[[V143]], %[[V18]] : i64
// CHECK-NEXT:    %[[V149:.*]] = llvm.add %[[V148]], %[[V19]] : i64
// CHECK-NEXT:    %[[V150:.*]] = llvm.udiv %[[V149]], %[[V20]] : i64
// CHECK-NEXT:    %[[V151:.*]] = llvm.mul %[[V150]], %[[V20]] : i64
// CHECK-NEXT:    %[[V152:.*]] = llvm.add %[[V147]], %[[V151]] : i64
// CHECK-NEXT:    %[[V153:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V154:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V153]], %[[V152]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V155:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V156:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V157:.*]] = llvm.insertvalue %[[V154]], %[[V156]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V158:.*]] = llvm.insertvalue %[[V154]], %[[V157]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V159:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V160:.*]] = llvm.insertvalue %[[V159]], %[[V158]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V161:.*]] = llvm.insertvalue %[[V152]], %[[V160]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V162:.*]] = llvm.insertvalue %[[V155]], %[[V161]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V163:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V164:.*]] = llvm.extractvalue %[[V162]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V165:.*]] = llvm.insertvalue %[[V164]], %[[V163]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V166:.*]] = llvm.extractvalue %[[V162]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V167:.*]] = llvm.getelementptr %[[V166]][%[[V52]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V168:.*]] = llvm.insertvalue %[[V167]], %[[V165]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V169:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V170:.*]] = llvm.insertvalue %[[V169]], %[[V168]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V171:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V172:.*]] = llvm.insertvalue %[[V171]], %[[V170]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V173:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V174:.*]] = llvm.insertvalue %[[V173]], %[[V172]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V175:.*]] = llvm.insertvalue %[[V143]], %[[V174]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V176:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V177:.*]] = llvm.insertvalue %[[V176]], %[[V175]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V178:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V179:.*]] = llvm.extractvalue %[[V162]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V180:.*]] = llvm.insertvalue %[[V179]], %[[V178]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V181:.*]] = llvm.extractvalue %[[V162]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V182:.*]] = llvm.getelementptr %[[V181]][%[[V147]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V183:.*]] = llvm.insertvalue %[[V182]], %[[V180]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V184:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V185:.*]] = llvm.insertvalue %[[V184]], %[[V183]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V186:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V187:.*]] = llvm.insertvalue %[[V186]], %[[V185]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V188:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V189:.*]] = llvm.insertvalue %[[V188]], %[[V187]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V190:.*]] = llvm.insertvalue %[[V143]], %[[V189]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V191:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V192:.*]] = llvm.insertvalue %[[V191]], %[[V190]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V193:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V194:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V195:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V196:.*]] = llvm.getelementptr %[[V195]][%[[V193]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V197:.*]] = llvm.ptrtoint %[[V196]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V198:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V199:.*]] = llvm.add %[[V197]], %[[V198]] : i64
// CHECK-NEXT:    %[[V200:.*]] = llvm.call @malloc(%[[V199]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V201:.*]] = llvm.ptrtoint %[[V200]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V202:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V203:.*]] = llvm.sub %[[V198]], %[[V202]] : i64
// CHECK-NEXT:    %[[V204:.*]] = llvm.add %[[V201]], %[[V203]] : i64
// CHECK-NEXT:    %[[V205:.*]] = llvm.urem %[[V204]], %[[V198]] : i64
// CHECK-NEXT:    %[[V206:.*]] = llvm.sub %[[V204]], %[[V205]] : i64
// CHECK-NEXT:    %[[V207:.*]] = llvm.inttoptr %[[V206]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V208:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V209:.*]] = llvm.insertvalue %[[V200]], %[[V208]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V210:.*]] = llvm.insertvalue %[[V207]], %[[V209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V211:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V212:.*]] = llvm.insertvalue %[[V211]], %[[V210]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V213:.*]] = llvm.insertvalue %[[V193]], %[[V212]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V214:.*]] = llvm.insertvalue %[[V194]], %[[V213]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V215:.*]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V216:.*]] = llvm.extractvalue %[[V15]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V217:.*]] = llvm.extractvalue %[[V50]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V218:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V219:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V220:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V221:.*]] = llvm.extractvalue %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V222:.*]] = llvm.extractvalue %[[V50]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V223:.*]] = llvm.extractvalue %[[V177]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V224:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V225:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V226:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V227:.*]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V220]], %[[V221]], %[[V222]], %[[V223]], %[[V215]], %[[V217]], %[[V216]], %[[V218]], %[[V224]], %[[V219]], %[[V225]], %[[V226]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V228:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V229:.*]] = llvm.extractvalue %[[V192]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V230:.*]] = llvm.mul %[[V228]], %[[V229]] : i64
// CHECK-NEXT:    %[[V231:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V232:.*]] = llvm.mul %[[V230]], %[[V231]] : i64
// CHECK-NEXT:    %[[V233:.*]] = llvm.extractvalue %[[V177]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V234:.*]] = llvm.extractvalue %[[V192]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V235:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V236:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V237:.*]] = llvm.call @wrap_cast(%[[ARG0]], %[[V233]], %[[V234]], %[[V232]], %[[V235]], %[[V236]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V238:.*]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V239:.*]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V240:.*]] = llvm.getelementptr inbounds|nuw %[[V239]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V238]], %[[V240]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V241:.*]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V242:.*]] = llvm.getelementptr inbounds|nuw %[[V241]][%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V17]], %[[V242]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V243:.*]] = llvm.extractvalue %[[V126]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V243]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V244:.*]] = llvm.extractvalue %[[V75]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V244]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V245:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V246:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V247:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V248:.*]] = llvm.getelementptr %[[V247]][%[[V245]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V249:.*]] = llvm.ptrtoint %[[V248]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V250:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V251:.*]] = llvm.add %[[V249]], %[[V250]] : i64
// CHECK-NEXT:    %[[V252:.*]] = llvm.call @malloc(%[[V251]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V253:.*]] = llvm.ptrtoint %[[V252]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V254:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V255:.*]] = llvm.sub %[[V250]], %[[V254]] : i64
// CHECK-NEXT:    %[[V256:.*]] = llvm.add %[[V253]], %[[V255]] : i64
// CHECK-NEXT:    %[[V257:.*]] = llvm.urem %[[V256]], %[[V250]] : i64
// CHECK-NEXT:    %[[V258:.*]] = llvm.sub %[[V256]], %[[V257]] : i64
// CHECK-NEXT:    %[[V259:.*]] = llvm.inttoptr %[[V258]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V260:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V261:.*]] = llvm.insertvalue %[[V252]], %[[V260]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V262:.*]] = llvm.insertvalue %[[V259]], %[[V261]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V263:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V264:.*]] = llvm.insertvalue %[[V263]], %[[V262]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V265:.*]] = llvm.insertvalue %[[V245]], %[[V264]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V266:.*]] = llvm.insertvalue %[[V246]], %[[V265]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V267:.*]] = llvm.extractvalue %[[V266]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V268:.*]] = llvm.getelementptr inbounds|nuw %[[V267]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V143]], %[[V268]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V269:.*]] = llvm.extractvalue %[[V266]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V270:.*]] = llvm.getelementptr inbounds|nuw %[[V269]][%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V51]], %[[V270]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V271:.*]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V272:.*]] = llvm.getelementptr inbounds|nuw %[[V271]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V273:.*]] = llvm.load %[[V272]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V274:.*]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V275:.*]] = llvm.getelementptr inbounds|nuw %[[V274]][%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V276:.*]] = llvm.load %[[V275]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V277:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V278:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V279:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V280:.*]] = llvm.getelementptr %[[V279]][%[[V277]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V281:.*]] = llvm.ptrtoint %[[V280]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V282:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V283:.*]] = llvm.add %[[V281]], %[[V282]] : i64
// CHECK-NEXT:    %[[V284:.*]] = llvm.call @malloc(%[[V283]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V285:.*]] = llvm.ptrtoint %[[V284]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V286:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V287:.*]] = llvm.sub %[[V282]], %[[V286]] : i64
// CHECK-NEXT:    %[[V288:.*]] = llvm.add %[[V285]], %[[V287]] : i64
// CHECK-NEXT:    %[[V289:.*]] = llvm.urem %[[V288]], %[[V282]] : i64
// CHECK-NEXT:    %[[V290:.*]] = llvm.sub %[[V288]], %[[V289]] : i64
// CHECK-NEXT:    %[[V291:.*]] = llvm.inttoptr %[[V290]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V292:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V293:.*]] = llvm.insertvalue %[[V284]], %[[V292]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V294:.*]] = llvm.insertvalue %[[V291]], %[[V293]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V295:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V296:.*]] = llvm.insertvalue %[[V295]], %[[V294]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V297:.*]] = llvm.insertvalue %[[V277]], %[[V296]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V298:.*]] = llvm.insertvalue %[[V278]], %[[V297]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V299:.*]] = llvm.extractvalue %[[V298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V300:.*]] = llvm.getelementptr inbounds|nuw %[[V299]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V273]], %[[V300]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V301:.*]] = llvm.extractvalue %[[V298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V302:.*]] = llvm.getelementptr inbounds|nuw %[[V301]][%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V276]], %[[V302]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V303:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V304:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V305:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V306:.*]] = llvm.getelementptr %[[V305]][%[[V303]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V307:.*]] = llvm.ptrtoint %[[V306]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V308:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V309:.*]] = llvm.add %[[V307]], %[[V308]] : i64
// CHECK-NEXT:    %[[V310:.*]] = llvm.call @malloc(%[[V309]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V311:.*]] = llvm.ptrtoint %[[V310]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V312:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V313:.*]] = llvm.sub %[[V308]], %[[V312]] : i64
// CHECK-NEXT:    %[[V314:.*]] = llvm.add %[[V311]], %[[V313]] : i64
// CHECK-NEXT:    %[[V315:.*]] = llvm.urem %[[V314]], %[[V308]] : i64
// CHECK-NEXT:    %[[V316:.*]] = llvm.sub %[[V314]], %[[V315]] : i64
// CHECK-NEXT:    %[[V317:.*]] = llvm.inttoptr %[[V316]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V318:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V319:.*]] = llvm.insertvalue %[[V310]], %[[V318]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V320:.*]] = llvm.insertvalue %[[V317]], %[[V319]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V321:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V322:.*]] = llvm.insertvalue %[[V321]], %[[V320]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V323:.*]] = llvm.insertvalue %[[V303]], %[[V322]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V324:.*]] = llvm.insertvalue %[[V304]], %[[V323]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb1(%[[V52]] : i64)
// CHECK-NEXT:    ^bb1(%[[V325:.*]]: i64):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:    %[[V326:.*]] = llvm.icmp "slt" %[[V325]], %[[V53]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V326]], ^bb2, ^bb7
// CHECK-NEXT:    ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %[[V327:.*]] = llvm.icmp "ult" %[[V325]], %[[V52]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V327]], ^bb3, ^bb4
// CHECK-NEXT:    ^bb3:  // pred: ^bb2
// CHECK-NEXT:    llvm.br ^bb5(%[[V51]] : i64)
// CHECK-NEXT:    ^bb4:  // pred: ^bb2
// CHECK-NEXT:    %[[V328:.*]] = llvm.extractvalue %[[V266]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V329:.*]] = llvm.getelementptr inbounds|nuw %[[V328]][%[[V325]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V330:.*]] = llvm.load %[[V329]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V331:.*]] = llvm.extractvalue %[[V298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V332:.*]] = llvm.getelementptr inbounds|nuw %[[V331]][%[[V325]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V333:.*]] = llvm.load %[[V332]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V334:.*]] = llvm.icmp "eq" %[[V333]], %[[V51]] : i64
// CHECK-NEXT:    %[[V335:.*]] = llvm.select %[[V334]], %[[V330]], %[[V333]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb5(%[[V335]] : i64)
// CHECK-NEXT:    ^bb5(%[[V336:.*]]: i64):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:    ^bb6:  // pred: ^bb5
// CHECK-NEXT:    %[[V337:.*]] = llvm.extractvalue %[[V324]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V338:.*]] = llvm.getelementptr inbounds|nuw %[[V337]][%[[V325]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V336]], %[[V338]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V339:.*]] = llvm.add %[[V325]], %[[V51]] : i64
// CHECK-NEXT:    llvm.br ^bb1(%[[V339]] : i64)
// CHECK-NEXT:    ^bb7:  // pred: ^bb1
// CHECK-NEXT:    %[[V340:.*]] = llvm.extractvalue %[[V298]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V340]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V341:.*]] = llvm.extractvalue %[[V266]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V341]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V342:.*]] = llvm.extractvalue %[[V324]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V343:.*]] = llvm.getelementptr inbounds|nuw %[[V342]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V344:.*]] = llvm.load %[[V343]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V345:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V346:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V347:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V348:.*]] = llvm.getelementptr %[[V347]][%[[V345]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V349:.*]] = llvm.ptrtoint %[[V348]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V350:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V351:.*]] = llvm.add %[[V349]], %[[V350]] : i64
// CHECK-NEXT:    %[[V352:.*]] = llvm.call @malloc(%[[V351]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V353:.*]] = llvm.ptrtoint %[[V352]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V354:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V355:.*]] = llvm.sub %[[V350]], %[[V354]] : i64
// CHECK-NEXT:    %[[V356:.*]] = llvm.add %[[V353]], %[[V355]] : i64
// CHECK-NEXT:    %[[V357:.*]] = llvm.urem %[[V356]], %[[V350]] : i64
// CHECK-NEXT:    %[[V358:.*]] = llvm.sub %[[V356]], %[[V357]] : i64
// CHECK-NEXT:    %[[V359:.*]] = llvm.inttoptr %[[V358]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V360:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V361:.*]] = llvm.insertvalue %[[V352]], %[[V360]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V362:.*]] = llvm.insertvalue %[[V359]], %[[V361]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V363:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V364:.*]] = llvm.insertvalue %[[V363]], %[[V362]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V365:.*]] = llvm.insertvalue %[[V345]], %[[V364]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V366:.*]] = llvm.insertvalue %[[V346]], %[[V365]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V367:.*]] = llvm.extractvalue %[[V366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V368:.*]] = llvm.getelementptr inbounds|nuw %[[V367]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V344]], %[[V368]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V369:.*]] = llvm.extractvalue %[[V366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V370:.*]] = llvm.getelementptr inbounds|nuw %[[V369]][%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V53]], %[[V370]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V371:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V372:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V373:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V374:.*]] = llvm.getelementptr %[[V373]][%[[V371]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V375:.*]] = llvm.ptrtoint %[[V374]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V376:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V377:.*]] = llvm.add %[[V375]], %[[V376]] : i64
// CHECK-NEXT:    %[[V378:.*]] = llvm.call @malloc(%[[V377]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V379:.*]] = llvm.ptrtoint %[[V378]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V380:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V381:.*]] = llvm.sub %[[V376]], %[[V380]] : i64
// CHECK-NEXT:    %[[V382:.*]] = llvm.add %[[V379]], %[[V381]] : i64
// CHECK-NEXT:    %[[V383:.*]] = llvm.urem %[[V382]], %[[V376]] : i64
// CHECK-NEXT:    %[[V384:.*]] = llvm.sub %[[V382]], %[[V383]] : i64
// CHECK-NEXT:    %[[V385:.*]] = llvm.inttoptr %[[V384]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V386:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V387:.*]] = llvm.insertvalue %[[V378]], %[[V386]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V388:.*]] = llvm.insertvalue %[[V385]], %[[V387]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V389:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V390:.*]] = llvm.insertvalue %[[V389]], %[[V388]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V391:.*]] = llvm.insertvalue %[[V371]], %[[V390]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V392:.*]] = llvm.insertvalue %[[V372]], %[[V391]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V393:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V394:.*]] = llvm.extractvalue %[[V366]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V395:.*]] = llvm.mul %[[V393]], %[[V394]] : i64
// CHECK-NEXT:    %[[V396:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V397:.*]] = llvm.getelementptr %[[V396]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V398:.*]] = llvm.ptrtoint %[[V397]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V399:.*]] = llvm.mul %[[V395]], %[[V398]] : i64
// CHECK-NEXT:    %[[V400:.*]] = llvm.extractvalue %[[V366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V401:.*]] = llvm.extractvalue %[[V366]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V402:.*]] = llvm.getelementptr %[[V400]][%[[V401]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V403:.*]] = llvm.extractvalue %[[V392]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V404:.*]] = llvm.extractvalue %[[V392]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V405:.*]] = llvm.getelementptr %[[V403]][%[[V404]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V405]], %[[V402]], %[[V399]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V406:.*]] = llvm.extractvalue %[[V366]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V406]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V407:.*]] = llvm.extractvalue %[[V324]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V408:.*]] = llvm.getelementptr inbounds|nuw %[[V407]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V409:.*]] = llvm.load %[[V408]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V410:.*]] = llvm.mul %[[V409]], %[[V16]] : i64
// CHECK-NEXT:    %[[V411:.*]] = llvm.add %[[V410]], %[[V19]] : i64
// CHECK-NEXT:    %[[V412:.*]] = llvm.udiv %[[V411]], %[[V20]] : i64
// CHECK-NEXT:    %[[V413:.*]] = llvm.mul %[[V412]], %[[V20]] : i64
// CHECK-NEXT:    %[[V414:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V415:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V414]], %[[V413]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V416:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V417:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V418:.*]] = llvm.insertvalue %[[V415]], %[[V417]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V419:.*]] = llvm.insertvalue %[[V415]], %[[V418]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V420:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V421:.*]] = llvm.insertvalue %[[V420]], %[[V419]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V422:.*]] = llvm.insertvalue %[[V413]], %[[V421]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V423:.*]] = llvm.insertvalue %[[V416]], %[[V422]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V424:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V425:.*]] = llvm.extractvalue %[[V423]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V426:.*]] = llvm.insertvalue %[[V425]], %[[V424]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V427:.*]] = llvm.extractvalue %[[V423]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V428:.*]] = llvm.getelementptr %[[V427]][%[[V52]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V429:.*]] = llvm.insertvalue %[[V428]], %[[V426]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V430:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V431:.*]] = llvm.insertvalue %[[V430]], %[[V429]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V432:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V433:.*]] = llvm.insertvalue %[[V432]], %[[V431]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V434:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V435:.*]] = llvm.insertvalue %[[V434]], %[[V433]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V436:.*]] = llvm.insertvalue %[[V409]], %[[V435]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V437:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V438:.*]] = llvm.insertvalue %[[V437]], %[[V436]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V439:.*]] = llvm.extractvalue %[[V392]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V440:.*]] = llvm.getelementptr inbounds|nuw %[[V439]][%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V441:.*]] = llvm.load %[[V440]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V442:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V443:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V444:.*]] = llvm.mul %[[V442]], %[[V441]] : i64
// CHECK-NEXT:    %[[V445:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V446:.*]] = llvm.getelementptr %[[V445]][%[[V444]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:    %[[V447:.*]] = llvm.ptrtoint %[[V446]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V448:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V449:.*]] = llvm.alloca %[[V448]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V450:.*]] = llvm.getelementptr %[[V449]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V441]], %[[V450]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V451:.*]] = llvm.getelementptr %[[V449]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V442]], %[[V451]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V452:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V453:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V454:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V455:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V452]], %[[V449]], %[[V453]], %[[V454]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V456:.*]] = llvm.addrspacecast %[[V455]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V457:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V458:.*]] = llvm.insertvalue %[[V456]], %[[V457]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V459:.*]] = llvm.insertvalue %[[V456]], %[[V458]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V460:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V461:.*]] = llvm.insertvalue %[[V460]], %[[V459]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V462:.*]] = llvm.insertvalue %[[V441]], %[[V461]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V463:.*]] = llvm.insertvalue %[[V442]], %[[V462]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V464:.*]] = llvm.insertvalue %[[V442]], %[[V463]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V465:.*]] = llvm.insertvalue %[[V443]], %[[V464]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V466:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V467:.*]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V468:.*]] = llvm.alloca %[[V466]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V469:.*]] = llvm.extractvalue %[[V192]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V470:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V471:.*]] = llvm.getelementptr %[[V468]][%[[V470]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V469]], %[[V471]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V472:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V473:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V474:.*]] = llvm.getelementptr %[[V468]][%[[V473]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V472]], %[[V474]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V475:.*]] = llvm.alloca %[[V466]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V476:.*]] = llvm.extractvalue %[[V438]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V477:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V478:.*]] = llvm.getelementptr %[[V475]][%[[V477]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V476]], %[[V478]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V479:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V480:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V481:.*]] = llvm.getelementptr %[[V475]][%[[V480]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V479]], %[[V481]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V482:.*]] = llvm.extractvalue %[[V192]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V483:.*]] = llvm.extractvalue %[[V438]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V484:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V485:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V486:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V487:.*]] = llvm.call @wrap_expand(%[[ARG0]], %[[V482]], %[[V467]], %[[V483]], %[[V468]], %[[V484]], %[[V475]], %[[V485]], %[[V486]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V488:.*]] = llvm.extractvalue %[[V214]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V488]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V489:.*]] = llvm.extractvalue %[[V438]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V490:.*]] = llvm.extractvalue %[[V438]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V491:.*]] = llvm.extractvalue %[[V35]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V492:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V493:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V494:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V495:.*]] = llvm.extractvalue %[[V438]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V496:.*]] = llvm.extractvalue %[[V35]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V497:.*]] = llvm.extractvalue %[[V465]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V498:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V499:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V500:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V501:.*]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V494]], %[[V495]], %[[V496]], %[[V497]], %[[V489]], %[[V491]], %[[V490]], %[[V492]], %[[V498]], %[[V493]], %[[V499]], %[[V500]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V502:.*]] = llvm.extractvalue %[[V324]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V502]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V503:.*]] = llvm.extractvalue %[[V392]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V503]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V465]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  }
// CHECK-LABEL: llvm.func @hipdnn_ep_op_states_init_fn(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr) -> i32 {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V2:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:    %[[V3:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V4:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[ARG0]], %[[V3]]) : (!llvm.ptr, i64) -> i8
// CHECK-NEXT:    %[[V5:.*]] = llvm.icmp "eq" %[[V4]], %[[V2]] : i8
// CHECK-NEXT:    llvm.cond_br %[[V5]], ^bb2, ^bb1
// CHECK-NEXT:    ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %[[V6:.*]] = llvm.call @hipdnn_ep_op_state_construct_matmul(%[[ARG0]], %[[V1]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT:    %[[V7:.*]] = llvm.call @hipdnn_ep_op_state_construct_matmul(%[[ARG0]], %[[V0]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT:    llvm.return %[[V1]] : i32
// CHECK-NEXT:    ^bb2:  // pred: ^bb0
// CHECK-NEXT:    llvm.return %[[V0]] : i32
// CHECK-NEXT:  }

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
