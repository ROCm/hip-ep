// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The hipsr pipeline on a graph where every extent is static. Each extent is a
// constant, so no allocation reads its size back out of a shape buffer.
//
// The checks cover the full LLVM IR after --hipsr-pipeline.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// Two constants, then the first pool and the matmul/cast pair, then the second
// pool, the output buffer, and the expand/matmul pair.

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
// CHECK-NEXT:    %[[V1:.*]] = llvm.insertvalue %[[ARG1]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[ARG2]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[ARG3]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[ARG4]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[ARG6]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.insertvalue %[[ARG5]], %[[V5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[ARG7]], %[[V6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.mlir.constant(512 : index) : i64
// CHECK-NEXT:    %[[V9:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT:    %[[V10:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V11:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V12:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V13:.*]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V12]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V14:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V15:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V16:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V17:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V18:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V19:.*]] = llvm.insertvalue %[[V13]], %[[V18]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V20:.*]] = llvm.insertvalue %[[V13]], %[[V19]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V22:.*]] = llvm.insertvalue %[[V21]], %[[V20]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.*]] = llvm.insertvalue %[[V14]], %[[V22]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V24:.*]] = llvm.insertvalue %[[V15]], %[[V23]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V25:.*]] = llvm.insertvalue %[[V17]], %[[V24]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V26:.*]] = llvm.insertvalue %[[V16]], %[[V25]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V27:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V28:.*]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V27]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V29:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V30:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V31:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V32:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V33:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V34:.*]] = llvm.insertvalue %[[V28]], %[[V33]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V35:.*]] = llvm.insertvalue %[[V28]], %[[V34]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V36:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V37:.*]] = llvm.insertvalue %[[V36]], %[[V35]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V38:.*]] = llvm.insertvalue %[[V29]], %[[V37]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V39:.*]] = llvm.insertvalue %[[V30]], %[[V38]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V40:.*]] = llvm.insertvalue %[[V32]], %[[V39]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V41:.*]] = llvm.insertvalue %[[V31]], %[[V40]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V42:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V43:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V44:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V45:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V46:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V47:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V48:.*]] = llvm.getelementptr %[[V47]][%[[V45]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V49:.*]] = llvm.ptrtoint %[[V48]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V50:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V51:.*]] = llvm.add %[[V49]], %[[V50]] : i64
// CHECK-NEXT:    %[[V52:.*]] = llvm.call @malloc(%[[V51]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V53:.*]] = llvm.ptrtoint %[[V52]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V54:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V55:.*]] = llvm.sub %[[V50]], %[[V54]] : i64
// CHECK-NEXT:    %[[V56:.*]] = llvm.add %[[V53]], %[[V55]] : i64
// CHECK-NEXT:    %[[V57:.*]] = llvm.urem %[[V56]], %[[V50]] : i64
// CHECK-NEXT:    %[[V58:.*]] = llvm.sub %[[V56]], %[[V57]] : i64
// CHECK-NEXT:    %[[V59:.*]] = llvm.inttoptr %[[V58]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V60:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V61:.*]] = llvm.insertvalue %[[V52]], %[[V60]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V62:.*]] = llvm.insertvalue %[[V59]], %[[V61]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V63:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V64:.*]] = llvm.insertvalue %[[V63]], %[[V62]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V65:.*]] = llvm.insertvalue %[[V45]], %[[V64]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V66:.*]] = llvm.insertvalue %[[V46]], %[[V65]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V67:.*]] = llvm.extractvalue %[[V66]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V68:.*]] = llvm.getelementptr inbounds|nuw %[[V67]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V68]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V69:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V70:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V71:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V72:.*]] = llvm.getelementptr %[[V71]][%[[V69]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V73:.*]] = llvm.ptrtoint %[[V72]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V74:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V75:.*]] = llvm.add %[[V73]], %[[V74]] : i64
// CHECK-NEXT:    %[[V76:.*]] = llvm.call @malloc(%[[V75]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V77:.*]] = llvm.ptrtoint %[[V76]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V78:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V79:.*]] = llvm.sub %[[V74]], %[[V78]] : i64
// CHECK-NEXT:    %[[V80:.*]] = llvm.add %[[V77]], %[[V79]] : i64
// CHECK-NEXT:    %[[V81:.*]] = llvm.urem %[[V80]], %[[V74]] : i64
// CHECK-NEXT:    %[[V82:.*]] = llvm.sub %[[V80]], %[[V81]] : i64
// CHECK-NEXT:    %[[V83:.*]] = llvm.inttoptr %[[V82]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V84:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V85:.*]] = llvm.insertvalue %[[V76]], %[[V84]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V86:.*]] = llvm.insertvalue %[[V83]], %[[V85]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V87:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V88:.*]] = llvm.insertvalue %[[V87]], %[[V86]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V89:.*]] = llvm.insertvalue %[[V69]], %[[V88]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V90:.*]] = llvm.insertvalue %[[V70]], %[[V89]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V91:.*]] = llvm.extractvalue %[[V90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V92:.*]] = llvm.getelementptr inbounds|nuw %[[V91]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V92]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V93:.*]] = llvm.extractvalue %[[V90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V94:.*]] = llvm.getelementptr inbounds|nuw %[[V93]][%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V42]], %[[V94]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V95:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V96:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V97:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V98:.*]] = llvm.getelementptr %[[V97]][%[[V95]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V99:.*]] = llvm.ptrtoint %[[V98]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V100:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V101:.*]] = llvm.add %[[V99]], %[[V100]] : i64
// CHECK-NEXT:    %[[V102:.*]] = llvm.call @malloc(%[[V101]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V103:.*]] = llvm.ptrtoint %[[V102]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V104:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V105:.*]] = llvm.sub %[[V100]], %[[V104]] : i64
// CHECK-NEXT:    %[[V106:.*]] = llvm.add %[[V103]], %[[V105]] : i64
// CHECK-NEXT:    %[[V107:.*]] = llvm.urem %[[V106]], %[[V100]] : i64
// CHECK-NEXT:    %[[V108:.*]] = llvm.sub %[[V106]], %[[V107]] : i64
// CHECK-NEXT:    %[[V109:.*]] = llvm.inttoptr %[[V108]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V110:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V111:.*]] = llvm.insertvalue %[[V102]], %[[V110]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V112:.*]] = llvm.insertvalue %[[V109]], %[[V111]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V113:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V114:.*]] = llvm.insertvalue %[[V113]], %[[V112]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V115:.*]] = llvm.insertvalue %[[V95]], %[[V114]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V116:.*]] = llvm.insertvalue %[[V96]], %[[V115]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V117:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V118:.*]] = llvm.extractvalue %[[V90]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V119:.*]] = llvm.mul %[[V117]], %[[V118]] : i64
// CHECK-NEXT:    %[[V120:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V121:.*]] = llvm.getelementptr %[[V120]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V122:.*]] = llvm.ptrtoint %[[V121]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V123:.*]] = llvm.mul %[[V119]], %[[V122]] : i64
// CHECK-NEXT:    %[[V124:.*]] = llvm.extractvalue %[[V90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V125:.*]] = llvm.extractvalue %[[V90]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V126:.*]] = llvm.getelementptr %[[V124]][%[[V125]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V127:.*]] = llvm.extractvalue %[[V116]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V128:.*]] = llvm.extractvalue %[[V116]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V129:.*]] = llvm.getelementptr %[[V127]][%[[V128]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V129]], %[[V126]], %[[V123]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V130:.*]] = llvm.extractvalue %[[V90]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V130]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V131:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V132:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V131]], %[[V8]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V133:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V134:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V135:.*]] = llvm.insertvalue %[[V132]], %[[V134]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V136:.*]] = llvm.insertvalue %[[V132]], %[[V135]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V137:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V138:.*]] = llvm.insertvalue %[[V137]], %[[V136]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V139:.*]] = llvm.insertvalue %[[V8]], %[[V138]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V140:.*]] = llvm.insertvalue %[[V133]], %[[V139]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V141:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V142:.*]] = llvm.extractvalue %[[V140]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V143:.*]] = llvm.insertvalue %[[V142]], %[[V141]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V144:.*]] = llvm.extractvalue %[[V140]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V145:.*]] = llvm.getelementptr %[[V144]][%[[V43]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V146:.*]] = llvm.insertvalue %[[V145]], %[[V143]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V147:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V148:.*]] = llvm.insertvalue %[[V147]], %[[V146]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V149:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V150:.*]] = llvm.insertvalue %[[V149]], %[[V148]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V151:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V152:.*]] = llvm.insertvalue %[[V151]], %[[V150]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V153:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V154:.*]] = llvm.insertvalue %[[V153]], %[[V152]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V155:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V156:.*]] = llvm.insertvalue %[[V155]], %[[V154]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V157:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V158:.*]] = llvm.extractvalue %[[V140]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V159:.*]] = llvm.insertvalue %[[V158]], %[[V157]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V160:.*]] = llvm.extractvalue %[[V140]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V161:.*]] = llvm.getelementptr %[[V160]][%[[V9]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V162:.*]] = llvm.insertvalue %[[V161]], %[[V159]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V163:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V164:.*]] = llvm.insertvalue %[[V163]], %[[V162]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V165:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V166:.*]] = llvm.insertvalue %[[V165]], %[[V164]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V167:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V168:.*]] = llvm.insertvalue %[[V167]], %[[V166]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V169:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V170:.*]] = llvm.insertvalue %[[V169]], %[[V168]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V171:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V172:.*]] = llvm.insertvalue %[[V171]], %[[V170]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V173:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V174:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V175:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V176:.*]] = llvm.getelementptr %[[V175]][%[[V173]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V177:.*]] = llvm.ptrtoint %[[V176]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V178:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V179:.*]] = llvm.add %[[V177]], %[[V178]] : i64
// CHECK-NEXT:    %[[V180:.*]] = llvm.call @malloc(%[[V179]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V181:.*]] = llvm.ptrtoint %[[V180]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V182:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V183:.*]] = llvm.sub %[[V178]], %[[V182]] : i64
// CHECK-NEXT:    %[[V184:.*]] = llvm.add %[[V181]], %[[V183]] : i64
// CHECK-NEXT:    %[[V185:.*]] = llvm.urem %[[V184]], %[[V178]] : i64
// CHECK-NEXT:    %[[V186:.*]] = llvm.sub %[[V184]], %[[V185]] : i64
// CHECK-NEXT:    %[[V187:.*]] = llvm.inttoptr %[[V186]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V188:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V189:.*]] = llvm.insertvalue %[[V180]], %[[V188]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V190:.*]] = llvm.insertvalue %[[V187]], %[[V189]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V191:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V192:.*]] = llvm.insertvalue %[[V191]], %[[V190]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V193:.*]] = llvm.insertvalue %[[V173]], %[[V192]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V194:.*]] = llvm.insertvalue %[[V174]], %[[V193]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V195:.*]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V196:.*]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V197:.*]] = llvm.extractvalue %[[V41]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V198:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V199:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V200:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V201:.*]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V202:.*]] = llvm.extractvalue %[[V41]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V203:.*]] = llvm.extractvalue %[[V156]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V204:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V205:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V206:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V207:.*]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V200]], %[[V201]], %[[V202]], %[[V203]], %[[V195]], %[[V197]], %[[V196]], %[[V198]], %[[V204]], %[[V199]], %[[V205]], %[[V206]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V208:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V209:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V210:.*]] = llvm.mul %[[V208]], %[[V209]] : i64
// CHECK-NEXT:    %[[V211:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V212:.*]] = llvm.mul %[[V210]], %[[V211]] : i64
// CHECK-NEXT:    %[[V213:.*]] = llvm.extractvalue %[[V156]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V214:.*]] = llvm.extractvalue %[[V172]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V215:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V216:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V217:.*]] = llvm.call @wrap_cast(%[[ARG0]], %[[V213]], %[[V214]], %[[V212]], %[[V215]], %[[V216]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V218:.*]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V219:.*]] = llvm.getelementptr inbounds|nuw %[[V218]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V10]], %[[V219]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V220:.*]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V221:.*]] = llvm.getelementptr inbounds|nuw %[[V220]][%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V11]], %[[V221]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V222:.*]] = llvm.extractvalue %[[V116]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V222]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V223:.*]] = llvm.extractvalue %[[V66]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V223]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V224:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V225:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V226:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V227:.*]] = llvm.getelementptr %[[V226]][%[[V224]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V228:.*]] = llvm.ptrtoint %[[V227]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V229:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V230:.*]] = llvm.add %[[V228]], %[[V229]] : i64
// CHECK-NEXT:    %[[V231:.*]] = llvm.call @malloc(%[[V230]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V232:.*]] = llvm.ptrtoint %[[V231]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V233:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V234:.*]] = llvm.sub %[[V229]], %[[V233]] : i64
// CHECK-NEXT:    %[[V235:.*]] = llvm.add %[[V232]], %[[V234]] : i64
// CHECK-NEXT:    %[[V236:.*]] = llvm.urem %[[V235]], %[[V229]] : i64
// CHECK-NEXT:    %[[V237:.*]] = llvm.sub %[[V235]], %[[V236]] : i64
// CHECK-NEXT:    %[[V238:.*]] = llvm.inttoptr %[[V237]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V239:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V240:.*]] = llvm.insertvalue %[[V231]], %[[V239]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V241:.*]] = llvm.insertvalue %[[V238]], %[[V240]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V242:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V243:.*]] = llvm.insertvalue %[[V242]], %[[V241]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V244:.*]] = llvm.insertvalue %[[V224]], %[[V243]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V245:.*]] = llvm.insertvalue %[[V225]], %[[V244]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V246:.*]] = llvm.extractvalue %[[V245]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V247:.*]] = llvm.getelementptr inbounds|nuw %[[V246]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V247]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V248:.*]] = llvm.extractvalue %[[V245]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V249:.*]] = llvm.getelementptr inbounds|nuw %[[V248]][%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V42]], %[[V249]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V250:.*]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V251:.*]] = llvm.getelementptr inbounds|nuw %[[V250]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V252:.*]] = llvm.load %[[V251]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V253:.*]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V254:.*]] = llvm.getelementptr inbounds|nuw %[[V253]][%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V255:.*]] = llvm.load %[[V254]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V256:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V257:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V258:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V259:.*]] = llvm.getelementptr %[[V258]][%[[V256]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V260:.*]] = llvm.ptrtoint %[[V259]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V261:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V262:.*]] = llvm.add %[[V260]], %[[V261]] : i64
// CHECK-NEXT:    %[[V263:.*]] = llvm.call @malloc(%[[V262]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V264:.*]] = llvm.ptrtoint %[[V263]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V265:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V266:.*]] = llvm.sub %[[V261]], %[[V265]] : i64
// CHECK-NEXT:    %[[V267:.*]] = llvm.add %[[V264]], %[[V266]] : i64
// CHECK-NEXT:    %[[V268:.*]] = llvm.urem %[[V267]], %[[V261]] : i64
// CHECK-NEXT:    %[[V269:.*]] = llvm.sub %[[V267]], %[[V268]] : i64
// CHECK-NEXT:    %[[V270:.*]] = llvm.inttoptr %[[V269]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V271:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V272:.*]] = llvm.insertvalue %[[V263]], %[[V271]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V273:.*]] = llvm.insertvalue %[[V270]], %[[V272]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V274:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V275:.*]] = llvm.insertvalue %[[V274]], %[[V273]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V276:.*]] = llvm.insertvalue %[[V256]], %[[V275]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V277:.*]] = llvm.insertvalue %[[V257]], %[[V276]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V278:.*]] = llvm.extractvalue %[[V277]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V279:.*]] = llvm.getelementptr inbounds|nuw %[[V278]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V252]], %[[V279]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V280:.*]] = llvm.extractvalue %[[V277]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V281:.*]] = llvm.getelementptr inbounds|nuw %[[V280]][%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V255]], %[[V281]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V282:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V283:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V284:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V285:.*]] = llvm.getelementptr %[[V284]][%[[V282]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V286:.*]] = llvm.ptrtoint %[[V285]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V287:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V288:.*]] = llvm.add %[[V286]], %[[V287]] : i64
// CHECK-NEXT:    %[[V289:.*]] = llvm.call @malloc(%[[V288]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V290:.*]] = llvm.ptrtoint %[[V289]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V291:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V292:.*]] = llvm.sub %[[V287]], %[[V291]] : i64
// CHECK-NEXT:    %[[V293:.*]] = llvm.add %[[V290]], %[[V292]] : i64
// CHECK-NEXT:    %[[V294:.*]] = llvm.urem %[[V293]], %[[V287]] : i64
// CHECK-NEXT:    %[[V295:.*]] = llvm.sub %[[V293]], %[[V294]] : i64
// CHECK-NEXT:    %[[V296:.*]] = llvm.inttoptr %[[V295]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V297:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V298:.*]] = llvm.insertvalue %[[V289]], %[[V297]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V299:.*]] = llvm.insertvalue %[[V296]], %[[V298]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V300:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V301:.*]] = llvm.insertvalue %[[V300]], %[[V299]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V302:.*]] = llvm.insertvalue %[[V282]], %[[V301]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V303:.*]] = llvm.insertvalue %[[V283]], %[[V302]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb1(%[[V43]] : i64)
// CHECK-NEXT:    ^bb1(%[[V304:.*]]: i64):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:    %[[V305:.*]] = llvm.icmp "slt" %[[V304]], %[[V44]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V305]], ^bb2, ^bb7
// CHECK-NEXT:    ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %[[V306:.*]] = llvm.icmp "ult" %[[V304]], %[[V43]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V306]], ^bb3, ^bb4
// CHECK-NEXT:    ^bb3:  // pred: ^bb2
// CHECK-NEXT:    llvm.br ^bb5(%[[V42]] : i64)
// CHECK-NEXT:    ^bb4:  // pred: ^bb2
// CHECK-NEXT:    %[[V307:.*]] = llvm.extractvalue %[[V245]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V308:.*]] = llvm.getelementptr inbounds|nuw %[[V307]][%[[V304]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V309:.*]] = llvm.load %[[V308]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V310:.*]] = llvm.extractvalue %[[V277]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V311:.*]] = llvm.getelementptr inbounds|nuw %[[V310]][%[[V304]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V312:.*]] = llvm.load %[[V311]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V313:.*]] = llvm.icmp "eq" %[[V312]], %[[V42]] : i64
// CHECK-NEXT:    %[[V314:.*]] = llvm.select %[[V313]], %[[V309]], %[[V312]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb5(%[[V314]] : i64)
// CHECK-NEXT:    ^bb5(%[[V315:.*]]: i64):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:    ^bb6:  // pred: ^bb5
// CHECK-NEXT:    %[[V316:.*]] = llvm.extractvalue %[[V303]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V317:.*]] = llvm.getelementptr inbounds|nuw %[[V316]][%[[V304]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V315]], %[[V317]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V318:.*]] = llvm.add %[[V304]], %[[V42]] : i64
// CHECK-NEXT:    llvm.br ^bb1(%[[V318]] : i64)
// CHECK-NEXT:    ^bb7:  // pred: ^bb1
// CHECK-NEXT:    %[[V319:.*]] = llvm.extractvalue %[[V277]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V319]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V320:.*]] = llvm.extractvalue %[[V245]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V320]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V321:.*]] = llvm.extractvalue %[[V303]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V322:.*]] = llvm.getelementptr inbounds|nuw %[[V321]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V323:.*]] = llvm.load %[[V322]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V324:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V325:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V326:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V327:.*]] = llvm.getelementptr %[[V326]][%[[V324]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V328:.*]] = llvm.ptrtoint %[[V327]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V329:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V330:.*]] = llvm.add %[[V328]], %[[V329]] : i64
// CHECK-NEXT:    %[[V331:.*]] = llvm.call @malloc(%[[V330]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V332:.*]] = llvm.ptrtoint %[[V331]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V333:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V334:.*]] = llvm.sub %[[V329]], %[[V333]] : i64
// CHECK-NEXT:    %[[V335:.*]] = llvm.add %[[V332]], %[[V334]] : i64
// CHECK-NEXT:    %[[V336:.*]] = llvm.urem %[[V335]], %[[V329]] : i64
// CHECK-NEXT:    %[[V337:.*]] = llvm.sub %[[V335]], %[[V336]] : i64
// CHECK-NEXT:    %[[V338:.*]] = llvm.inttoptr %[[V337]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V339:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V340:.*]] = llvm.insertvalue %[[V331]], %[[V339]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V341:.*]] = llvm.insertvalue %[[V338]], %[[V340]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V342:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V343:.*]] = llvm.insertvalue %[[V342]], %[[V341]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V344:.*]] = llvm.insertvalue %[[V324]], %[[V343]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V345:.*]] = llvm.insertvalue %[[V325]], %[[V344]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V346:.*]] = llvm.extractvalue %[[V345]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V347:.*]] = llvm.getelementptr inbounds|nuw %[[V346]][%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V323]], %[[V347]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V348:.*]] = llvm.extractvalue %[[V345]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V349:.*]] = llvm.getelementptr inbounds|nuw %[[V348]][%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V349]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V350:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V351:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V352:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V353:.*]] = llvm.getelementptr %[[V352]][%[[V350]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V354:.*]] = llvm.ptrtoint %[[V353]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V355:.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V356:.*]] = llvm.add %[[V354]], %[[V355]] : i64
// CHECK-NEXT:    %[[V357:.*]] = llvm.call @malloc(%[[V356]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V358:.*]] = llvm.ptrtoint %[[V357]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V359:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V360:.*]] = llvm.sub %[[V355]], %[[V359]] : i64
// CHECK-NEXT:    %[[V361:.*]] = llvm.add %[[V358]], %[[V360]] : i64
// CHECK-NEXT:    %[[V362:.*]] = llvm.urem %[[V361]], %[[V355]] : i64
// CHECK-NEXT:    %[[V363:.*]] = llvm.sub %[[V361]], %[[V362]] : i64
// CHECK-NEXT:    %[[V364:.*]] = llvm.inttoptr %[[V363]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V365:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V366:.*]] = llvm.insertvalue %[[V357]], %[[V365]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V367:.*]] = llvm.insertvalue %[[V364]], %[[V366]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V368:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V369:.*]] = llvm.insertvalue %[[V368]], %[[V367]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V370:.*]] = llvm.insertvalue %[[V350]], %[[V369]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V371:.*]] = llvm.insertvalue %[[V351]], %[[V370]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V372:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V373:.*]] = llvm.extractvalue %[[V345]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V374:.*]] = llvm.mul %[[V372]], %[[V373]] : i64
// CHECK-NEXT:    %[[V375:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V376:.*]] = llvm.getelementptr %[[V375]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V377:.*]] = llvm.ptrtoint %[[V376]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V378:.*]] = llvm.mul %[[V374]], %[[V377]] : i64
// CHECK-NEXT:    %[[V379:.*]] = llvm.extractvalue %[[V345]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V380:.*]] = llvm.extractvalue %[[V345]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V381:.*]] = llvm.getelementptr %[[V379]][%[[V380]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V382:.*]] = llvm.extractvalue %[[V371]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V383:.*]] = llvm.extractvalue %[[V371]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V384:.*]] = llvm.getelementptr %[[V382]][%[[V383]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V384]], %[[V381]], %[[V378]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V385:.*]] = llvm.extractvalue %[[V345]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V385]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V386:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V387:.*]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V386]], %[[V9]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V388:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V389:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V390:.*]] = llvm.insertvalue %[[V387]], %[[V389]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V391:.*]] = llvm.insertvalue %[[V387]], %[[V390]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V392:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V393:.*]] = llvm.insertvalue %[[V392]], %[[V391]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V394:.*]] = llvm.insertvalue %[[V9]], %[[V393]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V395:.*]] = llvm.insertvalue %[[V388]], %[[V394]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V396:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V397:.*]] = llvm.extractvalue %[[V395]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V398:.*]] = llvm.insertvalue %[[V397]], %[[V396]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V399:.*]] = llvm.extractvalue %[[V395]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V400:.*]] = llvm.getelementptr %[[V399]][%[[V43]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V401:.*]] = llvm.insertvalue %[[V400]], %[[V398]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V402:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V403:.*]] = llvm.insertvalue %[[V402]], %[[V401]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V404:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V405:.*]] = llvm.insertvalue %[[V404]], %[[V403]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V406:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V407:.*]] = llvm.insertvalue %[[V406]], %[[V405]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V408:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V409:.*]] = llvm.insertvalue %[[V408]], %[[V407]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V410:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V411:.*]] = llvm.insertvalue %[[V410]], %[[V409]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V412:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V413:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V414:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V415:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V416:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V417:.*]] = llvm.getelementptr %[[V416]][%[[V415]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:    %[[V418:.*]] = llvm.ptrtoint %[[V417]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V419:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V420:.*]] = llvm.alloca %[[V419]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V421:.*]] = llvm.getelementptr %[[V420]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V412]], %[[V421]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V422:.*]] = llvm.getelementptr %[[V420]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V413]], %[[V422]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V423:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V424:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V425:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V426:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V423]], %[[V420]], %[[V424]], %[[V425]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V427:.*]] = llvm.addrspacecast %[[V426]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V428:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V429:.*]] = llvm.insertvalue %[[V427]], %[[V428]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V430:.*]] = llvm.insertvalue %[[V427]], %[[V429]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V431:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V432:.*]] = llvm.insertvalue %[[V431]], %[[V430]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V433:.*]] = llvm.insertvalue %[[V412]], %[[V432]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V434:.*]] = llvm.insertvalue %[[V413]], %[[V433]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V435:.*]] = llvm.insertvalue %[[V413]], %[[V434]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V436:.*]] = llvm.insertvalue %[[V414]], %[[V435]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V437:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V438:.*]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V439:.*]] = llvm.alloca %[[V437]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V440:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V441:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V442:.*]] = llvm.getelementptr %[[V439]][%[[V441]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V440]], %[[V442]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V443:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V444:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V445:.*]] = llvm.getelementptr %[[V439]][%[[V444]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V443]], %[[V445]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V446:.*]] = llvm.alloca %[[V437]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V447:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V448:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V449:.*]] = llvm.getelementptr %[[V446]][%[[V448]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V447]], %[[V449]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V450:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V451:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V452:.*]] = llvm.getelementptr %[[V446]][%[[V451]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V450]], %[[V452]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V453:.*]] = llvm.extractvalue %[[V172]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V454:.*]] = llvm.extractvalue %[[V411]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V455:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V456:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V457:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V458:.*]] = llvm.call @wrap_expand(%[[ARG0]], %[[V453]], %[[V438]], %[[V454]], %[[V439]], %[[V455]], %[[V446]], %[[V456]], %[[V457]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V459:.*]] = llvm.extractvalue %[[V194]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V459]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V460:.*]] = llvm.extractvalue %[[V411]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V461:.*]] = llvm.extractvalue %[[V411]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V462:.*]] = llvm.extractvalue %[[V26]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V463:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V464:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V465:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V466:.*]] = llvm.extractvalue %[[V411]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V467:.*]] = llvm.extractvalue %[[V26]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V468:.*]] = llvm.extractvalue %[[V436]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V469:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V470:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V471:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V472:.*]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V465]], %[[V466]], %[[V467]], %[[V468]], %[[V460]], %[[V462]], %[[V461]], %[[V463]], %[[V469]], %[[V464]], %[[V470]], %[[V471]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V473:.*]] = llvm.extractvalue %[[V303]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V473]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V474:.*]] = llvm.extractvalue %[[V371]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V474]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V436]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
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
