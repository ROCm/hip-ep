// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --convert-to-llvm | FileCheck %s

// Count is i32; positions are i64.
// CHECK-LABEL: llvm.func @wrap_nonzero(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-LABEL: llvm.func @nonzero_static(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG2:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG3:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG4:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG5:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG6:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG7:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG8:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG9:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG10:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG11:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG12:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG13:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG14:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG15:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG16:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG17:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG18:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG19:[^,]*]]: i64) {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1:.*]] = llvm.insertvalue %[[ARG15]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[ARG16]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[ARG17]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[ARG18]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[ARG19]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[ARG8]], %[[V6]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.insertvalue %[[ARG9]], %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V9:.*]] = llvm.insertvalue %[[ARG10]], %[[V8]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V10:.*]] = llvm.insertvalue %[[ARG11]], %[[V9]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V11:.*]] = llvm.insertvalue %[[ARG13]], %[[V10]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V12:.*]] = llvm.insertvalue %[[ARG12]], %[[V11]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V13:.*]] = llvm.insertvalue %[[ARG14]], %[[V12]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V14:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.*]] = llvm.insertvalue %[[ARG1]], %[[V14]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.*]] = llvm.insertvalue %[[ARG2]], %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V17:.*]] = llvm.insertvalue %[[ARG3]], %[[V16]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V18:.*]] = llvm.insertvalue %[[ARG4]], %[[V17]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V19:.*]] = llvm.insertvalue %[[ARG6]], %[[V18]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V20:.*]] = llvm.insertvalue %[[ARG5]], %[[V19]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.*]] = llvm.insertvalue %[[ARG7]], %[[V20]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V23:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V24:.*]] = llvm.mul %[[V22]], %[[V23]] : i64
// CHECK-NEXT:    %[[V25:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V26:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK-NEXT:    %[[V27:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V28:.*]] = llvm.alloca %[[V27]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V29:.*]] = llvm.getelementptr %[[V28]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V22]], %[[V29]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V30:.*]] = llvm.getelementptr %[[V28]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V23]], %[[V30]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V31:.*]] = llvm.extractvalue %[[V21]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V32:.*]] = llvm.extractvalue %[[V13]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V33:.*]] = llvm.extractvalue %[[V5]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V34:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V35:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:    %[[V36:.*]] = llvm.call @wrap_nonzero(%[[ARG0]], %[[V31]], %[[V32]], %[[V33]], %[[V24]], %[[V34]], %[[V28]], %[[V26]], %[[V35]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @nonzero_static(%ctx: !hipsr.context,
                          %mask: memref<2x3xi1, #hipsr.mem<device>>,
                          %indices: memref<2x6xi64, #hipsr.mem<device>>,
                          %count: memref<1xi32, #hipsr.mem<device>>) {
  hipsr.nonzero(%ctx) ins(%mask : memref<2x3xi1, #hipsr.mem<device>>)
      outs(%indices, %count : memref<2x6xi64, #hipsr.mem<device>>,
                              memref<1xi32, #hipsr.mem<device>>)
  return
}

// -----

// Dynamic extents and capacity come from the memref descriptors.
// CHECK-LABEL: llvm.func @wrap_nonzero(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-LABEL: llvm.func @nonzero_dynamic(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG2:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG3:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG4:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG5:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG6:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG7:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG8:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG9:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG10:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG11:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG12:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG13:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG14:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG15:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG16:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG17:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG18:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG19:[^,]*]]: i64) {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1:.*]] = llvm.insertvalue %[[ARG15]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[ARG16]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[ARG17]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[ARG18]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[ARG19]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[ARG8]], %[[V6]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.insertvalue %[[ARG9]], %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V9:.*]] = llvm.insertvalue %[[ARG10]], %[[V8]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V10:.*]] = llvm.insertvalue %[[ARG11]], %[[V9]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V11:.*]] = llvm.insertvalue %[[ARG13]], %[[V10]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V12:.*]] = llvm.insertvalue %[[ARG12]], %[[V11]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V13:.*]] = llvm.insertvalue %[[ARG14]], %[[V12]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V14:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.*]] = llvm.insertvalue %[[ARG1]], %[[V14]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.*]] = llvm.insertvalue %[[ARG2]], %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V17:.*]] = llvm.insertvalue %[[ARG3]], %[[V16]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V18:.*]] = llvm.insertvalue %[[ARG4]], %[[V17]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V19:.*]] = llvm.insertvalue %[[ARG6]], %[[V18]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V20:.*]] = llvm.insertvalue %[[ARG5]], %[[V19]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.*]] = llvm.insertvalue %[[ARG7]], %[[V20]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.*]] = llvm.extractvalue %[[V21]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.*]] = llvm.extractvalue %[[V21]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V24:.*]] = llvm.mul %[[V22]], %[[V23]] : i64
// CHECK-NEXT:    %[[V25:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V26:.*]] = llvm.extractvalue %[[V13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V27:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V28:.*]] = llvm.alloca %[[V27]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V29:.*]] = llvm.getelementptr %[[V28]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V22]], %[[V29]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V30:.*]] = llvm.getelementptr %[[V28]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V23]], %[[V30]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V31:.*]] = llvm.extractvalue %[[V21]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V32:.*]] = llvm.extractvalue %[[V13]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V33:.*]] = llvm.extractvalue %[[V5]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V34:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V35:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-NEXT:    %[[V36:.*]] = llvm.call @wrap_nonzero(%[[ARG0]], %[[V31]], %[[V32]], %[[V33]], %[[V24]], %[[V34]], %[[V28]], %[[V26]], %[[V35]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @nonzero_dynamic(%ctx: !hipsr.context,
                           %mask: memref<?x?xi8, #hipsr.mem<device>>,
                           %indices: memref<2x?xi64, #hipsr.mem<device>>,
                           %count: memref<1xi32, #hipsr.mem<device>>) {
  hipsr.nonzero(%ctx) ins(%mask : memref<?x?xi8, #hipsr.mem<device>>)
      outs(%indices, %count : memref<2x?xi64, #hipsr.mem<device>>,
                              memref<1xi32, #hipsr.mem<device>>)
  return
}
