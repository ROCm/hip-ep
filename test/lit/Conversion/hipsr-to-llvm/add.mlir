// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @wrap_elementwise(!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-LABEL: llvm.func @add(
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
// CHECK-SAME:    %[[ARG13:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG14:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[ARG15:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG16:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG17:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG18:[^,]*]]: i64,
// CHECK-SAME:    %[[ARG19:[^,]*]]: i64) {
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1:.*]] = llvm.insertvalue %[[ARG13]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[ARG14]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[ARG15]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[ARG16]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.*]] = llvm.insertvalue %[[ARG18]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.*]] = llvm.insertvalue %[[ARG17]], %[[V5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.*]] = llvm.insertvalue %[[ARG19]], %[[V6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V9:.*]] = llvm.insertvalue %[[ARG8]], %[[V8]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V10:.*]] = llvm.insertvalue %[[ARG9]], %[[V9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V11:.*]] = llvm.insertvalue %[[ARG10]], %[[V10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V12:.*]] = llvm.insertvalue %[[ARG11]], %[[V11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V13:.*]] = llvm.insertvalue %[[ARG12]], %[[V12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V14:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.*]] = llvm.insertvalue %[[ARG1]], %[[V14]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.*]] = llvm.insertvalue %[[ARG2]], %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V17:.*]] = llvm.insertvalue %[[ARG3]], %[[V16]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V18:.*]] = llvm.insertvalue %[[ARG4]], %[[V17]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V19:.*]] = llvm.insertvalue %[[ARG6]], %[[V18]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V20:.*]] = llvm.insertvalue %[[ARG5]], %[[V19]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.*]] = llvm.insertvalue %[[ARG7]], %[[V20]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V23:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V24:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V25:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:    %[[V26:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V27:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V28:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V29:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:    %[[V30:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V31:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V32:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V33:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:    %[[V34:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    %[[V35:.*]] = llvm.extractvalue %[[V21]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V36:.*]] = llvm.extractvalue %[[V13]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V37:.*]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V38:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V39:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V40:.*]] = llvm.call @wrap_elementwise(%[[ARG0]], %[[V34]], %[[V35]], %[[V36]], %[[V37]], %[[V22]], %[[V23]], %[[V24]], %[[V25]], %[[V26]], %[[V27]], %[[V28]], %[[V29]], %[[V30]], %[[V31]], %[[V32]], %[[V33]], %[[V38]], %[[V39]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @add(%ctx: !hipsr.context,
               %lhs: memref<4x1024xf16, #hipsr.mem<device>>,
               %rhs: memref<1024xf16, #hipsr.mem<device>>,
               %init: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.add(%ctx) ins(%lhs, %rhs : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
             outs(%init : memref<4x1024xf16, #hipsr.mem<device>>)
  return
}
