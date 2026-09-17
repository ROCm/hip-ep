// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @wrap_cast(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-LABEL: llvm.func @cast_static(
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
// CHECK-SAME:    %[[ARG14:[^,]*]]: i64) {
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
// CHECK-NEXT:    %[[V16:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V17:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V18:.*]] = llvm.mul %[[V16]], %[[V17]] : i64
// CHECK-NEXT:    %[[V19:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %[[V20:.*]] = llvm.mul %[[V18]], %[[V19]] : i64
// CHECK-NEXT:    %[[V21:.*]] = llvm.extractvalue %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.*]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V24:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V25:.*]] = llvm.call @wrap_cast(%[[ARG0]], %[[V21]], %[[V22]], %[[V20]], %[[V23]], %[[V24]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @cast_static(
    %ctx: !hipsr.context,
    %input: memref<4x8xf32, #hipsr.mem<device>>,
    %init: memref<4x8xf16, #hipsr.mem<device>>) {
  hipsr.cast(%ctx)
      ins(%input : memref<4x8xf32, #hipsr.mem<device>>)
      outs(%init : memref<4x8xf16, #hipsr.mem<device>>)
  return
}

// -----

// CHECK-LABEL: llvm.func @wrap_cast(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-LABEL: llvm.func @cast_dynamic(
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
// CHECK-SAME:    %[[ARG14:[^,]*]]: i64) {
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
// CHECK-NEXT:    %[[V16:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V17:.*]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V18:.*]] = llvm.mul %[[V16]], %[[V17]] : i64
// CHECK-NEXT:    %[[V19:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %[[V20:.*]] = llvm.mul %[[V18]], %[[V19]] : i64
// CHECK-NEXT:    %[[V21:.*]] = llvm.extractvalue %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.*]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V24:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V25:.*]] = llvm.call @wrap_cast(%[[ARG0]], %[[V21]], %[[V22]], %[[V20]], %[[V23]], %[[V24]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @cast_dynamic(
    %ctx: !hipsr.context,
    %input: memref<?x8xf16, #hipsr.mem<device>>,
    %init: memref<?x8xf32, #hipsr.mem<device>>) {
  hipsr.cast(%ctx)
      ins(%input : memref<?x8xf16, #hipsr.mem<device>>)
      outs(%init : memref<?x8xf32, #hipsr.mem<device>>)
  return
}
