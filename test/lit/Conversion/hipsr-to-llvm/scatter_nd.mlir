// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @wrap_scatter_nd(
// CHECK-SAME:  !llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32

// Each memref arrives unpacked: an allocated pointer, an aligned pointer, an
// offset, then one size and one stride per axis. The context comes first, then
// the data, the indices, the updates and the destination.
// CHECK-NEXT:  llvm.func @scatter_nd(
// CHECK-SAME:      %arg0: !llvm.ptr,
// CHECK-SAME:      %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64,
// CHECK-SAME:      %arg8: !llvm.ptr<1>, %arg9: !llvm.ptr<1>, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64,
// CHECK-SAME:      %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>, %arg17: i64, %arg18: i64, %arg19: i64,
// CHECK-SAME:      %arg20: !llvm.ptr<1>, %arg21: !llvm.ptr<1>, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64) {

// A descriptor per operand, rebuilt field by field, which pins each pointer and
// each dynamic size to the argument it came from. Sizes land in field 3 and
// strides in field 4.
// CHECK-NEXT:  %[[OUT_0:.*]] = llvm.mlir.poison : ![[DESC2:.*]]
// CHECK-NEXT:  %[[OUT_1:.*]] = llvm.insertvalue %arg20, %[[OUT_0]][0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_2:.*]] = llvm.insertvalue %arg21, %[[OUT_1]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_3:.*]] = llvm.insertvalue %arg22, %[[OUT_2]][2] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_4:.*]] = llvm.insertvalue %arg23, %[[OUT_3]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_5:.*]] = llvm.insertvalue %arg25, %[[OUT_4]][4, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_6:.*]] = llvm.insertvalue %arg24, %[[OUT_5]][3, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_DESC:.*]] = llvm.insertvalue %arg26, %[[OUT_6]][4, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[UPDATES_0:.*]] = llvm.mlir.poison : ![[DESC1:.*]]
// CHECK-NEXT:  %[[UPDATES_1:.*]] = llvm.insertvalue %arg15, %[[UPDATES_0]][0] : ![[DESC1]]
// CHECK-NEXT:  %[[UPDATES_2:.*]] = llvm.insertvalue %arg16, %[[UPDATES_1]][1] : ![[DESC1]]
// CHECK-NEXT:  %[[UPDATES_3:.*]] = llvm.insertvalue %arg17, %[[UPDATES_2]][2] : ![[DESC1]]
// CHECK-NEXT:  %[[UPDATES_4:.*]] = llvm.insertvalue %arg18, %[[UPDATES_3]][3, 0] : ![[DESC1]]
// CHECK-NEXT:  %[[UPDATES_DESC:.*]] = llvm.insertvalue %arg19, %[[UPDATES_4]][4, 0] : ![[DESC1]]
// CHECK-NEXT:  %[[IDS_0:.*]] = llvm.mlir.poison : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_1:.*]] = llvm.insertvalue %arg8, %[[IDS_0]][0] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_2:.*]] = llvm.insertvalue %arg9, %[[IDS_1]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_3:.*]] = llvm.insertvalue %arg10, %[[IDS_2]][2] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_4:.*]] = llvm.insertvalue %arg11, %[[IDS_3]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_5:.*]] = llvm.insertvalue %arg13, %[[IDS_4]][4, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_6:.*]] = llvm.insertvalue %arg12, %[[IDS_5]][3, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_DESC:.*]] = llvm.insertvalue %arg14, %[[IDS_6]][4, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_0:.*]] = llvm.mlir.poison : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_1:.*]] = llvm.insertvalue %arg1, %[[DATA_0]][0] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_2:.*]] = llvm.insertvalue %arg2, %[[DATA_1]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_3:.*]] = llvm.insertvalue %arg3, %[[DATA_2]][2] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_4:.*]] = llvm.insertvalue %arg4, %[[DATA_3]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_5:.*]] = llvm.insertvalue %arg6, %[[DATA_4]][4, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_6:.*]] = llvm.insertvalue %arg5, %[[DATA_5]][3, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_DESC:.*]] = llvm.insertvalue %arg7, %[[DATA_6]][4, 1] : ![[DESC2]]

// The call carries all four shapes as host arrays next to their ranks: a static
// extent is a constant, a dynamic one comes off the descriptor.
// CHECK-NEXT:  %[[DATA_DIM0:.*]] = llvm.extractvalue %[[DATA_DESC]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[DATA_DIM1:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  %[[DATA_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[DATA_SHAPE:.*]] = llvm.alloca %[[DATA_ONE]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:  %[[DATA_SLOT0:.*]] = llvm.getelementptr %[[DATA_SHAPE]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[DATA_DIM0]], %[[DATA_SLOT0]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[DATA_SLOT1:.*]] = llvm.getelementptr %[[DATA_SHAPE]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[DATA_DIM1]], %[[DATA_SLOT1]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[IDS_DIM0:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:  %[[IDS_DIM1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[IDS_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[IDS_SHAPE:.*]] = llvm.alloca %[[IDS_ONE]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:  %[[IDS_SLOT0:.*]] = llvm.getelementptr %[[IDS_SHAPE]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[IDS_DIM0]], %[[IDS_SLOT0]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[IDS_SLOT1:.*]] = llvm.getelementptr %[[IDS_SHAPE]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[IDS_DIM1]], %[[IDS_SLOT1]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[UPDATES_DIM0:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:  %[[UPDATES_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[UPDATES_SHAPE:.*]] = llvm.alloca %[[UPDATES_ONE]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:  %[[UPDATES_SLOT0:.*]] = llvm.getelementptr %[[UPDATES_SHAPE]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[UPDATES_DIM0]], %[[UPDATES_SLOT0]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[OUT_DIM0:.*]] = llvm.extractvalue %[[OUT_DESC]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_DIM1:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  %[[OUT_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[OUT_SHAPE:.*]] = llvm.alloca %[[OUT_ONE]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:  %[[OUT_SLOT0:.*]] = llvm.getelementptr %[[OUT_SHAPE]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[OUT_DIM0]], %[[OUT_SLOT0]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[OUT_SLOT1:.*]] = llvm.getelementptr %[[OUT_SHAPE]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[OUT_DIM1]], %[[OUT_SLOT1]] : i64, !llvm.ptr

// Field 1 is the aligned pointer, the one the runtime reads and writes. The
// sixth pointer is the row count that trims padded indices, which this op has
// no operand for, so it goes over as null.
// CHECK-NEXT:  %[[DATA_PTR:.*]] = llvm.extractvalue %[[DATA_DESC]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_PTR:.*]] = llvm.extractvalue %[[IDS_DESC]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[UPDATES_PTR:.*]] = llvm.extractvalue %[[UPDATES_DESC]][1] : ![[DESC1]]
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue %[[OUT_DESC]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[COUNT_PTR:.*]] = llvm.mlir.zero : !llvm.ptr<1>
// CHECK-NEXT:  %[[DATA_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[IDS_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[UPDATES_RANK:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[OUT_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[REDUCTION:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[DATA_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %{{.*}} = llvm.call @wrap_scatter_nd(%arg0, %[[DATA_PTR]], %[[IDS_PTR]], %[[UPDATES_PTR]], %[[OUT_PTR]], %[[COUNT_PTR]], %[[DATA_SHAPE]], %[[DATA_RANK]], %[[IDS_SHAPE]], %[[IDS_RANK]], %[[UPDATES_SHAPE]], %[[UPDATES_RANK]], %[[OUT_SHAPE]], %[[OUT_RANK]], %[[REDUCTION]], %[[DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }
func.func @scatter_nd(%ctx: !hipsr.context,
                      %data: memref<?x4xf16, #hipsr.mem<device>>,
                      %ids: memref<3x2xi64, #hipsr.mem<device>>,
                      %updates: memref<3xf16, #hipsr.mem<device>>,
                      %init: memref<?x4xf16, #hipsr.mem<device>>) {
  hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : memref<?x4xf16, #hipsr.mem<device>>,
                                  memref<3x2xi64, #hipsr.mem<device>>,
                                  memref<3xf16, #hipsr.mem<device>>)
      outs(%init : memref<?x4xf16, #hipsr.mem<device>>)
  return
}
