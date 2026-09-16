// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @wrap_gather(
// CHECK-SAME:  !llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// Each memref arrives unpacked: an allocated pointer, an aligned pointer, an
// offset, then one size and one stride per axis. The context comes first, then
// the table, the indices and the destination.
// CHECK-NEXT:  llvm.func @gather(
// CHECK-SAME:      %arg0: !llvm.ptr,
// CHECK-SAME:      %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64,
// CHECK-SAME:      %arg8: !llvm.ptr<1>, %arg9: !llvm.ptr<1>, %arg10: i64, %arg11: i64, %arg12: i64,
// CHECK-SAME:      %arg13: !llvm.ptr<1>, %arg14: !llvm.ptr<1>, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64) {

// A descriptor per operand, rebuilt field by field, which is what pins each
// pointer and each dynamic size to the argument it came from. Sizes land in
// field 3 and strides in field 4.
// CHECK-NEXT:  %[[OUT_0:.*]] = llvm.mlir.poison : ![[DESC2:.*]]
// CHECK-NEXT:  %[[OUT_1:.*]] = llvm.insertvalue %arg13, %[[OUT_0]][0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_2:.*]] = llvm.insertvalue %arg14, %[[OUT_1]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_3:.*]] = llvm.insertvalue %arg15, %[[OUT_2]][2] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_4:.*]] = llvm.insertvalue %arg16, %[[OUT_3]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_5:.*]] = llvm.insertvalue %arg18, %[[OUT_4]][4, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_6:.*]] = llvm.insertvalue %arg17, %[[OUT_5]][3, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_DESC:.*]] = llvm.insertvalue %arg19, %[[OUT_6]][4, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_0:.*]] = llvm.mlir.poison : ![[DESC1:.*]]
// CHECK-NEXT:  %[[IDS_1:.*]] = llvm.insertvalue %arg8, %[[IDS_0]][0] : ![[DESC1]]
// CHECK-NEXT:  %[[IDS_2:.*]] = llvm.insertvalue %arg9, %[[IDS_1]][1] : ![[DESC1]]
// CHECK-NEXT:  %[[IDS_3:.*]] = llvm.insertvalue %arg10, %[[IDS_2]][2] : ![[DESC1]]
// CHECK-NEXT:  %[[IDS_4:.*]] = llvm.insertvalue %arg11, %[[IDS_3]][3, 0] : ![[DESC1]]
// CHECK-NEXT:  %[[IDS_DESC:.*]] = llvm.insertvalue %arg12, %[[IDS_4]][4, 0] : ![[DESC1]]
// CHECK-NEXT:  %[[TABLE_0:.*]] = llvm.mlir.poison : ![[DESC2]]
// CHECK-NEXT:  %[[TABLE_1:.*]] = llvm.insertvalue %arg1, %[[TABLE_0]][0] : ![[DESC2]]
// CHECK-NEXT:  %[[TABLE_2:.*]] = llvm.insertvalue %arg2, %[[TABLE_1]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[TABLE_3:.*]] = llvm.insertvalue %arg3, %[[TABLE_2]][2] : ![[DESC2]]
// CHECK-NEXT:  %[[TABLE_4:.*]] = llvm.insertvalue %arg4, %[[TABLE_3]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[TABLE_5:.*]] = llvm.insertvalue %arg6, %[[TABLE_4]][4, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[TABLE_6:.*]] = llvm.insertvalue %arg5, %[[TABLE_5]][3, 1] : ![[DESC2]]
// CHECK-NEXT:  %[[TABLE_DESC:.*]] = llvm.insertvalue %arg7, %[[TABLE_6]][4, 1] : ![[DESC2]]

// The runtime takes element counts rather than shapes, plus the two sizes it
// rebuilds the data layout from: axis_size 8 and inner_size 4. A static
// dimension is a constant, a dynamic one comes off the descriptor, and each
// count seeds with the first dimension so no `mul 1` is emitted.
// CHECK-NEXT:  %[[AXIS_SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:  %[[INNER_SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  %[[IDS_COUNT:.*]] = llvm.extractvalue %[[IDS_DESC]][3, 0] : ![[DESC1]]
// CHECK-NEXT:  %[[OUT_ROWS:.*]] = llvm.extractvalue %[[OUT_DESC]][3, 0] : ![[DESC2]]
// CHECK-NEXT:  %[[OUT_COLS:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:  %[[TABLE_COUNT:.*]] = llvm.mul %[[AXIS_SIZE]], %[[INNER_SIZE]] : i64
// CHECK-NEXT:  %[[OUT_COUNT:.*]] = llvm.mul %[[OUT_ROWS]], %[[OUT_COLS]] : i64

// Field 1 is the aligned pointer, the one the runtime reads and writes.
// CHECK-NEXT:  %[[TABLE_PTR:.*]] = llvm.extractvalue %[[TABLE_DESC]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[IDS_PTR:.*]] = llvm.extractvalue %[[IDS_DESC]][1] : ![[DESC1]]
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue %[[OUT_DESC]][1] : ![[DESC2]]
// CHECK-NEXT:  %[[AXIS:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[ELEMENT_BYTES:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[INDEX_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:  %{{.*}} = llvm.call @wrap_gather(%arg0, %[[TABLE_PTR]], %[[IDS_PTR]], %[[OUT_PTR]], %[[AXIS]], %[[TABLE_COUNT]], %[[IDS_COUNT]], %[[OUT_COUNT]], %[[AXIS_SIZE]], %[[INNER_SIZE]], %[[ELEMENT_BYTES]], %[[INDEX_BYTES]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }
func.func @gather(%ctx: !hipsr.context,
                  %table: memref<8x4xf16, #hipsr.mem<device>>,
                  %ids: memref<?xi64, #hipsr.mem<device>>,
                  %init: memref<?x4xf16, #hipsr.mem<device>>) {
  hipsr.gather(%ctx)
      ins(%table, %ids : memref<8x4xf16, #hipsr.mem<device>>,
                         memref<?xi64, #hipsr.mem<device>>)
      outs(%init : memref<?x4xf16, #hipsr.mem<device>>) {axis = 0 : i64}
  return
}
