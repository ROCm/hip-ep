// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// The input shape and perm go on the stack, because the runtime reads them on
// the host. It takes the element count and an element width in bytes, here 8
// for i64, rather than a data type.
//
// Each memref arrives as its fields and is rebuilt into a descriptor, the
// destination's first, so the two pointers the call takes are pinned to the
// operand each comes from.
// CHECK-LABEL: llvm.func @wrap_transpose(
// CHECK-SAME:  !llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @transpose(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: !llvm.ptr<1>, %arg9: !llvm.ptr<1>, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64) {
// CHECK-NEXT:  %[[OUT_POISON:.*]] = llvm.mlir.poison : ![[DESC:.*]]
// CHECK-NEXT:  %[[OUT_ALLOC:.*]] = llvm.insertvalue %arg8, %[[OUT_POISON]][0] : ![[DESC]]
// CHECK-NEXT:  %[[OUT_ALIGNED:.*]] = llvm.insertvalue %arg9, %[[OUT_ALLOC]][1] : ![[DESC]]
// CHECK-NEXT:  %[[OUT_OFFSET:.*]] = llvm.insertvalue %arg10, %[[OUT_ALIGNED]][2] : ![[DESC]]
// CHECK-NEXT:  %[[OUT_SIZE0:.*]] = llvm.insertvalue %arg11, %[[OUT_OFFSET]][3, 0] : ![[DESC]]
// CHECK-NEXT:  %[[OUT_STRIDE0:.*]] = llvm.insertvalue %arg13, %[[OUT_SIZE0]][4, 0] : ![[DESC]]
// CHECK-NEXT:  %[[OUT_SIZE1:.*]] = llvm.insertvalue %arg12, %[[OUT_STRIDE0]][3, 1] : ![[DESC]]
// CHECK-NEXT:  %[[OUT_DESC:.*]] = llvm.insertvalue %arg14, %[[OUT_SIZE1]][4, 1] : ![[DESC]]
// CHECK-NEXT:  %[[IN_POISON:.*]] = llvm.mlir.poison : ![[DESC]]
// CHECK-NEXT:  %[[IN_ALLOC:.*]] = llvm.insertvalue %arg1, %[[IN_POISON]][0] : ![[DESC]]
// CHECK-NEXT:  %[[IN_ALIGNED:.*]] = llvm.insertvalue %arg2, %[[IN_ALLOC]][1] : ![[DESC]]
// CHECK-NEXT:  %[[IN_OFFSET:.*]] = llvm.insertvalue %arg3, %[[IN_ALIGNED]][2] : ![[DESC]]
// CHECK-NEXT:  %[[IN_SIZE0:.*]] = llvm.insertvalue %arg4, %[[IN_OFFSET]][3, 0] : ![[DESC]]
// CHECK-NEXT:  %[[IN_STRIDE0:.*]] = llvm.insertvalue %arg6, %[[IN_SIZE0]][4, 0] : ![[DESC]]
// CHECK-NEXT:  %[[IN_SIZE1:.*]] = llvm.insertvalue %arg5, %[[IN_STRIDE0]][3, 1] : ![[DESC]]
// CHECK-NEXT:  %[[IN_DESC:.*]] = llvm.insertvalue %arg7, %[[IN_SIZE1]][4, 1] : ![[DESC]]
// CHECK-NEXT:  %[[ROWS:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:  %[[COLS:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:  %[[COUNT:.*]] = llvm.mul %[[ROWS]], %[[COLS]] : i64
// CHECK-NEXT:  %[[SHAPE_LEN:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[SHAPE:.*]] = llvm.alloca %[[SHAPE_LEN]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:  %[[SHAPE_0:.*]] = llvm.getelementptr %[[SHAPE]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[ROWS]], %[[SHAPE_0]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[SHAPE_1:.*]] = llvm.getelementptr %[[SHAPE]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[COLS]], %[[SHAPE_1]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[AXIS_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[AXIS_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[PERM_LEN:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[PERM:.*]] = llvm.alloca %[[PERM_LEN]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:  %[[PERM_0:.*]] = llvm.getelementptr %[[PERM]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[AXIS_1]], %[[PERM_0]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[PERM_1:.*]] = llvm.getelementptr %[[PERM]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:  llvm.store %[[AXIS_0]], %[[PERM_1]] : i64, !llvm.ptr
// CHECK-NEXT:  %[[IN_PTR:.*]] = llvm.extractvalue %[[IN_DESC]][1] : ![[DESC]]
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue %[[OUT_DESC]][1] : ![[DESC]]
// CHECK-NEXT:  %[[RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[WIDTH:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:  %{{.*}} = llvm.call @wrap_transpose(%arg0, %[[IN_PTR]], %[[OUT_PTR]], %[[RANK]], %[[SHAPE]], %[[PERM]], %[[COUNT]], %[[WIDTH]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }
func.func @transpose(%ctx: !hipsr.context,
                     %input: memref<3x1024xi64, #hipsr.mem<device>>,
                     %init: memref<1024x3xi64, #hipsr.mem<device>>) {
  hipsr.transpose(%ctx) ins(%input : memref<3x1024xi64, #hipsr.mem<device>>)
      outs(%init : memref<1024x3xi64, #hipsr.mem<device>>)
      {perm = array<i64: 1, 0>}
  return
}
