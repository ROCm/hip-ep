// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// The whole function is checked. Seventeen operands reach the runtime call, and
// which value lands in which position is the whole contract, so a check that
// skipped the descriptor and extent lines would let two of them swap unnoticed.

// The lowering declares the runtime entry point before calling it.
// CHECK-LABEL:   llvm.func @wrap_equal(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// The call's data type is 4 (i64), the operand type being compared. The mask
// needs none of its own: the runtime always writes one byte per element.
// CHECK-LABEL:   llvm.func @equal(
// CHECK-SAME:                     %[[CTX:[^,]*]]: !llvm.ptr,
// CHECK-SAME:                     %[[ARG1:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:                     %[[ARG2:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:                     %[[ARG3:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG4:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG5:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG6:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG7:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG8:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:                     %[[ARG9:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:                     %[[ARG10:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG11:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG12:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG13:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:                     %[[ARG14:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:                     %[[ARG15:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG16:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG17:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG18:[^,]*]]: i64,
// CHECK-SAME:                     %[[ARG19:[^,]*]]: i64) {
// Each memref arrives as unpacked fields, so the lowering rebuilds a descriptor
// per operand, in the order init, rhs, lhs. Only the aligned pointer each one
// holds at index 1 reaches the call.
// CHECK-NEXT:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG13]], %[[MLIR_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG14]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG15]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[ARG16]], %[[INSERTVALUE_2]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[ARG18]], %[[INSERTVALUE_3]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[ARG17]], %[[INSERTVALUE_4]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_6:.*]] = llvm.insertvalue %[[ARG19]], %[[INSERTVALUE_5]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[MLIR_1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_7:.*]] = llvm.insertvalue %[[ARG8]], %[[MLIR_1]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_8:.*]] = llvm.insertvalue %[[ARG9]], %[[INSERTVALUE_7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_9:.*]] = llvm.insertvalue %[[ARG10]], %[[INSERTVALUE_8]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_10:.*]] = llvm.insertvalue %[[ARG11]], %[[INSERTVALUE_9]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_11:.*]] = llvm.insertvalue %[[ARG12]], %[[INSERTVALUE_10]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[MLIR_2:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_12:.*]] = llvm.insertvalue %[[ARG1]], %[[MLIR_2]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_13:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_12]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_14:.*]] = llvm.insertvalue %[[ARG3]], %[[INSERTVALUE_13]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_15:.*]] = llvm.insertvalue %[[ARG4]], %[[INSERTVALUE_14]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_16:.*]] = llvm.insertvalue %[[ARG6]], %[[INSERTVALUE_15]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_17:.*]] = llvm.insertvalue %[[ARG5]], %[[INSERTVALUE_16]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_18:.*]] = llvm.insertvalue %[[ARG7]], %[[INSERTVALUE_17]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[LHS_N:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[LHS_C:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[LHS_H:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:           %[[LHS_W:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:           %[[RHS_N:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[RHS_C:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[RHS_H:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[RHS_W:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:           %[[OUT_N:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[OUT_C:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[OUT_H:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:           %[[OUT_W:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK-NEXT:           %[[LHS_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_18]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[RHS_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_11]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[OUT_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_6]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[DATA_TYPE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:           %[[CALL_0:.*]] = llvm.call @wrap_equal(%[[CTX]], %[[LHS_PTR]], %[[RHS_PTR]], %[[OUT_PTR]], %[[LHS_N]], %[[LHS_C]], %[[LHS_H]], %[[LHS_W]], %[[RHS_N]], %[[RHS_C]], %[[RHS_H]], %[[RHS_W]], %[[OUT_N]], %[[OUT_C]], %[[OUT_H]], %[[OUT_W]], %[[DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
func.func @equal(%ctx: !hipsr.context,
                 %lhs: memref<4x1024xi64, #hipsr.mem<device>>,
                 %rhs: memref<1024xi64, #hipsr.mem<device>>,
                 %init: memref<4x1024xi1, #hipsr.mem<device>>) {
  hipsr.equal(%ctx) ins(%lhs, %rhs : memref<4x1024xi64, #hipsr.mem<device>>, memref<1024xi64, #hipsr.mem<device>>)
               outs(%init : memref<4x1024xi1, #hipsr.mem<device>>)
  return
}
