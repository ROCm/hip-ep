// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// The runtime resolves the window against the data shape, so the call carries
// both shapes as host arrays plus how many entries the window holds. Each slot
// reaches it as a host pointer: `starts`, `axes` and `steps` are attributes, so
// their entries go on the stack, while `ends` is an operand and its buffer goes
// over as an address-space-0 pointer. Every slot is the same length, so one
// count covers the three the runtime takes.
// CHECK-LABEL: llvm.func @slice
// CHECK-SAME:  (%[[CTX:[^,]+]]: !llvm.ptr,
// CHECK:       %[[DATA_DIM0:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK:       %[[DATA_SHAPE:.*]] = llvm.alloca {{.*}} x !llvm.array<2 x i64>
// CHECK-NEXT:  %[[DATA_SLOT0:.*]] = llvm.getelementptr %[[DATA_SHAPE]][0]
// CHECK-NEXT:  llvm.store %[[DATA_DIM0]], %[[DATA_SLOT0]]
// CHECK:       %[[OUT_DIM0:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK:       %[[OUT_SHAPE:.*]] = llvm.alloca {{.*}} x !llvm.array<2 x i64>
// CHECK-NEXT:  %[[OUT_SLOT0:.*]] = llvm.getelementptr %[[OUT_SHAPE]][0]
// CHECK-NEXT:  llvm.store %[[OUT_DIM0]], %[[OUT_SLOT0]]
// CHECK:       %[[START0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:       %[[STARTS_PTR:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK-NEXT:  %[[STARTS_SLOT0:.*]] = llvm.getelementptr %[[STARTS_PTR]][0]
// CHECK-NEXT:  llvm.store %[[START0]], %[[STARTS_SLOT0]]
// CHECK-NEXT:  %[[ENDS_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr,
// CHECK-NEXT:  %[[AXIS0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:       %[[AXES_PTR:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK-NEXT:  %[[AXES_SLOT0:.*]] = llvm.getelementptr %[[AXES_PTR]][0]
// CHECK-NEXT:  llvm.store %[[AXIS0]], %[[AXES_SLOT0]]
// CHECK:       %[[STEPS_PTR:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK-NEXT:  %[[STEPS_SLOT0:.*]] = llvm.getelementptr %[[STEPS_PTR]][0]
// CHECK-NEXT:  llvm.store %{{.*}}, %[[STEPS_SLOT0]]
// CHECK-NEXT:  %[[ENTRIES:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  %[[DATA_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[OUT_PTR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<1>,
// CHECK-NEXT:  %[[DATA_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[OUT_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:  %[[DATA_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:  llvm.call @wrap_slice(%[[CTX]], %[[DATA_PTR]], %[[STARTS_PTR]], %[[ENDS_PTR]], %[[AXES_PTR]], %[[STEPS_PTR]], %[[OUT_PTR]], %[[DATA_SHAPE]], %[[DATA_RANK]], %[[OUT_SHAPE]], %[[OUT_RANK]], %[[ENTRIES]], %[[ENTRIES]], %[[ENTRIES]], %[[DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
func.func @slice(%ctx: !hipsr.context,
                 %data: memref<8x4xf16, #hipsr.mem<device>>,
                 %ends: memref<1xi64, #hipsr.mem<host>>,
                 %init: memref<3x4xf16, #hipsr.mem<device>>) {
  hipsr.slice(%ctx)
      ins(%data : memref<8x4xf16, #hipsr.mem<device>>)
      ends(%ends : memref<1xi64, #hipsr.mem<host>>)
      outs(%init : memref<3x4xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, axes_attr = array<i64: 0>,
       steps_attr = array<i64: 1>}
  return
}
