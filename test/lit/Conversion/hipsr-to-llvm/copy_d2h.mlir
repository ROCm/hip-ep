// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --convert-to-llvm | FileCheck %s

// A graph input is always device memory, so the host destination is allocated
// inside the function and reaches the call as a malloc'd descriptor.
//
// The declaration is the calling convention: the context pointer, the host
// destination in address space 0, the device source in address space 1, and a
// byte count. The element size uses the `getelementptr null, 1` sizeof idiom,
// and a static shape adds constant extents.
// CHECK:       llvm.func @wrap_copy_d2h(!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK:       llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-LABEL: llvm.func @copy_d2h_static(
// CHECK-SAME:    %[[CTX:[^:]+]]: !llvm.ptr,
// CHECK-SAME:    %[[SRC_ALLOC:[^:]+]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[SRC_ALIGN:[^:]+]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[SRC_OFFSET:[^:]+]]: i64,
// CHECK-SAME:    %[[SRC_SIZE0:[^:]+]]: i64,
// CHECK-SAME:    %[[SRC_SIZE1:[^:]+]]: i64,
// CHECK-SAME:    %[[SRC_STRIDE0:[^:]+]]: i64,
// CHECK-SAME:    %[[SRC_STRIDE1:[^:]+]]: i64) {
// CHECK-NEXT:    %[[SRC_0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_1:.+]] = llvm.insertvalue %[[SRC_ALLOC]], %[[SRC_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_2:.+]] = llvm.insertvalue %[[SRC_ALIGN]], %[[SRC_1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_3:.+]] = llvm.insertvalue %[[SRC_OFFSET]], %[[SRC_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_4:.+]] = llvm.insertvalue %[[SRC_SIZE0]], %[[SRC_3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_5:.+]] = llvm.insertvalue %[[SRC_STRIDE0]], %[[SRC_4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_6:.+]] = llvm.insertvalue %[[SRC_SIZE1]], %[[SRC_5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_DESC:.+]] = llvm.insertvalue %[[SRC_STRIDE1]], %[[SRC_6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[DIM0:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[DIM1:.+]] = llvm.mlir.constant(8 : index) : i64
// CHECK-NEXT:    %[[ONE:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[NUM_ELEMS:.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK-NEXT:    %[[MALLOC_NULL:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[MALLOC_GEP:.+]] = llvm.getelementptr %[[MALLOC_NULL]]{{\[}}%[[NUM_ELEMS]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[MALLOC_BYTES:.+]] = llvm.ptrtoint %[[MALLOC_GEP]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[DST_BUF:.+]] = llvm.call @malloc(%[[MALLOC_BYTES]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[DST_0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[DST_1:.+]] = llvm.insertvalue %[[DST_BUF]], %[[DST_0]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[DST_2:.+]] = llvm.insertvalue %[[DST_BUF]], %[[DST_1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[ZERO:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[DST_3:.+]] = llvm.insertvalue %[[ZERO]], %[[DST_2]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[DST_4:.+]] = llvm.insertvalue %[[DIM0]], %[[DST_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[DST_5:.+]] = llvm.insertvalue %[[DIM1]], %[[DST_4]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[DST_6:.+]] = llvm.insertvalue %[[DIM1]], %[[DST_5]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[DST_DESC:.+]] = llvm.insertvalue %[[ONE]], %[[DST_6]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[GEP:.+]] = llvm.getelementptr %[[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[ELEM:.+]] = llvm.ptrtoint %[[GEP]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[EXT0:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[EXT1:.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %[[PARTIAL:.+]] = llvm.mul %[[ELEM]], %[[EXT0]] : i64
// CHECK-NEXT:    %[[BYTES:.+]] = llvm.mul %[[PARTIAL]], %[[EXT1]] : i64
// CHECK-NEXT:    %[[DST_PTR:.+]] = llvm.extractvalue %[[DST_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[SRC_PTR:.+]] = llvm.extractvalue %[[SRC_DESC]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %{{.+}} = llvm.call @wrap_copy_d2h(%[[CTX]], %[[DST_PTR]], %[[SRC_PTR]], %[[BYTES]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @copy_d2h_static(
    %ctx: !hipsr.context,
    %src: memref<4x8xi64, #hipsr.mem<device>>) {
  %init = memref.alloc() : memref<4x8xi64, #hipsr.mem<host>>
  hipsr.copy_d2h(%ctx)
      ins(%src : memref<4x8xi64, #hipsr.mem<device>>)
      outs(%init : memref<4x8xi64, #hipsr.mem<host>>)
  return
}

// -----

// A dynamic extent is read out of the destination descriptor instead of folded,
// so the byte count is computed at run time.
// CHECK:       llvm.func @wrap_copy_d2h(!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK:       llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-LABEL: llvm.func @copy_d2h_dynamic(
// CHECK-SAME:    %[[CTX:[^:]+]]: !llvm.ptr,
// CHECK-SAME:    %[[SRC_ALLOC:[^:]+]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[SRC_ALIGN:[^:]+]]: !llvm.ptr<1>,
// CHECK-SAME:    %[[SRC_OFFSET:[^:]+]]: i64,
// CHECK-SAME:    %[[SRC_SIZE0:[^:]+]]: i64,
// CHECK-SAME:    %[[SRC_STRIDE0:[^:]+]]: i64) {
// CHECK-NEXT:    %[[SRC_0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[SRC_1:.+]] = llvm.insertvalue %[[SRC_ALLOC]], %[[SRC_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[SRC_2:.+]] = llvm.insertvalue %[[SRC_ALIGN]], %[[SRC_1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[SRC_3:.+]] = llvm.insertvalue %[[SRC_OFFSET]], %[[SRC_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[SRC_4:.+]] = llvm.insertvalue %[[SRC_SIZE0]], %[[SRC_3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[SRC_DESC:.+]] = llvm.insertvalue %[[SRC_STRIDE0]], %[[SRC_4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[AXIS:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[EXTENT:.+]] = llvm.extractvalue %[[SRC_DESC]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[ONE:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[MALLOC_NULL:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[MALLOC_GEP:.+]] = llvm.getelementptr %[[MALLOC_NULL]]{{\[}}%[[EXTENT]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-NEXT:    %[[MALLOC_BYTES:.+]] = llvm.ptrtoint %[[MALLOC_GEP]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[DST_BUF:.+]] = llvm.call @malloc(%[[MALLOC_BYTES]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[DST_0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[DST_1:.+]] = llvm.insertvalue %[[DST_BUF]], %[[DST_0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[DST_2:.+]] = llvm.insertvalue %[[DST_BUF]], %[[DST_1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[ZERO:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[DST_3:.+]] = llvm.insertvalue %[[ZERO]], %[[DST_2]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[DST_4:.+]] = llvm.insertvalue %[[EXTENT]], %[[DST_3]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[DST_DESC:.+]] = llvm.insertvalue %[[ONE]], %[[DST_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[GEP:.+]] = llvm.getelementptr %[[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, f16
// CHECK-NEXT:    %[[ELEM:.+]] = llvm.ptrtoint %[[GEP]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[DIM0:.+]] = llvm.extractvalue %[[DST_DESC]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[BYTES:.+]] = llvm.mul %[[ELEM]], %[[DIM0]] : i64
// CHECK-NEXT:    %[[DST_PTR:.+]] = llvm.extractvalue %[[DST_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[SRC_PTR:.+]] = llvm.extractvalue %[[SRC_DESC]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %{{.+}} = llvm.call @wrap_copy_d2h(%[[CTX]], %[[DST_PTR]], %[[SRC_PTR]], %[[BYTES]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @copy_d2h_dynamic(
    %ctx: !hipsr.context,
    %src: memref<?xf16, #hipsr.mem<device>>) {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %src, %c0 : memref<?xf16, #hipsr.mem<device>>
  %init = memref.alloc(%d0) : memref<?xf16, #hipsr.mem<host>>
  hipsr.copy_d2h(%ctx)
      ins(%src : memref<?xf16, #hipsr.mem<device>>)
      outs(%init : memref<?xf16, #hipsr.mem<host>>)
  return
}
