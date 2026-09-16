// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// Each case checks its whole function. The lowering writes the input and output
// extents into two stack arrays one element at a time, and the call is only
// correct if every store lands at the index it was meant for, so the stores are
// checked in order rather than skipped over.

// The lowering declares the runtime entry point before calling it.
// CHECK-LABEL:   llvm.func @wrap_expand(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32

// A shape operand in host memory reaches the call as its own pointer.
// CHECK-LABEL:   llvm.func @expand_host_shape(
// CHECK-SAME:      %[[CTX:[^,]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG2:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG3:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG4:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG5:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG6:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG7:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG8:[^,]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG9:[^,]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG10:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG11:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG12:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG13:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG14:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG15:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG16:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG17:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG18:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG19:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG20:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG21:[^,]*]]: i64) {
// CHECK-NEXT:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG13]], %[[MLIR_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG14]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG15]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[ARG16]], %[[INSERTVALUE_2]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[ARG19]], %[[INSERTVALUE_3]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[ARG17]], %[[INSERTVALUE_4]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_6:.*]] = llvm.insertvalue %[[ARG20]], %[[INSERTVALUE_5]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_7:.*]] = llvm.insertvalue %[[ARG18]], %[[INSERTVALUE_6]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_8:.*]] = llvm.insertvalue %[[ARG21]], %[[INSERTVALUE_7]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[MLIR_1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_9:.*]] = llvm.insertvalue %[[ARG8]], %[[MLIR_1]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_10:.*]] = llvm.insertvalue %[[ARG9]], %[[INSERTVALUE_9]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_11:.*]] = llvm.insertvalue %[[ARG10]], %[[INSERTVALUE_10]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_12:.*]] = llvm.insertvalue %[[ARG11]], %[[INSERTVALUE_11]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_13:.*]] = llvm.insertvalue %[[ARG12]], %[[INSERTVALUE_12]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[MLIR_2:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_14:.*]] = llvm.insertvalue %[[ARG1]], %[[MLIR_2]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_15:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_14]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_16:.*]] = llvm.insertvalue %[[ARG3]], %[[INSERTVALUE_15]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_17:.*]] = llvm.insertvalue %[[ARG4]], %[[INSERTVALUE_16]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_18:.*]] = llvm.insertvalue %[[ARG6]], %[[INSERTVALUE_17]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_19:.*]] = llvm.insertvalue %[[ARG5]], %[[INSERTVALUE_18]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_20:.*]] = llvm.insertvalue %[[ARG7]], %[[INSERTVALUE_19]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[MLIR_3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[SHAPE_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_13]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:           %[[INPUT_SHAPE:.*]] = llvm.alloca %[[MLIR_3]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:           %[[MLIR_4:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:           %[[MLIR_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[INPUT_SHAPE]]{{\[}}%[[MLIR_5]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_4]], %[[GETELEMENTPTR_0]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[MLIR_6:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[MLIR_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[INPUT_SHAPE]]{{\[}}%[[MLIR_7]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_6]], %[[GETELEMENTPTR_1]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[OUTPUT_SHAPE:.*]] = llvm.alloca %[[MLIR_3]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:           %[[MLIR_8:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:           %[[MLIR_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_2:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_9]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_8]], %[[GETELEMENTPTR_2]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[MLIR_10:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:           %[[MLIR_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_3:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_11]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_10]], %[[GETELEMENTPTR_3]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[MLIR_12:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK-NEXT:           %[[MLIR_13:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_4:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_13]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_12]], %[[GETELEMENTPTR_4]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[INPUT_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_20]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[OUTPUT_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_8]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INPUT_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:           %[[OUTPUT_RANK:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:           %[[DATA_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:           %[[CALL_0:.*]] = llvm.call @wrap_expand(%[[CTX]], %[[INPUT_PTR]], %[[SHAPE_PTR]], %[[OUTPUT_PTR]], %[[INPUT_SHAPE]], %[[INPUT_RANK]], %[[OUTPUT_SHAPE]], %[[OUTPUT_RANK]], %[[DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
func.func @expand_host_shape(
    %ctx: !hipsr.context,
    %input: memref<3x1xf32, #hipsr.mem<device>>,
    %shape: memref<3xi64, #hipsr.mem<host>>,
    %init: memref<2x3x6xf32, #hipsr.mem<device>>) {
  hipsr.expand(%ctx)
      ins(%input, %shape : memref<3x1xf32, #hipsr.mem<device>>,
                            memref<3xi64, #hipsr.mem<host>>)
      outs(%init : memref<2x3x6xf32, #hipsr.mem<device>>)
  return
}

// A shape carried in an attribute leaves nothing to pass, so the shape operand
// is null and the extents are read from the descriptors instead.
// CHECK-LABEL:   llvm.func @expand_shape_attr(
// CHECK-SAME:      %[[CTX:[^,]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG2:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG3:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG4:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG5:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG6:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG7:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG8:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG9:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG10:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG11:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG12:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG13:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG14:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG15:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG16:[^,]*]]: i64) {
// CHECK-NEXT:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG8]], %[[MLIR_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG9]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG10]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[ARG11]], %[[INSERTVALUE_2]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[ARG14]], %[[INSERTVALUE_3]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[ARG12]], %[[INSERTVALUE_4]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_6:.*]] = llvm.insertvalue %[[ARG15]], %[[INSERTVALUE_5]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_7:.*]] = llvm.insertvalue %[[ARG13]], %[[INSERTVALUE_6]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_8:.*]] = llvm.insertvalue %[[ARG16]], %[[INSERTVALUE_7]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[MLIR_1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_9:.*]] = llvm.insertvalue %[[ARG1]], %[[MLIR_1]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_10:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_11:.*]] = llvm.insertvalue %[[ARG3]], %[[INSERTVALUE_10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_12:.*]] = llvm.insertvalue %[[ARG4]], %[[INSERTVALUE_11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_13:.*]] = llvm.insertvalue %[[ARG6]], %[[INSERTVALUE_12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_14:.*]] = llvm.insertvalue %[[ARG5]], %[[INSERTVALUE_13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_15:.*]] = llvm.insertvalue %[[ARG7]], %[[INSERTVALUE_14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[MLIR_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[NULL_SHAPE:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:           %[[INPUT_SHAPE:.*]] = llvm.alloca %[[MLIR_2]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:           %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[INSERTVALUE_15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[MLIR_4:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[INPUT_SHAPE]]{{\[}}%[[MLIR_4]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[EXTRACTVALUE_0]], %[[GETELEMENTPTR_0]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[MLIR_5:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[MLIR_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[INPUT_SHAPE]]{{\[}}%[[MLIR_6]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_5]], %[[GETELEMENTPTR_1]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[OUTPUT_SHAPE:.*]] = llvm.alloca %[[MLIR_2]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:           %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[INSERTVALUE_8]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[MLIR_7:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_2:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_7]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[EXTRACTVALUE_1]], %[[GETELEMENTPTR_2]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[MLIR_8:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:           %[[MLIR_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_3:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_9]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_8]], %[[GETELEMENTPTR_3]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[EXTRACTVALUE_2:.*]] = llvm.extractvalue %[[INSERTVALUE_8]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[MLIR_10:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_4:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_10]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[EXTRACTVALUE_2]], %[[GETELEMENTPTR_4]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[INPUT_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[OUTPUT_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_8]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INPUT_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:           %[[OUTPUT_RANK:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:           %[[DATA_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[CALL_0:.*]] = llvm.call @wrap_expand(%[[CTX]], %[[INPUT_PTR]], %[[NULL_SHAPE]], %[[OUTPUT_PTR]], %[[INPUT_SHAPE]], %[[INPUT_RANK]], %[[OUTPUT_SHAPE]], %[[OUTPUT_RANK]], %[[DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
func.func @expand_shape_attr(
    %ctx: !hipsr.context,
    %input: memref<?x1xf16, #hipsr.mem<device>>,
    %init: memref<?x3x?xf16, #hipsr.mem<device>>) {
  hipsr.expand(%ctx)
      ins(%input : memref<?x1xf16, #hipsr.mem<device>>)
      outs(%init : memref<?x3x?xf16, #hipsr.mem<device>>)
      {shape_attr = array<i64: 2, 3, 6>}
  return
}

// A mask arrives as i1 and takes data type 7, the unsigned-byte slot, because
// the runtime gives every mask element a whole byte.
// CHECK-LABEL:   llvm.func @expand_bool_mask(
// CHECK-SAME:      %[[CTX:[^,]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG2:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG3:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG4:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG5:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG6:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG7:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG8:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG9:[^,]*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[ARG10:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG11:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG12:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG13:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG14:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG15:[^,]*]]: i64,
// CHECK-SAME:      %[[ARG16:[^,]*]]: i64) {
// CHECK-NEXT:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG8]], %[[MLIR_0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG9]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG10]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[ARG11]], %[[INSERTVALUE_2]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[ARG14]], %[[INSERTVALUE_3]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[ARG12]], %[[INSERTVALUE_4]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_6:.*]] = llvm.insertvalue %[[ARG15]], %[[INSERTVALUE_5]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_7:.*]] = llvm.insertvalue %[[ARG13]], %[[INSERTVALUE_6]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_8:.*]] = llvm.insertvalue %[[ARG16]], %[[INSERTVALUE_7]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[MLIR_1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_9:.*]] = llvm.insertvalue %[[ARG1]], %[[MLIR_1]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_10:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_11:.*]] = llvm.insertvalue %[[ARG3]], %[[INSERTVALUE_10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_12:.*]] = llvm.insertvalue %[[ARG4]], %[[INSERTVALUE_11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_13:.*]] = llvm.insertvalue %[[ARG6]], %[[INSERTVALUE_12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_14:.*]] = llvm.insertvalue %[[ARG5]], %[[INSERTVALUE_13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[INSERTVALUE_15:.*]] = llvm.insertvalue %[[ARG7]], %[[INSERTVALUE_14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[MLIR_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[NULL_SHAPE:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:           %[[INPUT_SHAPE:.*]] = llvm.alloca %[[MLIR_2]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:           %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[INSERTVALUE_15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[MLIR_4:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[INPUT_SHAPE]]{{\[}}%[[MLIR_4]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[EXTRACTVALUE_0]], %[[GETELEMENTPTR_0]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[MLIR_5:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:           %[[MLIR_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[INPUT_SHAPE]]{{\[}}%[[MLIR_6]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_5]], %[[GETELEMENTPTR_1]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[OUTPUT_SHAPE:.*]] = llvm.alloca %[[MLIR_2]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:           %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[INSERTVALUE_8]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[MLIR_7:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_2:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_7]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[EXTRACTVALUE_1]], %[[GETELEMENTPTR_2]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[MLIR_8:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:           %[[MLIR_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_3:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_9]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[MLIR_8]], %[[GETELEMENTPTR_3]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[EXTRACTVALUE_2:.*]] = llvm.extractvalue %[[INSERTVALUE_8]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[MLIR_10:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:           %[[GETELEMENTPTR_4:.*]] = llvm.getelementptr %[[OUTPUT_SHAPE]]{{\[}}%[[MLIR_10]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:           llvm.store %[[EXTRACTVALUE_2]], %[[GETELEMENTPTR_4]] : i64, !llvm.ptr
// CHECK-NEXT:           %[[INPUT_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:           %[[OUTPUT_PTR:.*]] = llvm.extractvalue %[[INSERTVALUE_8]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:           %[[INPUT_RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:           %[[OUTPUT_RANK:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:           %[[MASK_DATA_TYPE:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:           %[[CALL_0:.*]] = llvm.call @wrap_expand(%[[CTX]], %[[INPUT_PTR]], %[[NULL_SHAPE]], %[[OUTPUT_PTR]], %[[INPUT_SHAPE]], %[[INPUT_RANK]], %[[OUTPUT_SHAPE]], %[[OUTPUT_RANK]], %[[MASK_DATA_TYPE]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
func.func @expand_bool_mask(
    %ctx: !hipsr.context,
    %input: memref<?x1xi1, #hipsr.mem<device>>,
    %init: memref<?x3x?xi1, #hipsr.mem<device>>) {
  hipsr.expand(%ctx)
      ins(%input : memref<?x1xi1, #hipsr.mem<device>>)
      outs(%init : memref<?x3x?xi1, #hipsr.mem<device>>)
      {shape_attr = array<i64: 2, 3, 6>}
  return
}
