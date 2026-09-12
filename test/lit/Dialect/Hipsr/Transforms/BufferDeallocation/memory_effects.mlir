// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --buffer-deallocation-pipeline | FileCheck %s

// Tensor operands are values, so a DPS op reports no memory effects.
// Nothing reads the result, so the op is deleted.
// CHECK-LABEL:   func.func @unread_tensor_result_is_dead(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @unread_tensor_result_is_dead(
    %ctx: !hipsr.context, %in: tensor<4xf16, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<4xf32, #hipsr.mem<device>>
  %result = hipsr.cast(%ctx)
      ins(%in : tensor<4xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4xf32, #hipsr.mem<device>>)
      : tensor<4xf32, #hipsr.mem<device>>
  return
}

// A memref destination is real memory, so the write keeps the op.
// Dealloc then frees both allocations after that write.
// CHECK-LABEL:   func.func @memref_destination_write_survives(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context) {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.alloc() : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.cast(%[[ARG0]]) ins(%[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>) outs(%[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.dealloc %[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @memref_destination_write_survives(%ctx: !hipsr.context) {
  %in = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
  %out = memref.alloc() : memref<4xf32, #hipsr.mem<device>>
  hipsr.cast(%ctx)
      ins(%in : memref<4xf16, #hipsr.mem<device>>)
      outs(%out : memref<4xf32, #hipsr.mem<device>>)
  return
}

// An unread allocation is dead, including alloc_output.
// CHECK-LABEL:   func.func @unread_alloc_output_is_dead(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context) {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @unread_alloc_output_is_dead(%ctx: !hipsr.context) {
  %out = hipsr.alloc_output(%ctx) {out_idx = 0 : i64} : memref<4xf32, #hipsr.mem<device>>
  return
}

// A dest write keeps alloc_output even when it is not returned.
// Dealloc then frees both allocations after that write.
// CHECK-LABEL:   func.func @alloc_output_write_not_returned(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context) {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VAL_1:.*]] = hipsr.alloc_output(%[[ARG0]]) {out_idx = 0 : i64} : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.cast(%[[ARG0]]) ins(%[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>) outs(%[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.dealloc %[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @alloc_output_write_not_returned(%ctx: !hipsr.context) {
  %in = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
  %out = hipsr.alloc_output(%ctx) {out_idx = 0 : i64} : memref<4xf32, #hipsr.mem<device>>
  hipsr.cast(%ctx)
      ins(%in : memref<4xf16, #hipsr.mem<device>>)
      outs(%out : memref<4xf32, #hipsr.mem<device>>)
  return
}

// A returned alloc_output is owned by the caller, so dealloc does not free it
// and does not clone it.
// CHECK-LABEL:   func.func @returned_alloc_output(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context) -> memref<4xf32, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VAL_1:.*]] = hipsr.alloc_output(%[[ARG0]]) {out_idx = 0 : i64} : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.cast(%[[ARG0]]) ins(%[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>) outs(%[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.dealloc %[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      return %[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:    }
func.func @returned_alloc_output(%ctx: !hipsr.context) -> memref<4xf32, #hipsr.mem<device>> {
  %in = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
  %out = hipsr.alloc_output(%ctx) {out_idx = 0 : i64} : memref<4xf32, #hipsr.mem<device>>
  hipsr.cast(%ctx)
      ins(%in : memref<4xf16, #hipsr.mem<device>>)
      outs(%out : memref<4xf32, #hipsr.mem<device>>)
  return %out : memref<4xf32, #hipsr.mem<device>>
}
