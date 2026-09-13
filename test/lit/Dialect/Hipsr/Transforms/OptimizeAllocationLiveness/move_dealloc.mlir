// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --optimize-allocation-liveness | FileCheck %s

// The first buffer's last user is the first cast. The pass moves that dealloc
// there. Later deallocs stay after the second cast, their last user.
// CHECK-LABEL:   func.func @dealloc_moves_after_last_user(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context) {
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.alloc() : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.cast(%[[ARG0]]) ins(%[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>) outs(%[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.dealloc %[[VAL_0]] : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.cast(%[[ARG0]]) ins(%[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>) outs(%[[VAL_2]] : memref<4xf16, #hipsr.mem<device>>)
// CHECK-NEXT:      memref.dealloc %[[VAL_2]] : memref<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      memref.dealloc %[[VAL_1]] : memref<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @dealloc_moves_after_last_user(%ctx: !hipsr.context) {
  %in = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
  %mid = memref.alloc() : memref<4xf32, #hipsr.mem<device>>
  hipsr.cast(%ctx)
      ins(%in : memref<4xf16, #hipsr.mem<device>>)
      outs(%mid : memref<4xf32, #hipsr.mem<device>>)
  %out = memref.alloc() : memref<4xf16, #hipsr.mem<device>>
  hipsr.cast(%ctx)
      ins(%mid : memref<4xf32, #hipsr.mem<device>>)
      outs(%out : memref<4xf16, #hipsr.mem<device>>)
  memref.dealloc %in : memref<4xf16, #hipsr.mem<device>>
  memref.dealloc %mid : memref<4xf32, #hipsr.mem<device>>
  memref.dealloc %out : memref<4xf16, #hipsr.mem<device>>
  return
}
