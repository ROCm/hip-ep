// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -assign-op-state-slots -generate-op-state-init | FileCheck %s

// hipsr.matmul implements OpStateOpInterface, so each instance gets a slot
// and a construct call. The runtime symbol is (state, slot) -> i8.

// CHECK-LABEL:   module attributes {hipdnn.num_op_state_slots = 2 : i32} {
// CHECK-NEXT:      llvm.func @hipdnn_ep_op_state_construct_matmul(!llvm.ptr, i32) -> i8
// CHECK-NEXT:      llvm.func @hipdnn_ep_op_states_alloc(!llvm.ptr, i64) -> i8
// CHECK-NEXT:      func.func @two_matmuls(%[[CTX:.*]]: !hipsr.context, %[[A:.*]]: memref<2x3xf16, #hipsr.mem<device>>, %[[B:.*]]: memref<3x4xf16, #hipsr.mem<device>>, %[[C:.*]]: memref<4x5xf16, #hipsr.mem<device>>, %[[AB:.*]]: memref<2x4xf16, #hipsr.mem<device>>, %[[ABC:.*]]: memref<2x5xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:        hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : memref<2x3xf16, #hipsr.mem<device>>, memref<3x4xf16, #hipsr.mem<device>>) outs(%[[AB]] : memref<2x4xf16, #hipsr.mem<device>>) {hip.op_state_slot = 0 : i32}
// CHECK-NEXT:        hipsr.matmul(%[[CTX]]) ins(%[[AB]], %[[C]] : memref<2x4xf16, #hipsr.mem<device>>, memref<4x5xf16, #hipsr.mem<device>>) outs(%[[ABC]] : memref<2x5xf16, #hipsr.mem<device>>) {hip.op_state_slot = 1 : i32}
// CHECK-NEXT:        return
// CHECK-NEXT:      }
// CHECK-NEXT:      llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32 {
// CHECK-NEXT:        %[[N:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:        %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]]) : (!llvm.ptr, i64) -> i8
// CHECK-NEXT:        %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:        %[[FAILED:.*]] = llvm.icmp "eq" %[[OK]], %[[ZERO_I8]] : i8
// CHECK-NEXT:        %[[SUCCESS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:        %[[FAILURE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:        llvm.cond_br %[[FAILED]], ^[[FAIL:bb2]], ^[[CONSTRUCT:bb1]]
// CHECK-NEXT:      ^[[CONSTRUCT]]:
// CHECK-NEXT:        %[[SLOT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:        llvm.call @hipdnn_ep_op_state_construct_matmul(%[[STATE]], %[[SLOT0]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT:        %[[SLOT1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:        llvm.call @hipdnn_ep_op_state_construct_matmul(%[[STATE]], %[[SLOT1]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT:        llvm.return %[[SUCCESS]] : i32
// CHECK-NEXT:      ^[[FAIL]]:
// CHECK-NEXT:        llvm.return %[[FAILURE]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:    }
func.func @two_matmuls(
    %ctx: !hipsr.context,
    %a: memref<2x3xf16, #hipsr.mem<device>>,
    %b: memref<3x4xf16, #hipsr.mem<device>>,
    %c: memref<4x5xf16, #hipsr.mem<device>>,
    %ab: memref<2x4xf16, #hipsr.mem<device>>,
    %abc: memref<2x5xf16, #hipsr.mem<device>>) {
  hipsr.matmul(%ctx) ins(%a, %b : memref<2x3xf16, #hipsr.mem<device>>,
                                  memref<3x4xf16, #hipsr.mem<device>>)
              outs(%ab : memref<2x4xf16, #hipsr.mem<device>>)
  hipsr.matmul(%ctx) ins(%ab, %c : memref<2x4xf16, #hipsr.mem<device>>,
                                   memref<4x5xf16, #hipsr.mem<device>>)
              outs(%abc : memref<2x5xf16, #hipsr.mem<device>>)
  return
}
