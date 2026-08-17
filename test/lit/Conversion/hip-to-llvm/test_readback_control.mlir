// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func @readback_control
// CHECK: %[[HOST:.*]] = llvm.alloca
// CHECK: llvm.getelementptr
// CHECK-COUNT-1: llvm.call @hipdnn_ep_readback_control
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
// CHECK: llvm.icmp "eq"
// CHECK-COUNT-3: llvm.load
func.func @readback_control(
    %ctx: !hip.context,
    %i32s: memref<2xi32, strided<[1], offset: ?>>,
    %i64scalar: memref<i64>) -> (i1, i64, i64, i64) {
  %r:4 = hip.readback_control(
      %ctx, %i32s, %i64scalar :
      memref<2xi32, strided<[1], offset: ?>>, memref<i64>)
      -> (i1, i64, i64, i64)
  return %r#0, %r#1, %r#2, %r#3 : i1, i64, i64, i64
}
