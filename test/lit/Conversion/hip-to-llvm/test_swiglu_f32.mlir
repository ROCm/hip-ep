// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func @dynamic_activation
// CHECK: llvm.call @wrap_swiglu_f32
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
func.func @dynamic_activation(%ctx: !hip.context,
    %gate: memref<?x?x4096xf16>, %up: memref<?x?x4096xf16>,
    %output: memref<?x?x4096xf16>) {
  hip.swiglu_f32(%ctx) ins(%gate, %up : memref<?x?x4096xf16>, memref<?x?x4096xf16>)
      outs(%output : memref<?x?x4096xf16>)
  return
}

// CHECK-LABEL: llvm.func @broadcast_activation
// CHECK: llvm.call @wrap_swiglu_f32
func.func @broadcast_activation(%ctx: !hip.context,
    %gate: memref<13xf16>, %up: memref<2x3x17x13xf16>,
    %output: memref<2x3x17x13xf16>) {
  hip.swiglu_f32(%ctx) ins(%gate, %up : memref<13xf16>, memref<2x3x17x13xf16>)
      outs(%output : memref<2x3x17x13xf16>)
  return
}
