// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// A static destination provides K without a device readback.
// CHECK-LABEL: llvm.func @static_k
// CHECK: llvm.call @wrap_top_k_static
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr,
func.func @static_k(%ctx: !hip.context, %x: memref<?x?x128xf32>,
                   %k: memref<1xi64>, %values: memref<?x?x6xf32>,
                   %indices: memref<?x?x6xi64>) {
  hip.top_k(%ctx) ins(%x, %k : memref<?x?x128xf32>, memref<1xi64>)
      outs(%values, %indices : memref<?x?x6xf32>, memref<?x?x6xi64>)
      {axis = -1 : i64}
  return
}

// A dynamic destination keeps the existing runtime K path.
// CHECK-LABEL: llvm.func @dynamic_k
// CHECK: llvm.call @wrap_top_k(
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
func.func @dynamic_k(%ctx: !hip.context, %x: memref<?x?x128xf32>,
                    %k: memref<1xi64>, %values: memref<?x?x?xf32>,
                    %indices: memref<?x?x?xi64>) {
  hip.top_k(%ctx) ins(%x, %k : memref<?x?x128xf32>, memref<1xi64>)
      outs(%values, %indices : memref<?x?x?xf32>, memref<?x?x?xi64>)
      {axis = 2 : i64}
  return
}
