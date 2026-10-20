// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.dynamic_quant_i8 lowers to wrap_dynamic_quant_i8(state, A, q,
// scale, rows, K, block_size) with rows the product of the leading dims.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @dynamic_quant
  // CHECK: %[[K:.*]] = llvm.mlir.constant(2048 : i64) : i64
  // CHECK: %[[BS:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: llvm.call @wrap_dynamic_quant_i8(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[K]], %[[BS]])
  // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func @dynamic_quant(%ctx: !hip.context,
                           %A: memref<1x?x2048xf16, 1>,
                           %Q: memref<1x?x2048xi8, 1>,
                           %S: memref<1x?x64xf32, 1>) {
    hip.dynamic_quant_i8(%ctx) ins(%A : memref<1x?x2048xf16, 1>)
        outs(%Q, %S : memref<1x?x2048xi8, 1>, memref<1x?x64xf32, 1>)
        {block_size = 32 : i64}
    return
  }
}
