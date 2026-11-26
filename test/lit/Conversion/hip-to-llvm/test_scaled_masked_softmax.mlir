// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// hip.scaled_masked_softmax lowers to one hip_scaled_masked_softmax call:
// five buffers (ctx, input, row ids, col ids, output), then outer, R, C, the
// two id lengths, the element size, and the scale and fill as f32. R and C are
// the last two input axes; outer is the product of the leading ones.

module {
  func.func @static_f16(%ctx: !hip.context, %in: memref<16x96x96xf16, 1>,
                        %rows: memref<96xi64, 1>, %cols: memref<96xi64, 1>,
                        %out: memref<16x96x96xf16, 1>) {
    // CHECK-LABEL: llvm.func @static_f16
    // CHECK: %[[C:.*]] = llvm.extractvalue %[[IN:[0-9]+]][3, 2]
    // CHECK: %[[R:.*]] = llvm.extractvalue %[[IN]][3, 1]
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[D0:.*]] = llvm.extractvalue %[[IN]][3, 0]
    // CHECK: %[[OUTER:.*]] = llvm.mul %[[ONE]], %[[D0]] : i64
    // CHECK: %[[ES:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[SCALE:.*]] = llvm.mlir.constant(1.250000e-01 : f32) : f32
    // CHECK: %[[FILL:.*]] = llvm.mlir.constant(-6.550400e+04 : f32) : f32
    // CHECK: llvm.call @hip_scaled_masked_softmax(%arg0, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OUTER]], %[[R]], %[[C]], %{{.*}}, %{{.*}}, %[[ES]], %[[SCALE]], %[[FILL]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, f32) -> ()
    hip.scaled_masked_softmax(%ctx) ins(%in, %rows, %cols : memref<16x96x96xf16, 1>,
        memref<96xi64, 1>, memref<96xi64, 1>) outs(%out : memref<16x96x96xf16, 1>)
        {fill = -6.550400e+04 : f32, scale = 1.250000e-01 : f32}
    return
  }

  // Rank 4: outer multiplies both leading axes.
  func.func @dynamic_f32(%ctx: !hip.context, %in: memref<2x4x?x?xf32, 1>,
                         %rows: memref<?xi64, 1>, %cols: memref<1xi64, 1>,
                         %out: memref<2x4x?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @dynamic_f32
    // CHECK: %[[C:.*]] = llvm.extractvalue %[[IN:[0-9]+]][3, 3]
    // CHECK: %[[R:.*]] = llvm.extractvalue %[[IN]][3, 2]
    // CHECK: %[[D0:.*]] = llvm.extractvalue %[[IN]][3, 0]
    // CHECK: %[[M0:.*]] = llvm.mul %{{.*}}, %[[D0]] : i64
    // CHECK: %[[D1:.*]] = llvm.extractvalue %[[IN]][3, 1]
    // CHECK: %[[OUTER:.*]] = llvm.mul %[[M0]], %[[D1]] : i64
    // CHECK: %[[ES:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK: llvm.call @hip_scaled_masked_softmax(%arg0, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OUTER]], %[[R]], %[[C]], %{{.*}}, %{{.*}}, %[[ES]], %{{.*}}, %{{.*}})
    hip.scaled_masked_softmax(%ctx) ins(%in, %rows, %cols : memref<2x4x?x?xf32, 1>,
        memref<?xi64, 1>, memref<1xi64, 1>) outs(%out : memref<2x4x?x?xf32, 1>)
        {fill = -1.000000e+09 : f32, scale = 5.000000e-01 : f32}
    return
  }
}
