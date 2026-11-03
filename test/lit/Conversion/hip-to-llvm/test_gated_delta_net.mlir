// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @gdn_packed(
      %ctx: !hip.context,
      %qkv: memref<?x?x10240xf16, 1>,
      %decay: memref<?x?x48xf32, 1>,
      %beta: memref<?x?x48xf32, 1>,
      %state: memref<?x48x128x128xf32, 1>,
      %a_log: memref<48xf32, 1>,
      %dt_bias: memref<48xf32, 1>,
      %out: memref<?x?x48x128xf16, 1>,
      %final: memref<?x48x128x128xf32, 1>) {
    hip.gated_delta_net(%ctx)
        ins(%qkv, %decay, %beta, %state, %a_log, %dt_bias :
            memref<?x?x10240xf16, 1>, memref<?x?x48xf32, 1>,
            memref<?x?x48xf32, 1>, memref<?x48x128x128xf32, 1>,
            memref<48xf32, 1>, memref<48xf32, 1>)
        outs(%out, %final : memref<?x?x48x128xf16, 1>,
                            memref<?x48x128x128xf32, 1>)
        {scale = 0.0 : f32}
    // CHECK-LABEL: llvm.func @gdn_packed
    // CHECK: %[[B:.*]] = llvm.extractvalue %[[QKV:.*]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[L:.*]] = llvm.extractvalue %[[QKV]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[HQK:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK: %[[HV:.*]] = llvm.mlir.constant(48 : i64) : i64
    // CHECK: %[[DK:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %[[DV:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %[[SCALE:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK: %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.call @wrap_gated_delta_net(%{{.*}}, %[[B]], %[[L]], %[[HQK]], %[[HV]], %[[DK]], %[[DV]], %[[SCALE]], %[[ELEM]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, i64) -> i32
    return
  }
}
