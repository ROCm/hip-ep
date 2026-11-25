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
    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @wrap_gated_delta_net(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NULL]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, i64) -> i32
    return
  }

  func.func @gdn_varlen(
      %ctx: !hip.context,
      %qkv: memref<?x10240xf16, 1>,
      %decay: memref<?x48xf32, 1>,
      %beta: memref<?x48xf32, 1>,
      %cu: memref<?xi32, 1>,
      %state: memref<?x48x128x128xf32, 1>,
      %a_log: memref<48xf32, 1>,
      %dt_bias: memref<48xf32, 1>,
      %out: memref<?x48x128xf16, 1>,
      %final: memref<?x48x128x128xf32, 1>) {
    hip.gated_delta_net(%ctx)
        ins(%qkv, %decay, %beta, %state, %a_log, %dt_bias :
            memref<?x10240xf16, 1>, memref<?x48xf32, 1>, memref<?x48xf32, 1>,
            memref<?x48x128x128xf32, 1>, memref<48xf32, 1>, memref<48xf32, 1>)
        cu(%cu : memref<?xi32, 1>)
        outs(%out, %final : memref<?x48x128xf16, 1>,
                            memref<?x48x128x128xf32, 1>)
        {scale = 0.0 : f32}
    // CHECK-LABEL: llvm.func @gdn_varlen
    // CHECK-NOT: llvm.mlir.zero
    // CHECK: llvm.call @wrap_gated_delta_net(%{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, i64) -> i32
    return
  }}
