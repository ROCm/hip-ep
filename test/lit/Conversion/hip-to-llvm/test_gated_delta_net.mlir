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

  // Projection and gate widths, conv width and columns are constants; the
  // row and state-batch extents arrive as element counts.
  func.func @gdn_conv_norm(
      %ctx: !hip.context, %proj: memref<?x?x768xf16, 1>,
      %w: memref<512x1x4xf16, 1>, %bias: memref<512xf16, 1>,
      %past: memref<?x512x3xf16, 1>, %gates: memref<?x?x6xf16, 1>,
      %S: memref<?x2x128x128xf32, 1>, %A: memref<2xf32, 1>,
      %dt: memref<2xf32, 1>, %nw: memref<128xf16, 1>,
      %out: memref<?x?x256xf16, 1>, %present: memref<?x512x3xf16, 1>,
      %final: memref<?x2x128x128xf32, 1>, %cin: memref<?x?x512xf16, 1>,
      %cout: memref<?x?x512xf16, 1>, %d: memref<?x?x2xf32, 1>,
      %b: memref<?x?x2xf32, 1>, %o: memref<?x?x2x128xf16, 1>,
      %z: memref<?x?x256xf16, 1>) {
    hip.gated_delta_net_conv_norm(%ctx)
        ins(%proj, %w, %bias, %past, %gates, %S, %A, %dt, %nw :
            memref<?x?x768xf16, 1>, memref<512x1x4xf16, 1>,
            memref<512xf16, 1>, memref<?x512x3xf16, 1>,
            memref<?x?x6xf16, 1>, memref<?x2x128x128xf32, 1>,
            memref<2xf32, 1>, memref<2xf32, 1>, memref<128xf16, 1>)
        outs(%out, %present, %final, %cin, %cout, %d, %b, %o, %z :
             memref<?x?x256xf16, 1>, memref<?x512x3xf16, 1>,
             memref<?x2x128x128xf32, 1>, memref<?x?x512xf16, 1>,
             memref<?x?x512xf16, 1>, memref<?x?x2xf32, 1>,
             memref<?x?x2xf32, 1>, memref<?x?x2x128xf16, 1>,
             memref<?x?x256xf16, 1>)
        {beta_column = 3 : i64, conv_channels = 512 : i64,
         decay_column = 1 : i64, epsilon = 9.99999997E-7 : f32,
         z_column = 512 : i64}
    // CHECK-LABEL: llvm.func @gdn_conv_norm
    // CHECK: %[[GSTRIDE:.*]] = llvm.mlir.constant(6 : i64) : i64
    // CHECK: %[[C:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %[[K:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK: %[[DC:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[BC:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %[[ZC:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %[[HQK:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[HV:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
    // CHECK: llvm.call @wrap_gated_delta_net_conv_norm(%{{.*}}, %[[GSTRIDE]], %[[C]], %[[K]], %[[DC]], %[[BC]], %[[ZC]], %[[HQK]], %[[HV]], %{{.*}}, %{{.*}}, %{{.*}}, %[[EPS]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32) -> i32
    return
  }
}
