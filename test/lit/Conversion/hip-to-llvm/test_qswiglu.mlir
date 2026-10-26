// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// hip.qswiglu -> wrap_qswiglu, hip.fake_quantize -> wrap_fake_quantize
//
// The qmul coefficients are folded on the host the way hip.qmul's lowering
// folds them: M_silu = 0.5 * 0.25 / 0.125 = 1.0, M_mul = 0.125 * 2.0 / 4.0.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
// ============================================================================

module {

// CHECK-LABEL: llvm.func @qswiglu
// CHECK-DAG:   llvm.mlir.constant(1.000000e+00 : f32)
// CHECK-DAG:   llvm.mlir.constant(6.250000e-02 : f32)
// CHECK-DAG:   llvm.mlir.constant(33 : i64)
// CHECK:       llvm.call @wrap_qswiglu({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, f32, i64, f32, i64, f32, i64, f32, i64, f32, i64, f32, i64) -> i32
  func.func @qswiglu(%ctx: !hip.context, %g: memref<4x64xf32, 1>,
                     %u: memref<4x64xf32, 1>, %y: memref<4x64xf32, 1>) {
    hip.qswiglu(%ctx) ins(%g, %u : memref<4x64xf32, 1>, memref<4x64xf32, 1>)
        outs(%y : memref<4x64xf32, 1>)
        {gate_scale = 5.000000e-01 : f32, gate_zp = 100 : i64,
         sigmoid_scale = 2.500000e-01 : f32, sigmoid_zp = 0 : i64,
         silu_scale = 1.250000e-01 : f32, silu_zp = 7 : i64,
         up_scale = 2.000000e+00 : f32, up_zp = 33 : i64,
         mul_scale = 4.000000e+00 : f32, mul_zp = 50 : i64,
         dq_scale = 4.000000e+00 : f32, dq_zp = 50 : i64}
    return
  }

// CHECK-LABEL: llvm.func @fake_quantize
// CHECK-DAG:   llvm.mlir.constant(32768 : i64)
// CHECK-DAG:   llvm.mlir.constant(65535 : i64)
// CHECK:       llvm.call @wrap_fake_quantize({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, f32, i64, i64, i64) -> i32
  func.func @fake_quantize(%ctx: !hip.context, %x: memref<?x5120xf32, 1>,
                           %y: memref<?x5120xf32, 1>) {
    hip.fake_quantize(%ctx) ins(%x : memref<?x5120xf32, 1>)
        outs(%y : memref<?x5120xf32, 1>)
        {scale = 1.250000e-01 : f32, zero_point = 32768 : i64, qmin = 0 : i64,
         qmax = 65535 : i64}
    return
  }
}
