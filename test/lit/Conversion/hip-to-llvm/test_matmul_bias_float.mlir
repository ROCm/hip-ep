// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A float hip.matmul with a bias lowers to wrap_gemm (hipBLASLt bias
// epilogue), the same call as hip.gemm with alpha = beta = 1. A rank-3 A folds
// its leading dims into the rows. Its op state is a GemmState.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init \
// RUN:   | FileCheck %s --check-prefix=STATE

// A is [2, 64, 128]: M = 2 * 64. C is [N], so cDim0 = 1, cDim1 = N = 32.
// CHECK-LABEL: llvm.func @float_matmul_bias(
// CHECK-NOT:   @wrap_hipblasLtMatmul
// CHECK:       %[[ROWS:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK:       %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK:       llvm.call @wrap_gemm(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ROWS]], %{{.*}}, %{{.*}}, %[[ONE]], %{{.*}}, {{.*}})
func.func @float_matmul_bias(%ctx: !hip.context, %A: memref<2x64x128xf16, 1>,
                             %B: memref<128x32xf16, 1>, %C: memref<32xf16, 1>,
                             %Y: memref<2x64x32xf16, 1>) {
  hip.matmul(%ctx) ins(%A, %B, %C : memref<2x64x128xf16, 1>, memref<128x32xf16, 1>, memref<32xf16, 1>)
      outs(%Y : memref<2x64x32xf16, 1>) {quant_types = [f16, f16, f16, f16]}
  return
}

// STATE: hipdnn_ep_op_state_construct_gemm
// STATE-NOT: hipdnn_ep_op_state_construct_matmul
