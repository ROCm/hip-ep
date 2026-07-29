// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// ----- Rank-2 broadcast B [K, N] ---------------------------------------------
// b_batch_stride MUST be a constant 0 (no leading dims to multiply).
// `CHECK-NOT` between the label and the call asserts the compile-time
// constant path was taken (no runtime `llvm.icmp` / `llvm.select` over a
// leading-dim product).

module {
  func.func @test_matmul_rank2_b(%ctx: !hip.context,
                                  %A: memref<1x128x4096xf16, 1>,
                                  %B: memref<4096x1024xf16, 1>,
                                  %output: memref<1x128x1024xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<1x128x4096xf16, 1>, memref<4096x1024xf16, 1>)
        outs(%output : memref<1x128x1024xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_rank2_b
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.select
// CHECK-NOT: @wrap_hipblasLtMatmul_v2
// CHECK: %[[A0_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[B0_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[A0_STRIDE]], %[[B0_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
// Verify 12 parameters:
// - 4 pointers: state, A, B, output
// - 7 i64: M, N, K, batch_count, elem_size, A stride, B stride
//   (B is rank-2 [K, N] = broadcast weight → stride = 0, compile-time const)
// - 1 i32: op_state_slot (-1 here — --assign-op-state-slots not run in this RUN)

// ----- Rank-3 leading-one B [1, K, N] ----------------------------------------
// Leading dim is statically 1: still ONE [K, N] matrix in the buffer, so
// b_batch_stride must be 0 (NOT K*N). The compiler folds this at compile
// time when all leading dims are static — verified via CHECK-NOT for icmp /
// select. Encoding this case as "rank > 2 ⇒ per-batch" (the prior `b_batched`
// bool rule) caused hipBLASLt to step K*N elements past the end of the
// weight buffer on batch > 0 and feed garbage into the GEMM.

module {
  func.func @test_matmul_rank3_leading_one_b(%ctx: !hip.context,
                                              %A: memref<2x128x4096xf16, 1>,
                                              %B: memref<1x4096x1024xf16, 1>,
                                              %output: memref<2x128x1024xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<2x128x4096xf16, 1>, memref<1x4096x1024xf16, 1>)
        outs(%output : memref<2x128x1024xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_rank3_leading_one_b
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.select
// CHECK-NOT: @wrap_hipblasLtMatmul_v2
// CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %[[A1_STRIDE:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %[[B1_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[A1_STRIDE]], %[[B1_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Rank-2 broadcast A against rank-3 B -------------------------------
// A contains one matrix and therefore uses A stride 0. B and the output carry
// two batches.

module {
  func.func @test_matmul_rank2_a(%ctx: !hip.context,
                                 %A: memref<128x4096xf16, 1>,
                                 %B: memref<2x4096x1024xf16, 1>,
                                 %output: memref<2x128x1024xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<128x4096xf16, 1>, memref<2x4096x1024xf16, 1>)
        outs(%output : memref<2x128x1024xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_rank2_a
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.select
// CHECK-NOT: @wrap_hipblasLtMatmul_v2
// CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %[[B2_STRIDE:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %[[A2_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[A2_STRIDE]], %[[B2_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
