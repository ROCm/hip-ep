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
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// Verify 14 parameters:
// - 4 pointers: state, A, B, output
// - 9 i64: M=128, N=1024, K=4096, batch_count=1, elem_size=2,
//   b_batch_stride=0, a_batch_stride=0, transA, transB
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
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- transB=1 (fused Q @ K^T) ----------------------------------------------
module {
  func.func @test_matmul_trans_b(%ctx: !hip.context,
                                  %A: memref<2x3x8x16xf16, 1>,
                                  %B: memref<2x3x8x16xf16, 1>,
                                  %output: memref<2x3x8x8xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<2x3x8x16xf16, 1>, memref<2x3x8x16xf16, 1>)
        outs(%output : memref<2x3x8x8xf16, 1>)
        {transB = 1 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_trans_b
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Broadcast A [1, 1, M, K] against per-batch B [1, H, K, N] -------------
// One activation multiplied by H weight matrices: the batch count comes from
// B (4), A's stride is 0 and B's is K*N. Taking the count from A alone ran a
// single GEMM and left batches 1+ of the output unwritten.

module {
  func.func @test_matmul_broadcast_a(%ctx: !hip.context,
                                      %A: memref<1x1x16x64xf16, 1>,
                                      %B: memref<1x4x64x32xf16, 1>,
                                      %output: memref<1x4x16x32xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<1x1x16x64xf16, 1>, memref<1x4x64x32xf16, 1>)
        outs(%output : memref<1x4x16x32xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_broadcast_a
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.select
// CHECK-DAG: %[[BATCH:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG: %[[BSTRIDE:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK-DAG: %[[ASTRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[BATCH]], %{{.*}}, %[[BSTRIDE]], %[[ASTRIDE]], %{{.*}}, %{{.*}})

// ----- Dynamic leading dims on both sides ------------------------------------
// The count is the larger runtime product and each stride is 0 when its
// operand's product is 1.

module {
  func.func @test_matmul_dynamic_both(%ctx: !hip.context,
                                       %A: memref<?x16x64xf16, 1>,
                                       %B: memref<?x64x32xf16, 1>,
                                       %output: memref<?x16x32xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<?x16x64xf16, 1>, memref<?x64x32xf16, 1>)
        outs(%output : memref<?x16x32xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_dynamic_both
// CHECK: %[[A_LARGER:.*]] = llvm.icmp "sge" %[[AP:.*]], %[[BP:.*]] : i64
// CHECK: %[[DBATCH:.*]] = llvm.select %[[A_LARGER]], %[[AP]], %[[BP]] : i1, i64
// CHECK: llvm.icmp "sle" %[[BP]], %{{.*}} : i64
// CHECK: %[[DBSTRIDE:.*]] = llvm.select
// CHECK: llvm.icmp "sle" %[[AP]], %{{.*}} : i64
// CHECK: %[[DASTRIDE:.*]] = llvm.select
// CHECK: llvm.call @wrap_hipblasLtMatmul(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[DBATCH]], %{{.*}}, %[[DBSTRIDE]], %[[DASTRIDE]], %{{.*}}, %{{.*}})
