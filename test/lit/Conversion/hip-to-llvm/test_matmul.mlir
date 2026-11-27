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
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// Verify 11 parameters:
// - 4 pointers: state, A, B, output
// - 6 i64: M=128, N=1024, K=4096, batch_count=1, data_type=1 (HALF),
//   b_batch_stride=0
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
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

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
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- i32 matmul, rank-2 B --------------------------------------------------
// batchCount comes from A. B holds one matrix, so b_batch_stride is the
// compile-time constant 0 rather than K*N.

module {
  func.func @test_matmul_i32_rank2_b(%ctx: !hip.context,
                                     %A: memref<2x1x2xi32, 1>,
                                     %B: memref<2x1xi32, 1>,
                                     %output: memref<2x1x1xi32, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<2x1x2xi32, 1>, memref<2x1xi32, 1>)
        outs(%output : memref<2x1x1xi32, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_i32_rank2_b
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.select
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_matmul_i32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[STRIDE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

// ----- bf16 -------------------------------------------------------------------
// bf16 and f16 are both 2 bytes, so the call must carry the element type
// HIPDNN_EP_DATATYPE_BFLOAT16 (2), not a byte size. The CHECK-NEXT chain pins
// the constant to the data_type slot (4th argument from the end).

module {
  func.func @test_matmul_bf16(%ctx: !hip.context,
                               %A: memref<1x128x4096xbf16, 1>,
                               %B: memref<4096x1024xbf16, 1>,
                               %output: memref<1x128x1024xbf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<1x128x4096xbf16, 1>, memref<4096x1024xbf16, 1>)
        outs(%output : memref<1x128x1024xbf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_bf16
// CHECK: %[[DT:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[DT]], %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) :

// ----- Rank-2 A shared by a batched B [batch, K, N] -------------------------
// The batch is B's, which wrap_hipblasLtMatmul cannot take (it reads the
// batch off A and steps A by M*K), so the strided entry point is called with
// A's batch stride 0 and B's K*N. This is a 1x1 convolution as a matmul:
// [Cout, Cin] x [N, Cin, H*W].

module {
  func.func @test_matmul_broadcast_a(%ctx: !hip.context,
                                     %A: memref<64x32xf16, 1>,
                                     %B: memref<8x32x3136xf16, 1>,
                                     %output: memref<8x64x3136xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<64x32xf16, 1>, memref<8x32x3136xf16, 1>)
        outs(%output : memref<8x64x3136xf16, 1>)
    return
  }
}

// The batch is 1 times B's leading size, then K*N; A's stride is the constant
// 0 after the slot (-1: no slot assignment in this RUN) and data_type (1, HALF).
// CHECK-LABEL: llvm.func @test_matmul_broadcast_a
// CHECK-NOT: llvm.call @wrap_hipblasLtMatmul(
// CHECK: %[[BATCH:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK-NEXT: %[[KN:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK-NEXT: llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT: %[[DT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmulStrided({{.*}}, %[[BATCH]], %[[DT]], %[[ZERO]], %[[KN]], %{{[0-9]+}}, %{{[0-9]+}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NOT: llvm.call @wrap_hipblasLtMatmul(

// ----- Rank-2 A with a dynamic-batch B [?, K, N] -----------------------------
// A dynamic leading dim may hold more than one matrix, so the strided entry
// point is taken and the batch is B's runtime extent.

module {
  func.func @test_matmul_broadcast_a_dynamic(%ctx: !hip.context,
                                             %A: memref<64x32xf32, 1>,
                                             %B: memref<?x32x49xf32, 1>,
                                             %output: memref<?x64x49xf32, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<64x32xf32, 1>, memref<?x32x49xf32, 1>)
        outs(%output : memref<?x64x49xf32, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_broadcast_a_dynamic
// CHECK: llvm.call @wrap_hipblasLtMatmulStrided({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NOT: llvm.call @wrap_hipblasLtMatmul(

// ----- Rank-2 A with a leading-one B [1, K, N] -------------------------------
// Neither operand is batched: the plain entry point, batch 1.

module {
  func.func @test_matmul_rank2_a_leading_one_b(%ctx: !hip.context,
                                               %A: memref<64x32xf16, 1>,
                                               %B: memref<1x32x49xf16, 1>,
                                               %output: memref<1x64x49xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<64x32xf16, 1>, memref<1x32x49xf16, 1>)
        outs(%output : memref<1x64x49xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_rank2_a_leading_one_b
// CHECK-NOT: wrap_hipblasLtMatmulStrided
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Leading-one A [1, M, K] with a batched B [batch, K, N] ----------------
// A's leading dims hold one matrix, so this is the shared-A case too: the
// batch is B's. The general path used to take the batch (1) from A and compute
// only the first of B's matrices.

module {
  func.func @test_matmul_leading_one_a_batched_b(%ctx: !hip.context,
                                                 %A: memref<1x64x32xf32, 1>,
                                                 %B: memref<4x32x16xf32, 1>,
                                                 %output: memref<4x64x16xf32, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<1x64x32xf32, 1>, memref<4x32x16xf32, 1>)
        outs(%output : memref<4x64x16xf32, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_leading_one_a_batched_b
// CHECK: llvm.call @wrap_hipblasLtMatmulStrided({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NOT: llvm.call @wrap_hipblasLtMatmul(

// ----- i32 with a shared A keeps the integer kernel --------------------------

module {
  func.func @test_matmul_i32_shared_a(%ctx: !hip.context,
                                      %A: memref<8x4xi32, 1>,
                                      %B: memref<2x4x16xi32, 1>,
                                      %output: memref<2x8x16xi32, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<8x4xi32, 1>, memref<2x4x16xi32, 1>)
        outs(%output : memref<2x8x16xi32, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_i32_shared_a
// CHECK-NOT: wrap_hipblasLtMatmulStrided
// CHECK: llvm.call @wrap_matmul_i32(
