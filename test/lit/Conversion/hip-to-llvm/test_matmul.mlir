// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// `wrap_hipblasLtMatmul` takes 12 arguments:
//   - 4 pointers: state, A, B, output
//   - 7 i64: M, N, K, batch_count, elem_size, A stride, B stride
//   - 1 i32: op_state_slot (-1 here; --assign-op-state-slots is not in this RUN)
//
// Each operand's stride is 0 when it holds a single matrix broadcast across the
// output batches, and the matrix size when it holds one matrix per output batch.
// The two strides are the last arguments, so each case anchors on the
// `elem_size` constant (2 for f16) that immediately precedes them.

// ----- Both operands hold a single matrix ------------------------------------
// A's leading dim is statically 1 and B is rank 2, so both strides fold to a
// compile-time 0. `CHECK-NOT` asserts no runtime `llvm.icmp` / `llvm.select`
// over a matrix count was emitted.

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
// CHECK: %[[ELEM0:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A0_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[B0_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM0]], %[[A0_STRIDE]], %[[B0_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Rank-3 leading-one B [1, K, N] ----------------------------------------
// B's leading dim is statically 1, so the buffer still holds ONE [K, N] matrix
// and its stride must be 0, not K*N. Treating "rank > 2" as per-batch made
// hipBLASLt step K*N elements past the end of the weight on batch > 0 and feed
// garbage into the GEMM. A carries two batches and gets stride M*K.

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
// CHECK: %[[ELEM1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A1_STRIDE:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %[[B1_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM1]], %[[A1_STRIDE]], %[[B1_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Rank-2 broadcast A against rank-3 B -----------------------------------
// The mirror image of the case above: A holds one matrix (stride 0) while B and
// the output carry two batches (stride K*N). Independent per-operand strides are
// what let either whole matrix broadcast across the other's batches.

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
// CHECK: %[[ELEM2:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A2_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[B2_STRIDE:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM2]], %[[A2_STRIDE]], %[[B2_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Dynamic batch on both operands ----------------------------------------
// `[?, 8, M, K] @ [?, 8, K, N]` is an ordinary batched matmul: neither operand
// broadcasts, so both are representable and must NOT be rejected. Because the
// leading extents are dynamic, each stride is chosen at runtime by comparing the
// operand's matrix count against the output's batch count -- which also keeps
// the read in bounds if the counts ever disagree.

module {
  func.func @test_matmul_dynamic_batch(%ctx: !hip.context,
                                       %A: memref<?x8x4x16xf16, 1>,
                                       %B: memref<?x8x16x32xf16, 1>,
                                       %output: memref<?x8x4x32xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<?x8x4x16xf16, 1>, memref<?x8x16x32xf16, 1>)
        outs(%output : memref<?x8x4x32xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_dynamic_batch
// CHECK: %[[ELEM3:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A3_CMP:.*]] = llvm.icmp "eq"
// CHECK: %[[A3_STRIDE:.*]] = llvm.select %[[A3_CMP]]
// CHECK: %[[B3_CMP:.*]] = llvm.icmp "eq"
// CHECK: %[[B3_STRIDE:.*]] = llvm.select %[[B3_CMP]]
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM3]], %[[A3_STRIDE]], %[[B3_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
