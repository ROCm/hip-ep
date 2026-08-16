// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// `wrap_hipblasLtMatmul` takes 16 arguments:
//   - 4 pointers: state, A, B, output
//   - 1 i1: right-aligned per-axis batch/output validity
//   - 10 i64: M, N, K_a, K_b, output batch count, element size,
//            A/B operand batch counts, A/B strides
//   - 1 i32: op_state_slot (-1 here; --assign-op-state-slots is not in this RUN)
//
// Each operand's stride is 0 when it holds a single matrix broadcast across the
// output batches, and the matrix size when it holds one matrix per output batch.
// The two strides are the last arguments, so each case anchors on the
// `elem_size` constant (2 for f16) that immediately precedes them.

// ----- Both operands hold a single matrix ------------------------------------
// A's leading dim is statically 1 and B is rank 2, so both strides fold to a
// compile-time 0. Per-axis runtime validation is still emitted independently.

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
// CHECK-NOT: @wrap_hipblasLtMatmul(
// CHECK: %[[ELEM0:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A0_COUNT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[B0_COUNT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[A0_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[B0_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM0]], %[[A0_COUNT]], %[[B0_COUNT]], %[[A0_STRIDE]], %[[B0_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

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
// CHECK-NOT: @wrap_hipblasLtMatmul(
// CHECK: %[[ELEM1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A1_COUNT:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[B1_COUNT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[A1_STRIDE:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: %[[B1_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM1]], %[[A1_COUNT]], %[[B1_COUNT]], %[[A1_STRIDE]], %[[B1_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

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
// CHECK-NOT: @wrap_hipblasLtMatmul(
// CHECK: %[[ELEM2:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A2_COUNT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[B2_COUNT:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A2_STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[B2_STRIDE:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM2]], %[[A2_COUNT]], %[[B2_COUNT]], %[[A2_STRIDE]], %[[B2_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Single-axis dynamic batch on both operands ----------------------------
// With one batch axis, each operand necessarily contains either one matrix or
// one matrix per output batch for every valid broadcast. Runtime stride
// selection can therefore represent both operands without extra metadata.

module {
  func.func @test_matmul_dynamic_batch(%ctx: !hip.context,
                                       %A: memref<?x4x16xf16, 1>,
                                       %B: memref<?x16x32xf16, 1>,
                                       %output: memref<?x4x32xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<?x4x16xf16, 1>, memref<?x16x32xf16, 1>)
        outs(%output : memref<?x4x32xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_dynamic_batch
// CHECK: %[[ELEM3:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[A3_COUNT:.*]] = llvm.mul
// CHECK: %[[B3_COUNT:.*]] = llvm.mul
// CHECK: %[[A3_CMP:.*]] = llvm.icmp "eq" %[[A3_COUNT]],
// CHECK: %[[A3_STRIDE:.*]] = llvm.select %[[A3_CMP]]
// CHECK: %[[B3_CMP:.*]] = llvm.icmp "eq" %[[B3_COUNT]],
// CHECK: %[[B3_STRIDE:.*]] = llvm.select %[[B3_CMP]]
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[ELEM3]], %[[A3_COUNT]], %[[B3_COUNT]], %[[A3_STRIDE]], %[[B3_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Dynamic K from both operands ------------------------------------------
// The ABI receives A[-1] and B[-2] independently. B's matrix stride must use
// B's K, not A's K, so a runtime mismatch cannot make the lowering address B
// with an unrelated extent before the wrapper rejects it.

module {
  func.func @test_matmul_dynamic_k(%ctx: !hip.context,
                                    %A: memref<2x?x?xf16, 1>,
                                    %B: memref<2x?x?xf16, 1>,
                                    %output: memref<2x?x?xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<2x?x?xf16, 1>, memref<2x?x?xf16, 1>)
        outs(%output : memref<2x?x?xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_dynamic_k
// CHECK: %[[M:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK: %[[KA:.*]] = llvm.extractvalue {{.*}}[3, 2]
// CHECK: %[[KB:.*]] = llvm.extractvalue {{.*}}[3, 1]
// CHECK: %[[N:.*]] = llvm.extractvalue {{.*}}[3, 2]
// CHECK: %[[B_STRIDE:.*]] = llvm.mul %[[KB]], %[[N]] : i64
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[M]], %[[N]], %[[KA]], %[[KB]], {{.*}}, %[[B_STRIDE]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// ----- Gemma-shaped multi-axis dynamic batches ------------------------------
// Both operand counts multiply both dynamic batch extents. Each stride selects
// matrix-size only when that count equals the output count; a partial runtime
// broadcast therefore reaches the runtime wrapper with an intermediate count
// and is rejected before BLAS.

module {
  func.func @test_matmul_gemma_dynamic_batches(
      %ctx: !hip.context,
      %A: memref<?x?x4x16xf16, 1>,
      %B: memref<?x?x16x32xf16, 1>,
      %output: memref<?x?x4x32xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<?x?x4x16xf16, 1>,
             memref<?x?x16x32xf16, 1>)
        outs(%output : memref<?x?x4x32xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_gemma_dynamic_batches
// CHECK: %[[A0_EQ_B0:.*]] = llvm.icmp "eq"
// CHECK: %[[A0_ONE:.*]] = llvm.icmp "eq"
// CHECK: %[[B0_ONE:.*]] = llvm.icmp "eq"
// CHECK: %[[EITHER0:.*]] = llvm.or %[[A0_ONE]], %[[B0_ONE]]
// CHECK: %[[COMPAT0:.*]] = llvm.or %[[A0_EQ_B0]], %[[EITHER0]]
// CHECK: %[[EXPECTED0:.*]] = llvm.select %[[A0_ONE]]
// CHECK: %[[OUT_MATCH0:.*]] = llvm.icmp "eq" {{.*}}, %[[EXPECTED0]]
// CHECK: %[[AXIS0:.*]] = llvm.and %[[COMPAT0]], %[[OUT_MATCH0]]
// CHECK: %[[VALID0:.*]] = llvm.and {{.*}}, %[[AXIS0]]
// CHECK: %[[A1_EQ_B1:.*]] = llvm.icmp "eq"
// CHECK: %[[A1_ONE:.*]] = llvm.icmp "eq"
// CHECK: %[[B1_ONE:.*]] = llvm.icmp "eq"
// CHECK: %[[EITHER1:.*]] = llvm.or %[[A1_ONE]], %[[B1_ONE]]
// CHECK: %[[COMPAT1:.*]] = llvm.or %[[A1_EQ_B1]], %[[EITHER1]]
// CHECK: %[[EXPECTED1:.*]] = llvm.select %[[A1_ONE]]
// CHECK: %[[OUT_MATCH1:.*]] = llvm.icmp "eq" {{.*}}, %[[EXPECTED1]]
// CHECK: %[[AXIS1:.*]] = llvm.and %[[COMPAT1]], %[[OUT_MATCH1]]
// CHECK: %[[BATCH_VALID:.*]] = llvm.and %[[VALID0]], %[[AXIS1]]
// CHECK: %[[OUT_BATCH0:.*]] = llvm.mul
// CHECK: %[[OUT_BATCH:.*]] = llvm.mul %[[OUT_BATCH0]],
// CHECK: %[[A_COUNT0:.*]] = llvm.mul
// CHECK: %[[A_COUNT:.*]] = llvm.mul %[[A_COUNT0]],
// CHECK: %[[B_COUNT0:.*]] = llvm.mul
// CHECK: %[[B_COUNT:.*]] = llvm.mul %[[B_COUNT0]],
// CHECK: %[[A_MATCH:.*]] = llvm.icmp "eq" %[[A_COUNT]], %[[OUT_BATCH]]
// CHECK: %[[A_STRIDE:.*]] = llvm.select %[[A_MATCH]]
// CHECK: %[[B_MATCH:.*]] = llvm.icmp "eq" %[[B_COUNT]], %[[OUT_BATCH]]
// CHECK: %[[B_STRIDE2:.*]] = llvm.select %[[B_MATCH]]
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[BATCH_VALID]], {{.*}}, %[[OUT_BATCH]], {{.*}}, %[[A_COUNT]], %[[B_COUNT]], %[[A_STRIDE]], %[[B_STRIDE2]])
