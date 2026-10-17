// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.matmul_nbits_group lowers to one wrap_matmul_nbits_group call
// with the fixed 4-member ABI: (state, slot, A, a_q, a_scale, count,
// [B_i, scales_i, Y_i, N_i] x 4, M, K, batch_count, bits, block_size,
// elem_size). Unused member slots get null pointers and N = 0, and so does
// a_q / a_scale without a_quant.
// ============================================================================

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_group_two(%ctx: !hip.context,
                            %A: memref<1x1x2048xf16, 1>,
                            %B0: memref<8192x64x16xui8, 1>,
                            %S0: memref<8192x64xf16, 1>,
                            %B1: memref<32x64x16xui8, 1>,
                            %S1: memref<32x64xf16, 1>,
                            %Y0: memref<1x1x8192xf16, 1>,
                            %Y1: memref<1x1x32xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x2048xf16, 1>)
        weights(%B0, %B1 : memref<8192x64x16xui8, 1>, memref<32x64x16xui8, 1>)
        scales(%S0, %S1 : memref<8192x64xf16, 1>, memref<32x64xf16, 1>)
        outs(%Y0, %Y1 : memref<1x1x8192xf16, 1>, memref<1x1x32xf16, 1>)
        {K = 2048 : i64, Ns = array<i64: 8192, 32>, bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_group_two
  // CHECK: %[[AQ:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[AS:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[N0:.*]] = llvm.mlir.constant(8192 : i64) : i64
  // CHECK: %[[N1:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[B2:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[S2:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[Y2:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[N2:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[B3:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[S3:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[Y3:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[N3:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @wrap_matmul_nbits_group(%{{.*}}, %{{.*}}, %{{.*}}, %[[AQ]], %[[AS]], %[[COUNT]],
  // CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %[[N0]],
  // CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %[[N1]],
  // CHECK-SAME: %[[B2]], %[[S2]], %[[Y2]], %[[N2]],
  // CHECK-SAME: %[[B3]], %[[S3]], %[[Y3]], %[[N3]],
  // CHECK-SAME: : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64,
  // CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64,
  // CHECK-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64,
  // CHECK-SAME: i64, i64, i64, i64, i64, i64) -> i32

  // With a_quant the producer's int8 activation and scales are passed.
  func.func @test_group_prequant(%ctx: !hip.context,
                                 %A: memref<1x1x2048xf16, 1>,
                                 %Aq: memref<1x1x2048xi8, 1>,
                                 %As: memref<1x1x64xf32, 1>,
                                 %B0: memref<8192x64x16xui8, 1>,
                                 %S0: memref<8192x64xf16, 1>,
                                 %B1: memref<32x64x16xui8, 1>,
                                 %S1: memref<32x64xf16, 1>,
                                 %Y0: memref<1x1x8192xf16, 1>,
                                 %Y1: memref<1x1x32xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x2048xf16, 1>)
        a_quant(%Aq, %As : memref<1x1x2048xi8, 1>, memref<1x1x64xf32, 1>)
        weights(%B0, %B1 : memref<8192x64x16xui8, 1>, memref<32x64x16xui8, 1>)
        scales(%S0, %S1 : memref<8192x64xf16, 1>, memref<32x64xf16, 1>)
        outs(%Y0, %Y1 : memref<1x1x8192xf16, 1>, memref<1x1x32xf16, 1>)
        {K = 2048 : i64, Ns = array<i64: 8192, 32>, bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_group_prequant
  // CHECK: %[[A:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
  // CHECK: %[[AQ2:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
  // CHECK: %[[AS2:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
  // CHECK: llvm.call @wrap_matmul_nbits_group(%{{.*}}, %{{.*}}, %[[A]], %[[AQ2]], %[[AS2]],
}
