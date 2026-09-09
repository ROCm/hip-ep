// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: hip.qmatmul -> llvm.call @wrap_qmatmul
//
// One case carries every non-trivial part of the lowering at once:
//   * A16W8 (ui16 / i8 / ui16) -> the three data-type codes are passed
//     independently, so a wider A must not drag B's code along.
//   * transB -> N comes from B's second-to-last extent, K still from A's last.
//   * rank-3 per-batch B -> batch_count from A's leading dim and a
//     b_batch_stride of K*N rather than the broadcast 0.
//   * Asymmetric zero points, with 32768 proving a ui16 zero point survives
//     instead of wrapping to -32768.
//
// The scales are chosen so the folded coefficient is exact in f32 and readable:
//   M = A_scale * B_scale / Y_scale = 0.25 * 0.125 / 0.5 = 0.0625 = 2^-4
// Anything the kernel could recompute from A_scale/B_scale/Y_scale at runtime
// would cost a float divide per output element, so the fold has to land here.
//
// Extents are read back from the descriptors even though they are static here,
// so the checks bind SSA values instead of matching shape constants.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
// ============================================================================

module {
  // Binding every argument lets the final CHECK pin the whole 18-parameter ABI
  // in order, matching wrap_qmatmul in lib/Runtime/hipdnn_ep_runtime.h:
  //   4 ptr : state, A, B, Y
  //   5 i64 : M, N, K, batch_count, b_batch_stride
  //   2 i64 : trans_a, trans_b
  //   3 i64 : a_data_type, b_data_type, y_data_type
  //   1 f32 : M_scale
  //   3 i64 : A_zero_point, B_zero_point, Y_zero_point
  //
  // CHECK-LABEL: llvm.func @test_qmatmul(
  // CHECK:      %[[MSCALE:.*]] = llvm.mlir.constant(6.250000e-02 : f32) : f32
  // CHECK-NEXT: %[[M:.*]] = llvm.extractvalue %[[ADESC:.*]][3, 1]
  // CHECK-NEXT: %[[K:.*]] = llvm.extractvalue %[[ADESC]][3, 2]
  // CHECK-NEXT: %[[N:.*]] = llvm.extractvalue %[[BDESC:.*]][3, 1]
  // CHECK-NEXT: %[[BATCH:.*]] = llvm.extractvalue %[[ADESC]][3, 0]
  // CHECK-NEXT: %[[STRIDE:.*]] = llvm.mul %[[K]], %[[N]] : i64
  // CHECK-NEXT: llvm.extractvalue %[[ADESC]][1]
  // CHECK-NEXT: %[[APTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[BDESC]][1]
  // CHECK-NEXT: %[[BPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[YDESC:.*]][1]
  // CHECK-NEXT: %[[YPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: %[[TRANSA:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[TRANSB:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[ADT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[BDT:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[YDT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[AZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: %[[BZP:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK-NEXT: %[[YZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_qmatmul(%{{.*}}, %[[APTR]], %[[BPTR]], %[[YPTR]], %[[M]], %[[N]], %[[K]], %[[BATCH]], %[[STRIDE]], %[[TRANSA]], %[[TRANSB]], %[[ADT]], %[[BDT]], %[[YDT]], %[[MSCALE]], %[[AZP]], %[[BZP]], %[[YZP]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64, i64) -> i32
  func.func @test_qmatmul(%ctx: !hip.context,
                          %A: memref<2x64x128xui16, 1>,
                          %B: memref<2x32x128xi8, 1>,
                          %Y: memref<2x64x32xui16, 1>) {
    hip.qmatmul(%ctx)
        ins(%A, %B : memref<2x64x128xui16, 1>, memref<2x32x128xi8, 1>)
        outs(%Y : memref<2x64x32xui16, 1>)
        {A_scale = 2.500000e-01 : f32, A_zero_point = 32768 : i64,
         B_scale = 1.250000e-01 : f32, B_zero_point = 3 : i64,
         Y_scale = 5.000000e-01 : f32, Y_zero_point = 32768 : i64,
         transA = 0 : i64, transB = 1 : i64}
    return
  }
}
