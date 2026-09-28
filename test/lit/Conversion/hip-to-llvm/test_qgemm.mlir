// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: hip.qgemm -> llvm.call @wrap_qgemm
//
// Two cases split the ABI along the axis that actually changes shape: whether
// B is quantized per tensor or per output channel. Everything else is chosen
// to make one lowering decision visible per operand.
//
// The requant coefficients are folded here rather than in the kernel, which is
// the whole reason the scales are attributes: recomputing
// alpha*s_a*s_b/s_y per output element would cost a float divide each. The
// scales below are powers of two so the fold is exact in f32 and readable.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
// ============================================================================

module {
  // Per-tensor B with an int32 bias -- the shape an ONNX quantizer emits, since
  // it accumulates the bias at C_scale = A_scale * B_scale.
  //
  //   M_ab = alpha * A_scale * B_scale / Y_scale = 2 * 0.25 * 0.125 / 0.5 = 0.125
  //   M_c  = beta * C_scale / Y_scale            = 0.5 * 0.03125 / 0.5   = 0.03125
  //
  // C is rank 1, so it normalizes to [1, 32] and broadcasts along M; the two
  // extents are what tell the kernel which of its indices to drop. The
  // per-channel pointers are null, which is how the runtime picks the
  // per-tensor path.
  //
  // The full 25-parameter ABI, matching wrap_qgemm in
  // lib/Runtime/hipdnn_ep_runtime.h:
  //   7 ptr : state, A, B, C, B_scales, B_zero_points, Y
  //   3 i64 : M, N, K
  //   2 i64 : trans_a, trans_b
  //   4 i64 : a/b/c/y_data_type
  //   3 i64 : b_bits, c_dim0, c_dim1
  //   2 f32 : M_ab, M_c
  //   4 i64 : A/B/C/Y_zero_point
  //
  // CHECK-LABEL: llvm.func @test_qgemm_per_tensor(
  // CHECK:      %[[CDIM0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[CDIM1:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-NEXT: %[[M:.*]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK-NEXT: %[[K:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK-NEXT: %[[N:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-NEXT: llvm.extractvalue %[[ADESC:.*]][1]
  // CHECK-NEXT: %[[APTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[BDESC:.*]][1]
  // CHECK-NEXT: %[[BPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[CDESC:.*]][1]
  // CHECK-NEXT: %[[CPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: %[[BSPTR:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT: %[[BZPTR:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT: llvm.extractvalue %[[YDESC:.*]][1]
  // CHECK-NEXT: %[[YPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: %[[TRANSA:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[TRANSB:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[ADT:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[BDT:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[CDT:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK-NEXT: %[[YDT:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[BBITS:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK-NEXT: %[[MAB:.*]] = llvm.mlir.constant(1.250000e-01 : f32) : f32
  // CHECK-NEXT: %[[MC:.*]] = llvm.mlir.constant(3.125000e-02 : f32) : f32
  // CHECK-NEXT: %[[AZP:.*]] = llvm.mlir.constant(-5 : i64) : i64
  // CHECK-NEXT: %[[BZP:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK-NEXT: %[[CZP:.*]] = llvm.mlir.constant(-2 : i64) : i64
  // CHECK-NEXT: %[[YZP:.*]] = llvm.mlir.constant(7 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_qgemm(%{{.*}}, %[[APTR]], %[[BPTR]], %[[CPTR]], %[[BSPTR]], %[[BZPTR]], %[[YPTR]], %[[M]], %[[N]], %[[K]], %[[TRANSA]], %[[TRANSB]], %[[ADT]], %[[BDT]], %[[CDT]], %[[YDT]], %[[BBITS]], %[[CDIM0]], %[[CDIM1]], %[[MAB]], %[[MC]], %[[AZP]], %[[BZP]], %[[CZP]], %[[YZP]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, i64, i64, i64, i64) -> i32
  func.func @test_qgemm_per_tensor(%ctx: !hip.context,
                                   %A: memref<64x128xi8, 1>,
                                   %B: memref<32x128xi8, 1>,
                                   %C: memref<32xi32, 1>,
                                   %Y: memref<64x32xi8, 1>) {
    hip.qgemm(%ctx)
        ins(%A, %B, %C : memref<64x128xi8, 1>, memref<32x128xi8, 1>,
                         memref<32xi32, 1>)
        outs(%Y : memref<64x32xi8, 1>)
        {A_scale = 2.500000e-01 : f32, A_zero_point = -5 : i64,
         B_scale = 1.250000e-01 : f32, B_zero_point = 3 : i64,
         C_scale = 3.125000e-02 : f32, C_zero_point = -2 : i64,
         Y_scale = 5.000000e-01 : f32, Y_zero_point = 7 : i64,
         alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32,
         transA = 0 : i64, transB = 1 : i64}
    return
  }

  // Per-channel 4-bit B against a 16-bit activation, no bias.
  //
  // B_scale stays at its 1.0 identity here, so the folded coefficient collapses
  // to alpha * A_scale / Y_scale = 0.25 / 0.5 = 0.5 and the per-N factor is
  // left for the kernel to read out of B_scales. An absent C zeroes M_c
  // outright rather than leaving a live beta behind a null pointer, and reports
  // c_data_type as -1 (HIPDNN_EP_DATATYPE_UNSUPPORTED) for "no bias".
  //
  // b_bits = 4 is the only thing distinguishing the packed weight: both widths
  // arrive as memref<...xi8>, and at 4 each byte carries two nibbles.
  //
  // M is dynamic, so it comes off the descriptor instead of folding to a
  // constant, while K and N still fold.
  //
  // A_zero_point = 32768 proves a ui16 zero point is carried as an unsigned
  // value rather than wrapping to -32768.
  //
  // CHECK-LABEL: llvm.func @test_qgemm_per_channel_w4(
  // CHECK:      %[[CDIM0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[CDIM1:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[M:.*]] = llvm.extractvalue %[[ADESC:.*]][3, 0]
  // CHECK-NEXT: %[[K:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-NEXT: %[[N:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-NEXT: llvm.extractvalue %[[ADESC]][1]
  // CHECK-NEXT: %[[APTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[BDESC:.*]][1]
  // CHECK-NEXT: %[[BPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: %[[CPTR:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT: llvm.extractvalue %[[BSDESC:.*]][1]
  // CHECK-NEXT: %[[BSPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[BZDESC:.*]][1]
  // CHECK-NEXT: %[[BZPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[YDESC:.*]][1]
  // CHECK-NEXT: %[[YPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: %[[TRANSA:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[TRANSB:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[ADT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[BDT:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[CDT:.*]] = llvm.mlir.constant(-1 : i64) : i64
  // CHECK-NEXT: %[[YDT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[BBITS:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK-NEXT: %[[MAB:.*]] = llvm.mlir.constant(5.000000e-01 : f32) : f32
  // CHECK-NEXT: %[[MC:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK-NEXT: %[[AZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: %[[BZP:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[CZP:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[YZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_qgemm(%{{.*}}, %[[APTR]], %[[BPTR]], %[[CPTR]], %[[BSPTR]], %[[BZPTR]], %[[YPTR]], %[[M]], %[[N]], %[[K]], %[[TRANSA]], %[[TRANSB]], %[[ADT]], %[[BDT]], %[[CDT]], %[[YDT]], %[[BBITS]], %[[CDIM0]], %[[CDIM1]], %[[MAB]], %[[MC]], %[[AZP]], %[[BZP]], %[[CZP]], %[[YZP]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, i64, i64, i64, i64) -> i32
  func.func @test_qgemm_per_channel_w4(%ctx: !hip.context,
                                       %A: memref<?x32xui16, 1>,
                                       %B: memref<16x32xi8, 1>,
                                       %Bs: memref<16xf32, 1>,
                                       %Bzp: memref<16xi8, 1>,
                                       %Y: memref<?x16xui16, 1>) {
    hip.qgemm(%ctx)
        ins(%A, %B : memref<?x32xui16, 1>, memref<16x32xi8, 1>)
        b_per_channel(%Bs, %Bzp : memref<16xf32, 1>, memref<16xi8, 1>)
        outs(%Y : memref<?x16xui16, 1>)
        {A_scale = 2.500000e-01 : f32, A_zero_point = 32768 : i64,
         B_bits = 4 : i64,
         Y_scale = 5.000000e-01 : f32, Y_zero_point = 32768 : i64,
         transB = 1 : i64}
    return
  }
}
