// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: hip.qmatmul -> llvm.call @wrap_qmatmul
//
// One case per B quantization form, because the form decides which scale is
// foldable, whether the weight's scale/zero point travel as pointers, and what
// value width b_bits reports.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
// ============================================================================

module {
  // ===== Per-tensor B =====
  //
  // Carries every non-trivial part of the lowering at once:
  //   * A16W8 (ui16 / i8 / ui16) -> the three data-type codes are passed
  //     independently, so a wider A must not drag B's code along.
  //   * transB -> N comes from B's second-to-last extent, K still from A's last.
  //   * rank-3 per-batch B -> batch_count from A's leading dim and a
  //     b_batch_stride of K*N rather than the broadcast 0.
  //   * Asymmetric zero points, with 32768 proving a ui16 zero point survives
  //     instead of wrapping to -32768.
  //
  // The scales are chosen so the folded coefficient is exact in f32 and
  // readable:
  //   M_scale = A_scale * B_scale / Y_scale = 0.25 * 0.125 / 0.5 = 0.0625
  // Anything the kernel could recompute from A_scale/B_scale/Y_scale at runtime
  // would cost a float divide per output element, so the fold has to land here.
  // AY_ratio is 0 because it belongs to the other form, and a zero coefficient
  // is what a runtime reading the wrong slot should produce.
  //
  // B_scales / B_zero_points lower to null pointers, which is how the runtime
  // tells the two forms apart.
  //
  // Extents are read back from the descriptors even though they are static
  // here, so the checks bind SSA values instead of matching shape constants.
  //
  // Binding every argument lets the final CHECK pin the whole 22-parameter ABI
  // in order, matching wrap_qmatmul in lib/Runtime/hipdnn_ep_runtime.h:
  //   6 ptr : state, A, B, Y, B_scales, B_zero_points
  //   5 i64 : M, N, K, batch_count, b_batch_stride
  //   2 i64 : trans_a, trans_b
  //   3 i64 : a_data_type, b_data_type, y_data_type
  //   1 i64 : b_bits
  //   2 f32 : M_scale, AY_ratio
  //   3 i64 : A_zero_point, B_zero_point, Y_zero_point
  //
  // CHECK-LABEL: llvm.func @test_qmatmul_per_tensor(
  // CHECK:      %[[MSCALE:.*]] = llvm.mlir.constant(6.250000e-02 : f32) : f32
  // CHECK-NEXT: %[[AYRATIO:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
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
  // CHECK-NEXT: %[[BSNULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT: %[[BZNULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT: %[[TRANSA:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[TRANSB:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[ADT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[BDT:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[YDT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[BBITS:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK-NEXT: %[[AZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: %[[BZP:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK-NEXT: %[[YZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_qmatmul(%{{.*}}, %[[APTR]], %[[BPTR]], %[[YPTR]], %[[BSNULL]], %[[BZNULL]], %[[M]], %[[N]], %[[K]], %[[BATCH]], %[[STRIDE]], %[[TRANSA]], %[[TRANSB]], %[[ADT]], %[[BDT]], %[[YDT]], %[[BBITS]], %[[MSCALE]], %[[AYRATIO]], %[[AZP]], %[[BZP]], %[[YZP]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, i64, i64, i64) -> i32
  func.func @test_qmatmul_per_tensor(%ctx: !hip.context,
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

  // ===== Per-output-column B, packed 4-bit =====
  //
  // B_scales holds one f32 per column and cannot fold, so only A_scale/Y_scale
  // does: AY_ratio = 0.25 / 0.5 = 0.5, and M_scale stays 0. The kernel
  // completes the coefficient as AY_ratio * B_scales[n].
  //
  // b_bits = 4 is the only signal that the i8-typed B holds two values per
  // byte: `packed_int4` keeps the logical element count and the 8-bit element
  // type, so neither memref<2048x3072xi8> nor b_data_type shows the width.
  //
  // B_zero_point is 0 because the per-column zero points live in the
  // B_zero_points array instead.
  //
  // CHECK-LABEL: llvm.func @test_qmatmul_per_column(
  // CHECK:      %[[MSCALE:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK-NEXT: %[[AYRATIO:.*]] = llvm.mlir.constant(5.000000e-01 : f32) : f32
  // CHECK-NEXT: %[[M:.*]] = llvm.extractvalue %[[ADESC:.*]][3, 0]
  // CHECK-NEXT: %[[K:.*]] = llvm.extractvalue %[[ADESC]][3, 1]
  // CHECK-NEXT: %[[N:.*]] = llvm.extractvalue %[[BDESC:.*]][3, 1]
  // CHECK-NEXT: %[[BATCH:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: llvm.extractvalue %[[ADESC]][1]
  // CHECK-NEXT: %[[APTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[BDESC]][1]
  // CHECK-NEXT: %[[BPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[YDESC:.*]][1]
  // CHECK-NEXT: %[[YPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[BSDESC:.*]][1]
  // CHECK-NEXT: %[[BSPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: llvm.extractvalue %[[BZDESC:.*]][1]
  // CHECK-NEXT: %[[BZPTR:.*]] = llvm.addrspacecast
  // CHECK-NEXT: %[[TRANSA:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[TRANSB:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[ADT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[BDT:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[YDT:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK-NEXT: %[[BBITS:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK-NEXT: %[[AZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: %[[BZP:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[YZP:.*]] = llvm.mlir.constant(32768 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_qmatmul(%{{.*}}, %[[APTR]], %[[BPTR]], %[[YPTR]], %[[BSPTR]], %[[BZPTR]], %[[M]], %[[N]], %[[K]], %[[BATCH]], %[[STRIDE]], %[[TRANSA]], %[[TRANSB]], %[[ADT]], %[[BDT]], %[[YDT]], %[[BBITS]], %[[MSCALE]], %[[AYRATIO]], %[[AZP]], %[[BZP]], %[[YZP]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, i64, i64, i64) -> i32
  func.func @test_qmatmul_per_column(%ctx: !hip.context,
                                     %A: memref<128x2048xui16, 1>,
                                     %B: memref<2048x3072xi8, 1>,
                                     %Bs: memref<3072xf32, 1>,
                                     %Bz: memref<3072xi8, 1>,
                                     %Y: memref<128x3072xui16, 1>) {
    hip.qmatmul(%ctx)
        ins(%A, %B, %Bs, %Bz : memref<128x2048xui16, 1>,
                               memref<2048x3072xi8, 1>,
                               memref<3072xf32, 1>, memref<3072xi8, 1>)
        outs(%Y : memref<128x3072xui16, 1>)
        {A_scale = 2.500000e-01 : f32, A_zero_point = 32768 : i64,
         Y_scale = 5.000000e-01 : f32, Y_zero_point = 32768 : i64,
         B_quant_axis = 1 : i64, packed_int4}
    return
  }

  // ===== Per-output-column B, full-width 8-bit =====
  //
  // The absent `packed_int4` is the only difference from the case above, so
  // this pins just what it changes: b_bits = 8 next to real B_scales /
  // B_zero_points pointers. That pairing is what the runtime reads to pick the
  // 8-bit per-column path, and it is unreachable from either case above --
  // per-tensor passes b_bits = 8 with null pointers, packed per-column passes
  // pointers with b_bits = 4.
  //
  // CHECK-LABEL: llvm.func @test_qmatmul_per_column_w8(
  // CHECK:      %[[MSCALE:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK-NEXT: %[[AYRATIO:.*]] = llvm.mlir.constant(5.000000e-01 : f32) : f32
  // CHECK:      %[[BSPTR:.*]] = llvm.addrspacecast
  // CHECK:      %[[BZPTR:.*]] = llvm.addrspacecast
  // CHECK:      %[[BBITS:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK:      llvm.call @wrap_qmatmul(%{{.*}}, %[[BSPTR]], %[[BZPTR]], {{.*}}, %[[BBITS]], %[[MSCALE]], %[[AYRATIO]],
  func.func @test_qmatmul_per_column_w8(%ctx: !hip.context,
                                        %A: memref<128x2048xui16, 1>,
                                        %B: memref<2048x3072xi8, 1>,
                                        %Bs: memref<3072xf32, 1>,
                                        %Bz: memref<3072xi8, 1>,
                                        %Y: memref<128x3072xui16, 1>) {
    hip.qmatmul(%ctx)
        ins(%A, %B, %Bs, %Bz : memref<128x2048xui16, 1>,
                               memref<2048x3072xi8, 1>,
                               memref<3072xf32, 1>, memref<3072xi8, 1>)
        outs(%Y : memref<128x3072xui16, 1>)
        {A_scale = 2.500000e-01 : f32, A_zero_point = 32768 : i64,
         Y_scale = 5.000000e-01 : f32, Y_zero_point = 32768 : i64,
         B_quant_axis = 1 : i64}
    return
  }
}
