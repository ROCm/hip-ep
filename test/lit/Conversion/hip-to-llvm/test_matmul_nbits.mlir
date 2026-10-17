// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.matmul_nbits is correctly lowered to an LLVM call to
// wrap_matmul_nbits runtime function.
//
// This test validates:
// - Runtime call generation (hip.matmul_nbits → llvm.call @wrap_matmul_nbits)
// - Pointer extraction for required operands (A, B, scales, output)
// - Null pointer generation for absent optional operands (zero_points, g_idx, bias)
// - Pointer extraction for present optional operands
// - Dimension extraction from memref descriptors (M, batch_count)
// - Attribute-to-constant lowering (N, K, bits, block_size, elem_size)
// ============================================================================

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s

module {
  // ===== Test 1: Basic MatMulNBits (no optional operands) =====

  func.func @test_matmul_nbits_basic(%ctx: !hip.context,
                                      %A: memref<1x128x2880xf16, 1>,
                                      %B: memref<5120x90x16xui8, 1>,
                                      %scales: memref<5120x90xf16, 1>,
                                      %output: memref<1x128x5120xf16, 1>) {
    hip.matmul_nbits(%ctx) ins(%A, %B, %scales :
        memref<1x128x2880xf16, 1>, memref<5120x90x16xui8, 1>,
        memref<5120x90xf16, 1>)
        outs(%output : memref<1x128x5120xf16, 1>)
        {K = 2880 : i64, N = 5120 : i64, bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64,
         zp_elem_size = 0 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_matmul_nbits_basic
  // CHECK: llvm.call @wrap_matmul_nbits({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  // Verify 19 parameters:
  // - 1 i32: op_state_slot (2nd arg; per-instance MatmulNbitsState; threaded by
  //          --assign-op-state-slots, replaces shared RuntimeState::zp_unpack_cache)
  // - 10 pointers: state, A, B, scales, zero_points(null), g_idx(null),
  //   bias(null), a_quant(null), a_scale(null), output
  // - 8 i64: M=128, N=5120, K=2880, batch_count=1, bits=4, block_size=32, elem_size=2, zp_elem_size=0

  // ===== Test 2: MatMulNBits with zero_points =====

  func.func @test_matmul_nbits_with_zp(%ctx: !hip.context,
                                        %A: memref<1x128x2880xf16, 1>,
                                        %B: memref<5120x90x16xui8, 1>,
                                        %scales: memref<5120x90xf16, 1>,
                                        %zp: memref<5120x90xui8, 1>,
                                        %output: memref<1x128x5120xf16, 1>) {
    hip.matmul_nbits(%ctx) ins(%A, %B, %scales :
        memref<1x128x2880xf16, 1>, memref<5120x90x16xui8, 1>,
        memref<5120x90xf16, 1>)
        zero_points(%zp : memref<5120x90xui8, 1>)
        outs(%output : memref<1x128x5120xf16, 1>)
        {K = 2880 : i64, N = 5120 : i64, bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64,
         zp_elem_size = 1 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_matmul_nbits_with_zp
  // CHECK: llvm.call @wrap_matmul_nbits({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64) -> i32

  // ===== Test 3: 2D MatMulNBits (no batch dimension) =====

  func.func @test_matmul_nbits_2d(%ctx: !hip.context,
                                   %A: memref<128x2880xf16, 1>,
                                   %B: memref<5120x90x16xui8, 1>,
                                   %scales: memref<5120x90xf16, 1>,
                                   %output: memref<128x5120xf16, 1>) {
    hip.matmul_nbits(%ctx) ins(%A, %B, %scales :
        memref<128x2880xf16, 1>, memref<5120x90x16xui8, 1>,
        memref<5120x90xf16, 1>)
        outs(%output : memref<128x5120xf16, 1>)
        {K = 2880 : i64, N = 5120 : i64, bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64,
         zp_elem_size = 0 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_matmul_nbits_2d
  // CHECK: llvm.call @wrap_matmul_nbits({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  // M=128, batch_count=1 (2D: no batch dims)

  // ===== Test 4: quantize_act feeding a prequantized MatMulNBits =====

  func.func @test_matmul_nbits_quantized(%ctx: !hip.context,
                                         %A: memref<1x?x2880xf16, 1>,
                                         %B: memref<5120x90x16xui8, 1>,
                                         %scales: memref<5120x90xf16, 1>,
                                         %aq: memref<1x?x2880xi8, 1>,
                                         %as: memref<1x?x90xf32, 1>,
                                         %output: memref<1x?x5120xf16, 1>) {
    hip.quantize_act(%ctx) ins(%A : memref<1x?x2880xf16, 1>)
        outs(%aq, %as : memref<1x?x2880xi8, 1>, memref<1x?x90xf32, 1>)
        {block_size = 32 : i64}
    hip.matmul_nbits(%ctx) ins(%A, %B, %scales :
        memref<1x?x2880xf16, 1>, memref<5120x90x16xui8, 1>,
        memref<5120x90xf16, 1>)
        quantized(%aq, %as : memref<1x?x2880xi8, 1>, memref<1x?x90xf32, 1>)
        outs(%output : memref<1x?x5120xf16, 1>)
        {K = 2880 : i64, N = 5120 : i64, bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64,
         zp_elem_size = 0 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_matmul_nbits_quantized
  // rows = 1 * dim(1), K = 2880, block_size = 32
  // CHECK: llvm.mul
  // CHECK: %[[ROWS:[0-9]+]] = llvm.mul %{{[0-9]+}}, %{{[0-9]+}} : i64
  // CHECK: %[[K:[0-9]+]] = llvm.mlir.constant(2880 : i64) : i64
  // CHECK: %[[BS:[0-9]+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: llvm.call @wrap_quantize_act(%arg0, %{{[0-9]+}}, %[[AQ:[0-9]+]], %[[AS:[0-9]+]], %[[ROWS]], %[[K]], %[[BS]]) :
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  // zero_points, g_idx and bias are null; a_quant and a_scale are passed:
  // CHECK: %[[ZP:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[GIDX:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[BIAS:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llvm.call @wrap_matmul_nbits(%arg0, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[ZP]], %[[GIDX]], %[[BIAS]], %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}},

  // ===== Test 5: MatMulNBitsGroup, q/k/v without zero points =====

  func.func @test_matmul_nbits_group_qkv(%ctx: !hip.context,
      %A: memref<1x1x64xf16, 1>, %aq: memref<1x1x64xi8, 1>,
      %as: memref<1x1x2xf32, 1>,
      %Bq: memref<64x2x16xui8, 1>, %Bk: memref<32x2x16xui8, 1>,
      %Bv: memref<32x2x16xui8, 1>, %Sq: memref<64x2xf16, 1>,
      %Sk: memref<32x2xf16, 1>, %Sv: memref<32x2xf16, 1>,
      %q: memref<1x1x64xf16, 1>, %k: memref<1x1x32xf16, 1>,
      %v: memref<1x1x32xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x64xf16, 1>)
        quantized(%aq, %as : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
        weights(%Bq, %Bk, %Bv : memref<64x2x16xui8, 1>,
                memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
        scales(%Sq, %Sk, %Sv : memref<64x2xf16, 1>, memref<32x2xf16, 1>,
               memref<32x2xf16, 1>)
        outs(%q, %k, %v : memref<1x1x64xf16, 1>, memref<1x1x32xf16, 1>,
             memref<1x1x32xf16, 1>)
        {K = 64 : i64, N = [64, 32, 32], bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_matmul_nbits_group_qkv
  // Slots: B x3, scales x3, zero_points x3 (all null), outputs x3, N x3;
  // then M, K, batch, block_size and epilogue 0.
  // CHECK: %[[Z0:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[Z1:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[Z2:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[N0:[0-9]+]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK: %[[N1:[0-9]+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[N2:[0-9]+]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[NSEG:[0-9]+]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[EPI:[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @wrap_matmul_nbits_group(%arg0, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[NSEG]],
  // CHECK-SAME: %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[Z0]], %[[Z1]], %[[Z2]],
  // CHECK-SAME: %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[N0]], %[[N1]], %[[N2]],
  // CHECK-SAME: %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[EPI]]) :
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64) -> i32

  // ===== Test 6: MatMulNBitsGroup, gate/up with SiLU and zero points =====

  func.func @test_matmul_nbits_group_silu(%ctx: !hip.context,
      %A: memref<1x?x64xf16, 1>, %aq: memref<1x?x64xi8, 1>,
      %as: memref<1x?x2xf32, 1>,
      %Bg: memref<128x2x16xui8, 1>, %Bu: memref<128x2x16xui8, 1>,
      %Sg: memref<128x2xf16, 1>, %Su: memref<128x2xf16, 1>,
      %Zg: memref<128x1xui8, 1>, %Zu: memref<128x1xui8, 1>,
      %y: memref<1x?x128xf16, 1>, %up: memref<1x?x128xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x?x64xf16, 1>)
        quantized(%aq, %as : memref<1x?x64xi8, 1>, memref<1x?x2xf32, 1>)
        weights(%Bg, %Bu : memref<128x2x16xui8, 1>, memref<128x2x16xui8, 1>)
        scales(%Sg, %Su : memref<128x2xf16, 1>, memref<128x2xf16, 1>)
        zero_points(%Zg, %Zu : memref<128x1xui8, 1>, memref<128x1xui8, 1>)
        outs(%y : memref<1x?x128xf16, 1>)
        up_tmp(%up : memref<1x?x128xf16, 1>)
        {K = 64 : i64, N = [128, 128], bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64, epilogue = 1 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @test_matmul_nbits_group_silu
  // The third slot of every operand list is null / 0, and up_tmp fills the
  // second output slot.
  // CHECK: %[[W2:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[S2:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[Z2:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[YP:[0-9]+]] = llvm.addrspacecast
  // CHECK: %[[UPP:[0-9]+]] = llvm.addrspacecast
  // CHECK: %[[O2:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[N0:[0-9]+]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK: %[[N1:[0-9]+]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK: %[[N2:[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[NSEG:[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[EPI:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.call @wrap_matmul_nbits_group(%arg0, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[NSEG]],
  // CHECK-SAME: %{{[0-9]+}}, %{{[0-9]+}}, %[[W2]], %{{[0-9]+}}, %{{[0-9]+}}, %[[S2]], %{{[0-9]+}}, %{{[0-9]+}}, %[[Z2]],
  // CHECK-SAME: %[[YP]], %[[UPP]], %[[O2]], %[[N0]], %[[N1]], %[[N2]],
  // CHECK-SAME: %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[EPI]]) :
}
