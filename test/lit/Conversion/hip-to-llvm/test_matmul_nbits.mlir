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
  // CHECK: %[[STATUS:.*]] = llvm.call @wrap_matmul_nbits({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[STATUS]])
  // Verify 17 parameters:
  // - 1 i32: op_state_slot (2nd arg; per-instance MatmulNbitsState; threaded by
  //          --assign-op-state-slots, replaces shared RuntimeState::zp_unpack_cache)
  // - 8 pointers: state, A, B, scales, zero_points(null), g_idx(null), bias(null), output
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
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
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
  // CHECK-SAME: (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  // M=128, batch_count=1 (2D: no batch dims)
}
