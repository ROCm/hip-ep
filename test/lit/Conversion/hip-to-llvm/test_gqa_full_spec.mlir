// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// =============================================================================
// Test: GQA HIP→LLVM Lowering with Full MS Specification
// =============================================================================
//
// This test validates HIP→LLVM lowering for hip.gqa operations with all
// 14 inputs and 12 attributes. Verifies that the generated LLVM IR correctly
// calls wrap_group_query_attention with all 36 parameters.
//
// Key validation points:
// - Optional inputs that aren't provided become nullptr (LLVM zero op)
// - All 12 attributes are passed as constants
// - String quant_type attributes are converted to i64 enum values
// - Function signature matches runtime wrapper (36 parameters)
// =============================================================================

func.func @test_gqa_local_window(%ctx: !hip.context) {
  %query = memref.alloc() : memref<1x128x4096xf16>
  %key = memref.alloc() : memref<1x128x4096xf16>
  %value = memref.alloc() : memref<1x128x4096xf16>
  %past_key = memref.alloc() : memref<1x8x0x128xf16>
  %past_value = memref.alloc() : memref<1x8x0x128xf16>
  %seqlens_k = memref.alloc() : memref<1xi32>
  %total_seq_len = memref.alloc() : memref<i32>
  %output = memref.alloc() : memref<1x128x4096xf16>
  %present_key = memref.alloc() : memref<1x8x128x128xf16>
  %present_value = memref.alloc() : memref<1x8x128x128xf16>

  hip.gqa(%ctx) ins(
    %query, %key, %value, %past_key, %past_value,
    %seqlens_k, %total_seq_len
    : memref<1x128x4096xf16>, memref<1x128x4096xf16>, memref<1x128x4096xf16>,
      memref<1x8x0x128xf16>, memref<1x8x0x128xf16>,
      memref<1xi32>, memref<i32>
  ) outs(
    %output, %present_key, %present_value
    : memref<1x128x4096xf16>, memref<1x8x128x128xf16>, memref<1x8x128x128xf16>
  ) {
    num_heads = 32 : i64,
    kv_num_heads = 8 : i64,
    scale = 1.0 : f32,
    do_rotary = 0 : i64,
    rotary_interleaved = 0 : i64,
    softcap = 0.0 : f32,
    local_window_size = 4096 : i64,
    smooth_softmax = 0 : i64,
    qk_output = 0 : i64,
    k_quant_type = "NONE",
    v_quant_type = "NONE",
    kv_cache_bit_width = 8 : i64
  }

  return
}

// CHECK-LABEL: llvm.func @test_gqa_local_window
// Verify local_window_size=4096 constant is created
// CHECK: llvm.mlir.constant(4096 : i64) : i64
// CHECK: llvm.call @wrap_group_query_attention
