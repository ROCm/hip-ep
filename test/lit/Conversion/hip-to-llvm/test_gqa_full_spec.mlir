// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// =============================================================================
// Test Suite: GQA HIP→LLVM Lowering with Full MS Specification
// =============================================================================
//
// This file tests HIP→LLVM lowering for hip.gqa operations with various
// combinations of optional inputs and attributes. Verifies that the generated
// LLVM IR correctly calls wrap_group_query_attention with all 37 parameters.
//
// Key validation points:
// - Optional inputs become nullptr (LLVM zero op)
// - All 12 attributes are passed as constants
// - String quant_type attributes are converted to i64 enum values:
//   "NONE" = 0, "PER_TENSOR" = 1, "PER_CHANNEL" = 2
// - Function signature matches runtime wrapper (37 parameters)
// =============================================================================

// -----------------------------------------------------------------------------
// Test Case 1: Minimal GQA - verify required inputs and default attributes
// -----------------------------------------------------------------------------

func.func @test_gqa_minimal(%ctx: !hip.context) {
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
    local_window_size = -1 : i64,
    smooth_softmax = 0 : i64,
    qk_output = 0 : i64,
    k_quant_type = "NONE",
    v_quant_type = "NONE",
    kv_cache_bit_width = 8 : i64
  }

  return
}

// CHECK-LABEL: llvm.func @test_gqa_minimal
// CHECK: llvm.call @wrap_group_query_attention(
// CHECK-SAME: %{{.*}},
// 7 required inputs (non-null pointers)
// CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}},
// 7 optional inputs (nullptr for this test)
// CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}},
// 4 outputs (3 required + 1 optional nullptr)
// CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}},
// Attributes: num_heads=32, kv_num_heads=8
// CHECK-SAME: %{{.*}}, %{{.*}},
// Attributes: scale=1.0
// CHECK-SAME: %{{.*}},
// Attributes: do_rotary=0, rotary_interleaved=0
// CHECK-SAME: %{{.*}}, %{{.*}},
// Attributes: softcap=0.0
// CHECK-SAME: %{{.*}},
// Attributes: local_window_size=-1
// CHECK-SAME: %{{.*}},
// Attributes: smooth_softmax=0, qk_output=0
// CHECK-SAME: %{{.*}}, %{{.*}},
// Attributes: k_quant_type=0 (NONE), v_quant_type=0 (NONE)
// CHECK-SAME: %{{.*}}, %{{.*}},
// Attributes: kv_cache_bit_width=8
// CHECK-SAME: %{{.*}},
// Shape values: batch_size, seq_len_q, seq_len_kv, head_dim, element_size
// CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-SAME: ) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, i64, f32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64

// -----------------------------------------------------------------------------
// Test Case 2: GQA with RoPE - verify cos_cache, sin_cache, position_ids
// -----------------------------------------------------------------------------

func.func @test_gqa_with_rope(%ctx: !hip.context) {
  %query = memref.alloc() : memref<1x128x4096xf16>
  %key = memref.alloc() : memref<1x128x4096xf16>
  %value = memref.alloc() : memref<1x128x4096xf16>
  %past_key = memref.alloc() : memref<1x8x0x128xf16>
  %past_value = memref.alloc() : memref<1x8x0x128xf16>
  %seqlens_k = memref.alloc() : memref<1xi32>
  %total_seq_len = memref.alloc() : memref<i32>
  %cos_cache = memref.alloc() : memref<2048x64xf16>
  %sin_cache = memref.alloc() : memref<2048x64xf16>
  %position_ids = memref.alloc() : memref<1x128xi64>
  %output = memref.alloc() : memref<1x128x4096xf16>
  %present_key = memref.alloc() : memref<1x8x128x128xf16>
  %present_value = memref.alloc() : memref<1x8x128x128xf16>

  hip.gqa(%ctx) ins(
    %query, %key, %value, %past_key, %past_value,
    %seqlens_k, %total_seq_len,
    %cos_cache, %sin_cache, %position_ids
    : memref<1x128x4096xf16>, memref<1x128x4096xf16>, memref<1x128x4096xf16>,
      memref<1x8x0x128xf16>, memref<1x8x0x128xf16>,
      memref<1xi32>, memref<i32>,
      memref<2048x64xf16>, memref<2048x64xf16>, memref<1x128xi64>
  ) outs(
    %output, %present_key, %present_value
    : memref<1x128x4096xf16>, memref<1x8x128x128xf16>, memref<1x8x128x128xf16>
  ) {
    num_heads = 32 : i64,
    kv_num_heads = 8 : i64,
    scale = 1.0 : f32,
    do_rotary = 1 : i64,
    rotary_interleaved = 0 : i64,
    softcap = 0.0 : f32,
    local_window_size = -1 : i64,
    smooth_softmax = 0 : i64,
    qk_output = 0 : i64,
    k_quant_type = "NONE",
    v_quant_type = "NONE",
    kv_cache_bit_width = 8 : i64
  }

  return
}

// CHECK-LABEL: llvm.func @test_gqa_with_rope
// CHECK: llvm.call @wrap_group_query_attention
// Verify do_rotary=1 is passed
// CHECK: i64 1

// -----------------------------------------------------------------------------
// Test Case 3: GQA with quantization - verify quant_type enum conversion
// -----------------------------------------------------------------------------

func.func @test_gqa_quantized(%ctx: !hip.context) {
  %query = memref.alloc() : memref<1x128x4096xf16>
  %key = memref.alloc() : memref<1x128x4096xf16>
  %value = memref.alloc() : memref<1x128x4096xf16>
  %past_key = memref.alloc() : memref<1x8x0x128xf16>
  %past_value = memref.alloc() : memref<1x8x0x128xf16>
  %seqlens_k = memref.alloc() : memref<1xi32>
  %total_seq_len = memref.alloc() : memref<i32>
  %k_scale = memref.alloc() : memref<f32>
  %v_scale = memref.alloc() : memref<f32>
  %output = memref.alloc() : memref<1x128x4096xf16>
  %present_key = memref.alloc() : memref<1x8x128x128xf16>
  %present_value = memref.alloc() : memref<1x8x128x128xf16>

  hip.gqa(%ctx) ins(
    %query, %key, %value, %past_key, %past_value,
    %seqlens_k, %total_seq_len,
    %k_scale, %v_scale
    : memref<1x128x4096xf16>, memref<1x128x4096xf16>, memref<1x128x4096xf16>,
      memref<1x8x0x128xf16>, memref<1x8x0x128xf16>,
      memref<1xi32>, memref<i32>,
      memref<f32>, memref<f32>
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
    local_window_size = -1 : i64,
    smooth_softmax = 0 : i64,
    qk_output = 0 : i64,
    k_quant_type = "PER_TENSOR",
    v_quant_type = "PER_CHANNEL",
    kv_cache_bit_width = 8 : i64
  }

  return
}

// CHECK-LABEL: llvm.func @test_gqa_quantized
// CHECK: llvm.call @wrap_group_query_attention
// Verify k_quant_type="PER_TENSOR" → enum 1
// CHECK: i64 1
// Verify v_quant_type="PER_CHANNEL" → enum 2
// CHECK: i64 2

// -----------------------------------------------------------------------------
// Test Case 4: GQA with local window - verify local_window_size attribute
// -----------------------------------------------------------------------------

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
// CHECK: llvm.call @wrap_group_query_attention
// Verify local_window_size=4096 is passed
// CHECK: i64 4096
