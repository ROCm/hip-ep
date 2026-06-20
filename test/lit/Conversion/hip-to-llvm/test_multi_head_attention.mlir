// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s
//
// The trailing i32 on each wrap_multi_head_attention call is op_state_slot,
// threaded by --assign-op-state-slots. It selects this MHA instance's
// per-instance MhaState (hipBLASLt GEMM descriptor cache), replacing the
// former shared RuntimeState::mha_gemm_cache.

module {
  func.func @test_multi_head_attention_lowering(
      %ctx: !hip.context,
      %query: memref<1x128x4096xf16, 1>,
      %key: memref<1x128x4096xf16, 1>,
      %value: memref<1x128x4096xf16, 1>,
      %output: memref<1x128x4096xf16, 1>) {
    hip.multi_head_attention(%ctx)
        ins(%query, %key, %value :
            memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>,
            memref<1x128x4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {num_heads = 32 : i64,
         mask_filter_value = -1.000000e+04 : f32,
         scale = 0.0883883461 : f32,
         unidirectional = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_multi_head_attention_lowering
// CHECK: llvm.call @wrap_multi_head_attention
// CHECK-SAME: -> i32

// Verify signature:
// - 15 pointers: state, query, key, value, bias(NULL), key_padding_mask(NULL),
//                attention_bias(NULL), past_key(NULL), past_value(NULL),
//                past_sequence_length(NULL), cache_indirection(NULL),
//                output, present_key(NULL), present_value(NULL), qk(NULL)
// - 4 attributes: num_heads=32, mask_filter_value=-10000.0, scale=0.0883883461,
//                 unidirectional=0
// - 8 shape params: batch_size=1, seq_len_q=128, seq_len_kv=128,
//                   query_hidden=4096, v_hidden=4096, head_size=0,
//                   query_rank=3, element_size_bytes=2
// - 1 i32: op_state_slot (per-instance MhaState; threaded by
//          --assign-op-state-slots, replaces shared RuntimeState::mha_gemm_cache)

// =============================================================================
// Dynamic shape: batch_size, seq_len, hidden_dim are all dynamic.
// Shape values fed to the runtime must come from the MemRef descriptor
// (llvm.extractvalue %md[3, N]) rather than compile-time constants.
// =============================================================================
module {
  func.func @test_multi_head_attention_dynamic(
      %ctx: !hip.context,
      %query: memref<?x?x?xf16, 1>,
      %key: memref<?x?x?xf16, 1>,
      %value: memref<?x?x?xf16, 1>,
      %output: memref<?x?x?xf16, 1>) {
    hip.multi_head_attention(%ctx)
        ins(%query, %key, %value :
            memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>)
        outs(%output : memref<?x?x?xf16, 1>)
        {num_heads = 32 : i64,
         mask_filter_value = -1.000000e+04 : f32,
         scale = 0.0883883461 : f32,
         unidirectional = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_multi_head_attention_dynamic
// CHECK: llvm.extractvalue {{.*}}[3, 0]
// CHECK: llvm.extractvalue {{.*}}[3, 1]
// CHECK: llvm.extractvalue {{.*}}[3, 2]
// CHECK: llvm.call @wrap_multi_head_attention
// CHECK-SAME: -> i32
