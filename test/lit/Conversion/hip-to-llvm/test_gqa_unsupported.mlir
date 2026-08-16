// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt --verify-each=0 --test-make-gqa-softcap-unsupported --convert-hip-to-llvm --mlir-print-ir-after-failure %s 2>&1 | FileCheck %s

// The input is valid when parsed. A tool-only pass injects nonzero softcap
// immediately before HIP-to-LLVM conversion, bypassing the normal verifier so
// this test reaches the lowering pattern's defensive check directly.
func.func @softcap(
    %ctx: !hip.context, %query: memref<1x1x32xf16, 1>,
    %key: memref<1x1x16xf16, 1>, %value: memref<1x1x16xf16, 1>,
    %past_key: memref<1x2x0x8xf16, 1>,
    %past_value: memref<1x2x0x8xf16, 1>,
    %seqlens: memref<1xi32, 1>, %total: memref<i32, 1>,
    %out: memref<1x1x32xf16, 1>, %present_key: memref<1x2x1x8xf16, 1>,
    %present_value: memref<1x2x1x8xf16, 1>) {
  // CHECK: error: 'hip.gqa' op softcap must be exactly zero
  hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value, %seqlens, %total :
        memref<1x1x32xf16, 1>, memref<1x1x16xf16, 1>,
        memref<1x1x16xf16, 1>, memref<1x2x0x8xf16, 1>,
        memref<1x2x0x8xf16, 1>, memref<1xi32, 1>, memref<i32, 1>)
    outs(%out, %present_key, %present_value :
        memref<1x1x32xf16, 1>, memref<1x2x1x8xf16, 1>,
        memref<1x2x1x8xf16, 1>)
    {num_heads = 4 : i64, kv_num_heads = 2 : i64,
     test.gqa_lowering_unsupported}
  return
}

// Lowering rejects before declaring or calling the runtime wrapper.
// CHECK-NOT: wrap_group_query_attention
