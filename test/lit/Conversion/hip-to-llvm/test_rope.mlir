// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_rope_lowering(%ctx: !hip.context,
                                 %input: memref<1x128x4096xf16, 1>,
                                 %position_ids: memref<1x128xi64, 1>,
                                 %cos_cache: memref<131072x64xf16, 1>,
                                 %sin_cache: memref<131072x64xf16, 1>,
                                 %output: memref<1x128x4096xf16, 1>) {
    hip.rope(%ctx)
        ins(%input, %position_ids, %cos_cache, %sin_cache :
            memref<1x128x4096xf16, 1>, memref<1x128xi64, 1>,
            memref<131072x64xf16, 1>, memref<131072x64xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {interleaved = 0 : i64, num_heads = 32 : i64,
         rotary_embedding_dim = 128 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_rope_lowering
// CHECK: llvm.call @wrap_rotary_embedding({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// Verify 15 parameters:
// - 6 pointers: state, input, position_ids, cos_cache, sin_cache, output
// - 9 i64: interleaved=0, batch=1, seq_len=128, num_heads=32, head_dim=128,
//   rotary_dim=128, cos_cache_num_elements=8388608, element_size_bytes=2,
//   is_bnsh=0 (3D BSH input)

// Dynamic batch+seq_len: batch and seq_len are read from the memref descriptor
// at runtime. num_heads and head_dim remain static (architecture constants).
module {
  func.func @test_rope_lowering_dynamic(%ctx: !hip.context,
                                         %input: memref<?x?x4096xf16, 1>,
                                         %position_ids: memref<?x?xi64, 1>,
                                         %cos_cache: memref<131072x64xf16, 1>,
                                         %sin_cache: memref<131072x64xf16, 1>,
                                         %output: memref<?x?x4096xf16, 1>) {
    hip.rope(%ctx)
        ins(%input, %position_ids, %cos_cache, %sin_cache :
            memref<?x?x4096xf16, 1>, memref<?x?xi64, 1>,
            memref<131072x64xf16, 1>, memref<131072x64xf16, 1>)
        outs(%output : memref<?x?x4096xf16, 1>)
        {interleaved = 0 : i64, num_heads = 32 : i64,
         rotary_embedding_dim = 128 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_rope_lowering_dynamic
// Batch and seq_len come from MemRefDescriptor::size() rather than constants.
// CHECK: llvm.extractvalue {{.*}}[3, 0]
// CHECK: llvm.extractvalue {{.*}}[3, 1]
// CHECK: llvm.call @wrap_rotary_embedding({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
