// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Negative-case lockdown for the 4D BNSH RoPE lowering.  When the
// `num_heads` attribute is the default-0 sentinel ("infer from shape") and
// the corresponding shape dim is also kDynamic, there is nothing to infer
// from -- the lowering must surface this as a notifyMatchFailure rather
// than emitting a kDynamic sentinel into the runtime call signature
// (which would produce a "0 head_dim" or unbounded head loop on the
// device side).
//
// This guards lib/Conversion/HipToLLVM/RopeLowering.cpp:95-98:
//   if (headDimVal == kDynamic || numHeadsVal == kDynamic)
//     return notifyMatchFailure("dynamic num_heads/head_dim not supported");
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt --convert-hip-to-llvm %s 2>&1 | FileCheck %s

// CHECK: error: failed to legalize operation 'hip.rope'

module {
  func.func @test_rope_bnsh_dynamic_num_heads(
      %ctx: !hip.context,
      %input: memref<?x?x?x128xf16, 1>,
      %position_ids: memref<?x?xi64, 1>,
      %cos_cache: memref<131072x64xf16, 1>,
      %sin_cache: memref<131072x64xf16, 1>,
      %output: memref<?x?x?x128xf16, 1>) {
    hip.rope(%ctx)
        ins(%input, %position_ids, %cos_cache, %sin_cache :
            memref<?x?x?x128xf16, 1>, memref<?x?xi64, 1>,
            memref<131072x64xf16, 1>, memref<131072x64xf16, 1>)
        outs(%output : memref<?x?x?x128xf16, 1>)
        {interleaved = 0 : i64, num_heads = 0 : i64,
         rotary_embedding_dim = 128 : i64}
    return
  }
}
