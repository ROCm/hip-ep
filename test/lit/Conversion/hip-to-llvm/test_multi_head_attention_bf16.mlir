// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The runtime wrapper takes an element size in bytes. bf16 is 2 bytes, so
// lowering it would call the f16 kernels. Conversion must fail instead.
// RUN: hip-mlir-opt --convert-hip-to-llvm --verify-diagnostics %s

module {
  func.func @bf16(%ctx: !hip.context,
                  %query: memref<1x4x16xbf16, 1>,
                  %key: memref<1x4x16xbf16, 1>,
                  %value: memref<1x4x16xbf16, 1>,
                  %output: memref<1x4x16xbf16, 1>) {
    // expected-error @+1 {{failed to legalize operation 'hip.multi_head_attention'}}
    hip.multi_head_attention(%ctx)
        ins(%query, %key, %value :
            memref<1x4x16xbf16, 1>, memref<1x4x16xbf16, 1>,
            memref<1x4x16xbf16, 1>)
        outs(%output : memref<1x4x16xbf16, 1>)
        {num_heads = 2 : i64,
         mask_filter_value = -1.000000e+04 : f32,
         scale = 5.000000e-01 : f32,
         unidirectional = 0 : i64}
    return
  }
}
