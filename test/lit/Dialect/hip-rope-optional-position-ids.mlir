// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s | hip-mlir-opt | FileCheck %s

// hip.rope's position_ids is optional; an op without it, printed and parsed
// back, keeps its three operands as input, cos and sin (the `,`-led group the
// format had read them as input, position_ids and cos).

// CHECK-LABEL: func.func @expanded
// CHECK: hip.rope(%{{.*}}) ins(%arg1, %arg2, %arg3 : tensor<?x?x4096xf16>, tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(
// CHECK-NOT: position_ids
func.func @expanded(%ctx: !hip.context, %x: tensor<?x?x4096xf16>,
                    %cos: tensor<?x?x128xf16>, %sin: tensor<?x?x128xf16>,
                    %init: tensor<?x?x4096xf16>) -> tensor<?x?x4096xf16> {
  %y = hip.rope(%ctx)
    ins(%x, %cos, %sin : tensor<?x?x4096xf16>, tensor<?x?x128xf16>, tensor<?x?x128xf16>)
    outs(%init : tensor<?x?x4096xf16>)
    {interleaved = 0 : i64, num_heads = 16 : i64, rotary_embedding_dim = 256 : i64} : tensor<?x?x4096xf16>
  return %y : tensor<?x?x4096xf16>
}

// CHECK-LABEL: func.func @indexed
// CHECK: hip.rope(%{{.*}}) ins(%arg1, %arg3, %arg4 : tensor<1x128x4096xf16>, tensor<131072x64xf16>, tensor<131072x64xf16>) position_ids(%arg2 : tensor<1x128xi64>) outs(
func.func @indexed(%ctx: !hip.context, %x: tensor<1x128x4096xf16>,
                   %pos: tensor<1x128xi64>, %cos: tensor<131072x64xf16>,
                   %sin: tensor<131072x64xf16>,
                   %init: tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16> {
  %y = hip.rope(%ctx)
    ins(%x, %cos, %sin : tensor<1x128x4096xf16>, tensor<131072x64xf16>, tensor<131072x64xf16>)
    position_ids(%pos : tensor<1x128xi64>)
    outs(%init : tensor<1x128x4096xf16>)
    {interleaved = 0 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64} : tensor<1x128x4096xf16>
  return %y : tensor<1x128x4096xf16>
}
