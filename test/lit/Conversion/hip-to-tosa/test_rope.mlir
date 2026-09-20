// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

// Position IDs select rows from the 2-D cos/sin lookup tables. The BSH input
// is unpacked to BNSH, rotated, and restored to BSH.
// CHECK-LABEL: func.func @rope_bsh_position_ids
// CHECK-COUNT-2: tosa.gather
// CHECK: tosa.negate
// CHECK: tosa.mul
// CHECK: tosa.add
// CHECK: tosa.concat
// CHECK-NOT: hip.rope
func.func @rope_bsh_position_ids(
    %ctx: !hip.context, %input: tensor<2x3x8xf16>,
    %position_ids: tensor<2x3xi64>, %cos: tensor<16x2xf16>,
    %sin: tensor<16x2xf16>, %init: tensor<2x3x8xf16>)
    -> tensor<2x3x8xf16> attributes {rock.kernel} {
  %result = hip.rope(%ctx)
      ins(%input, %position_ids, %cos, %sin :
          tensor<2x3x8xf16>, tensor<2x3xi64>, tensor<16x2xf16>,
          tensor<16x2xf16>)
      outs(%init : tensor<2x3x8xf16>)
      {interleaved = 0 : i64, num_heads = 2 : i64,
       rotary_embedding_dim = 4 : i64}
      : tensor<2x3x8xf16>
  return %result : tensor<2x3x8xf16>
}

// Native ONNX RotaryEmbedding can omit position_ids and provide caches that
// are already expanded to [B,S,rotary_dim/2]. Cover interleaving and a tail
// that lies outside rotary_embedding_dim.
// CHECK-LABEL: func.func @rope_bnsh_expanded_partial_interleaved
// CHECK-NOT: tosa.gather
// CHECK: tosa.slice
// CHECK: tosa.negate
// CHECK: tosa.concat
// CHECK-NOT: hip.rope
func.func @rope_bnsh_expanded_partial_interleaved(
    %ctx: !hip.context, %input: tensor<1x2x3x6xf32>,
    %cos: tensor<1x3x2xf32>, %sin: tensor<1x3x2xf32>,
    %init: tensor<1x2x3x6xf32>)
    -> tensor<1x2x3x6xf32> attributes {rock.kernel} {
  %result = "hip.rope"(%ctx, %input, %cos, %sin, %init)
      {interleaved = 1 : i64, num_heads = 2 : i64,
       rotary_embedding_dim = 4 : i64}
      : (!hip.context, tensor<1x2x3x6xf32>, tensor<1x3x2xf32>,
         tensor<1x3x2xf32>, tensor<1x2x3x6xf32>) -> tensor<1x2x3x6xf32>
  return %result : tensor<1x2x3x6xf32>
}
