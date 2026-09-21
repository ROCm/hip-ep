// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// PatchEmbedConvToGemm is a native C++ pattern and QConvFusion is a PDLL one,
// but both root at hip.conv and share one RewritePatternSet, so `benefit`
// decides which is offered a conv first. That comparison is the reason
// QConvFusion roots at the conv rather than at the hip.quantize_linear it
// replaces -- patterns on different roots are never ranked against each other.
//
// The two cases below are one conv geometry with and without a Q/DQ pair
// around it, so the only thing differing between them is which of the two
// patterns can match at all.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A 1x1 conv over a 1x1 input is a single patch, which exempts it from the
// unit-kernel guard, so PatchEmbedConvToGemm matches this conv. So does
// QConvFusion, and it carries the higher benefit, so the conv goes to it first
// and the fused integer kernel wins over a dequantized GEMM that would have
// stranded the surrounding Q/DQ pair.
// CHECK-LABEL: func.func @qconv_wins_on_benefit
// CHECK-NOT:     hip.gemm
// CHECK:         hip.qconv
// CHECK-NOT:     hip.gemm
func.func @qconv_wins_on_benefit(%ctx: !hip.context,
                                 %x: tensor<1x8x1x1xui16>) -> tensor<1x4x1x1xui16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 16 : i64} : tensor<4x8x1x1xi8>
  %w_zp = hip.constant {location = "w.bin", offset = 16 : i64, size = 2 : i64} : tensor<4xi8>
  %w_scale = hip.constant {location = "w.bin", offset = 32 : i64, size = 16 : i64} : tensor<4xf32>
  %x_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %x_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x8x1x1xf32>
  %dq_x = hip.dequantize_linear(%ctx)
      ins(%x, %x_scale : tensor<1x8x1x1xui16>, tensor<f32>)
      zero_point(%x_zp : tensor<ui16>)
      outs(%e0 : tensor<1x8x1x1xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x1x1xf32>

  %e1 = tensor.empty() : tensor<4x8x1x1xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<4x8x1x1xi8>, tensor<4xf32>)
      zero_point(%w_zp : tensor<4xi8>)
      outs(%e1 : tensor<4x8x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4} : tensor<4x8x1x1xf32>

  %e2 = tensor.empty() : tensor<1x4x1x1xf32>
  %conv = hip.conv(%ctx)
      ins(%dq_x, %dq_w : tensor<1x8x1x1xf32>, tensor<4x8x1x1xf32>)
      outs(%e2 : tensor<1x4x1x1xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x1x1xf32>

  %e3 = tensor.empty() : tensor<1x4x1x1xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%conv, %y_scale : tensor<1x4x1x1xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e3 : tensor<1x4x1x1xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x4x1x1xui16>

  return %q : tensor<1x4x1x1xui16>
}

// -----

// The same conv geometry with no Q/DQ around it. QConvFusion is a Q/DQ fusion
// by definition, so there is nothing for it to match and the conv falls
// through to the lower-benefit pattern on the same root -- a higher benefit
// only wins where it matches.
//
// The absence of the Q/DQ pair is deliberately the reason it does not match.
// Making the conv quantized but unfusable would date the test against the
// quantization formats hip.qconv grows.
// CHECK-LABEL: func.func @patch_embed_takes_over_without_qdq
// CHECK-NOT:     hip.qconv
// CHECK-NOT:     hip.conv
// CHECK:         hip.gemm
func.func @patch_embed_takes_over_without_qdq(%ctx: !hip.context,
                                              %x: tensor<1x8x1x1xf32>,
                                              %w: tensor<4x8x1x1xf32>) -> tensor<1x4x1x1xf32> {
  %init = tensor.empty() : tensor<1x4x1x1xf32>
  %y = hip.conv(%ctx)
      ins(%x, %w : tensor<1x8x1x1xf32>, tensor<4x8x1x1xf32>)
      outs(%init : tensor<1x4x1x1xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x1x1xf32>
  return %y : tensor<1x4x1x1xf32>
}
