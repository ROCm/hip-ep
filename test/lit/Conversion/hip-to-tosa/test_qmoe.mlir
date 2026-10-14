// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.qmoe decomposes to TOSA as softmax routing + a top-k mask,
// followed by every expert's dequantized FC1 / SwiGLU / FC2 folded in weighted
// by its routing weight. TOSA cannot express the runtime's data-dependent
// sparse dispatch, so the expert loop is unrolled and unselected tokens are
// carried by a zero routing weight instead of being skipped.
//
// FILE LAYOUT:
// Converting cases first, each in its own --split-input-file chunk; the
// declined configurations follow. Declining is not an error -- the op is
// dynamically legal, so an unsupported configuration stays a hip op instead of
// failing the pass.
//
// Shapes throughout: 2 experts, hidden=32, inter=16, 2 tokens, block_size=16.
// That makes fc1 [E, 2*inter, hidden/2] = [2, 32, 16] with [2, 32, 2] scales,
// and fc2 [E, hidden, inter/2] = [2, 32, 8] with [2, 32, 1] scales.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Baseline: k=1, no bias, no zero points, no renormalization.
// reduce_max/exp/reduce_sum is the routing softmax; reduce_min is the
// lower-index tie-break inside the top-k round; sigmoid is SwiGLU.
// CHECK-LABEL: func.func @qmoe_basic
// CHECK: tosa.reduce_max
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum
// CHECK: tosa.reduce_max
// CHECK: tosa.equal
// CHECK: tosa.reduce_min
// CHECK: tosa.select
// CHECK: tosa.logical_right_shift
// CHECK: tosa.concat
// CHECK: tosa.matmul
// CHECK: tosa.sigmoid
// CHECK: tosa.matmul
// CHECK-NOT: hip.qmoe
func.func @qmoe_basic(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                      %router: tensor<2x2xf16>,
                      %fc1w: tensor<2x32x16xui8>, %fc1s: tensor<2x32x2xf16>,
                      %fc2w: tensor<2x32x8xui8>, %fc2s: tensor<2x32x1xf16>,
                      %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// k=2 runs a second selection round, so a second reduce_min appears and the
// two one-hot rows are merged with a bitwise or (which on i1 is logical or).
// CHECK-LABEL: func.func @qmoe_topk2
// CHECK: tosa.reduce_min
// CHECK: tosa.bitwise_or
// CHECK-NOT: hip.qmoe
func.func @qmoe_topk2(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                      %router: tensor<2x2xf16>,
                      %fc1w: tensor<2x32x16xui8>, %fc1s: tensor<2x32x2xf16>,
                      %fc2w: tensor<2x32x8xui8>, %fc2s: tensor<2x32x1xf16>,
                      %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 2 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// normalize_routing_weights divides the masked weights by their own sum, which
// is a reduce_sum/reciprocal pair after the mask rather than before it.
// CHECK-LABEL: func.func @qmoe_normalized
// CHECK: tosa.reduce_min
// CHECK: tosa.select
// CHECK: tosa.reduce_sum
// CHECK: tosa.reciprocal
// CHECK-NOT: hip.qmoe
func.func @qmoe_normalized(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                           %router: tensor<2x2xf16>,
                           %fc1w: tensor<2x32x16xui8>, %fc1s: tensor<2x32x2xf16>,
                           %fc2w: tensor<2x32x8xui8>, %fc2s: tensor<2x32x1xf16>,
                           %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 2 : i64, block_size = 16 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// Biases are per-expert rows sliced out and broadcast across the tokens.
// CHECK-LABEL: func.func @qmoe_with_bias
// CHECK: tosa.matmul
// CHECK: tosa.add
// CHECK: tosa.sigmoid
// CHECK-NOT: hip.qmoe
func.func @qmoe_with_bias(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                          %router: tensor<2x2xf16>,
                          %fc1w: tensor<2x32x16xui8>, %fc1s: tensor<2x32x2xf16>,
                          %fc2w: tensor<2x32x8xui8>, %fc2s: tensor<2x32x1xf16>,
                          %fc1b: tensor<2x32xf16>, %fc2b: tensor<2x32xf16>,
                          %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      fc1_bias(%fc1b : tensor<2x32xf16>)
      fc2_bias(%fc2b : tensor<2x32xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// fc1 has 2 blocks per row, so its zero points arrive as one packed byte per
// row and get unpacked the same way the weights do. fc2 has a single block, so
// its zero-point byte is already one value per block and is used directly.
// CHECK-LABEL: func.func @qmoe_packed_zero_points
// CHECK: tosa.matmul
// CHECK: tosa.sub
// CHECK-NOT: hip.qmoe
func.func @qmoe_packed_zero_points(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                                   %router: tensor<2x2xf16>,
                                   %fc1w: tensor<2x32x16xui8>,
                                   %fc1s: tensor<2x32x2xf16>,
                                   %fc2w: tensor<2x32x8xui8>,
                                   %fc2s: tensor<2x32x1xf16>,
                                   %fc1z: tensor<2x32x1xui8>,
                                   %fc2z: tensor<2x32x1xui8>,
                                   %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      fc1_zero_points(%fc1z : tensor<2x32x1xui8>)
      fc2_zero_points(%fc2z : tensor<2x32x1xui8>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// Unfused SwiGLU (swiglu_fusion=0) puts the gate in a separate fc3 GEMM rather
// than interleaved in fc1. The runtime rejects it too, so it stays a hip op.
// CHECK-LABEL: func.func @qmoe_reject_unfused_swiglu
// CHECK: hip.qmoe
func.func @qmoe_reject_unfused_swiglu(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                                      %router: tensor<2x2xf16>,
                                      %fc1w: tensor<2x32x16xui8>,
                                      %fc1s: tensor<2x32x2xf16>,
                                      %fc2w: tensor<2x32x8xui8>,
                                      %fc2s: tensor<2x32x1xf16>,
                                      %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 0 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// 8-bit experts do not use the packed-nibble layout the dequantize assumes.
// CHECK-LABEL: func.func @qmoe_reject_bits8
// CHECK: hip.qmoe
func.func @qmoe_reject_bits8(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                             %router: tensor<2x2xf16>,
                             %fc1w: tensor<2x32x32xui8>,
                             %fc1s: tensor<2x32x2xf16>,
                             %fc2w: tensor<2x32x16xui8>,
                             %fc2s: tensor<2x32x1xf16>,
                             %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x32xui8>, tensor<2x32x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x1xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 8 : i64, k = 1 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// The sparse mixer replaces top-k with a different, jitter-based selection.
// CHECK-LABEL: func.func @qmoe_reject_sparse_mixer
// CHECK: hip.qmoe
func.func @qmoe_reject_sparse_mixer(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                                    %router: tensor<2x2xf16>,
                                    %fc1w: tensor<2x32x16xui8>,
                                    %fc1s: tensor<2x32x2xf16>,
                                    %fc2w: tensor<2x32x8xui8>,
                                    %fc2s: tensor<2x32x1xf16>,
                                    %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 1 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// router_weights takes precedence over router_probs and skips the softmax, so
// the routing modelled here would be wrong.
// CHECK-LABEL: func.func @qmoe_reject_router_weights
// CHECK: hip.qmoe
func.func @qmoe_reject_router_weights(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                                      %router: tensor<2x2xf16>,
                                      %fc1w: tensor<2x32x16xui8>,
                                      %fc1s: tensor<2x32x2xf16>,
                                      %fc2w: tensor<2x32x8xui8>,
                                      %fc2s: tensor<2x32x1xf16>,
                                      %rw: tensor<2x2xf16>,
                                      %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      router_weights(%rw : tensor<2x2xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// block_size must be a power of two of at least 16 for the scale broadcast to
// tile cleanly across each block.
// CHECK-LABEL: func.func @qmoe_reject_block_size
// CHECK: hip.qmoe
func.func @qmoe_reject_block_size(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                                  %router: tensor<2x2xf16>,
                                  %fc1w: tensor<2x32x16xui8>,
                                  %fc1s: tensor<2x32x4xf16>,
                                  %fc2w: tensor<2x32x8xui8>,
                                  %fc2s: tensor<2x32x2xf16>,
                                  %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x4xf16>,
        tensor<2x32x8xui8>, tensor<2x32x2xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 8 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}

// -----

// k must not exceed the expert count; there is nothing left to select.
// CHECK-LABEL: func.func @qmoe_reject_k_gt_experts
// CHECK: hip.qmoe
func.func @qmoe_reject_k_gt_experts(%ctx: !hip.context, %x: tensor<1x2x32xf16>,
                                    %router: tensor<2x2xf16>,
                                    %fc1w: tensor<2x32x16xui8>,
                                    %fc1s: tensor<2x32x2xf16>,
                                    %fc2w: tensor<2x32x8xui8>,
                                    %fc2s: tensor<2x32x1xf16>,
                                    %init: tensor<1x2x32xf16>) -> tensor<1x2x32xf16>
    attributes {rock.kernel} {
  %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<1x2x32xf16>, tensor<2x2xf16>,
        tensor<2x32x16xui8>, tensor<2x32x2xf16>,
        tensor<2x32x8xui8>, tensor<2x32x1xf16>)
      outs(%init : tensor<1x2x32xf16>)
      {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 16 : i64,
       normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64,
       activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
       swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
      : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}
