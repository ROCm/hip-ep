// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.conv_transpose is split into stride-1 hip.conv residues that
// rocMLIR can anchor on, and that unsupported cases are left for MIOpen.
//
// Test cases:
// 1. stride2      - 2x upsample, 3x3 kernel -> 4 residues (2x2, 2x1, 1x2, 1x1)
// 2. stride1      - no upsampling -> a single conv, no reassembly
// 3. dilated      - dilation > 1 is out of scope, op survives
// 4. stride_gt_k  - an empty residue would leave holes, op survives
// 5. grouped      - tosa.conv2d has no grouped form, op survives
// 6. residue_taps - exact sub-filter values: tap phase and spatial flip
// 7. channel_swap - exact sub-filter values: [C, M] -> [M, C]
// 8. with_bias    - bias is one broadcast add on the reassembled result
// 9. f64          - hip.conv has no f64 lowering, op survives
// 10. many_residues - stride^2 past the kernel-count cap, op survives
//
// Cases 1-5 use a splat filter and so only pin structure. Cases 6 and 7 use
// distinct per-tap values, which is what actually catches a wrong tap phase,
// a missing reversal, or a transposed channel mapping.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-decompose-conv-transpose --canonicalize | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. Stride 2: residue (i, j) keeps taps {i, i+2} x {j, j+2}, so the filter
  //    splits 3x3 into 2x2 / 2x1 / 1x2 / 1x1 sub-filters with the channels
  //    swapped from [C, M, ...] to [M, C, ...].
  // --------------------------------------------------------------------------
  func.func @stride2(%ctx: !hip.context, %x: tensor<1x8x16x16xf32>)
      -> tensor<1x16x32x32xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<8x16x3x3xf32>}
        : tensor<8x16x3x3xf32>
    %init = tensor.empty() : tensor<1x16x32x32xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x8x16x16xf32>,
                                               tensor<8x16x3x3xf32>)
        outs(%init : tensor<1x16x32x32xf32>)
        {kernel_shape = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1],
         dilations = [1, 1], output_padding = [1, 1], group = 1 : i64}
        : tensor<1x16x32x32xf32>
    return %y : tensor<1x16x32x32xf32>
  }

  // CHECK-LABEL: func.func @stride2
  // CHECK-NOT: hip.conv_transpose
  // Every residue is a dense stride-1 convolution over the full input.
  // CHECK-DAG: hip.conv({{.*}} tensor<16x8x2x2xf32>){{.*}} {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]}
  // CHECK-DAG: hip.conv({{.*}} tensor<16x8x2x1xf32>){{.*}} kernel_shape = [2, 1]
  // CHECK-DAG: hip.conv({{.*}} tensor<16x8x1x2xf32>){{.*}} kernel_shape = [1, 2]
  // CHECK-DAG: hip.conv({{.*}} tensor<16x8x1x1xf32>){{.*}} kernel_shape = [1, 1]
  // Residue (i, j) lands on output positions {i + 2k} x {j + 2k}...
  // CHECK-DAG: tensor.insert_slice {{.*}}[0, 0, 0, 0] [1, 16, 17, 17] [1, 1, 2, 2]
  // CHECK-DAG: tensor.insert_slice {{.*}}[0, 0, 1, 1] [1, 16, 17, 17] [1, 1, 2, 2]
  // ...and the leading pad is cropped back off.
  // CHECK: tensor.extract_slice {{.*}}[0, 0, 1, 1] [1, 16, 32, 32] [1, 1, 1, 1]

  // --------------------------------------------------------------------------
  // 2. Stride 1: one residue keeping every tap, so the decomposition is just a
  //    spatial flip plus the channel swap. No scatter, no crop beyond the pad.
  // --------------------------------------------------------------------------
  func.func @stride1(%ctx: !hip.context, %x: tensor<1x8x16x16xf32>)
      -> tensor<1x16x16x16xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<8x16x3x3xf32>}
        : tensor<8x16x3x3xf32>
    %init = tensor.empty() : tensor<1x16x16x16xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x8x16x16xf32>,
                                               tensor<8x16x3x3xf32>)
        outs(%init : tensor<1x16x16x16xf32>)
        {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x16x16x16xf32>
    return %y : tensor<1x16x16x16xf32>
  }

  // CHECK-LABEL: func.func @stride1
  // CHECK-NOT: hip.conv_transpose
  // CHECK: hip.conv({{.*}} tensor<16x8x3x3xf32>){{.*}} kernel_shape = [3, 3]
  // CHECK-NOT: hip.conv(

  // --------------------------------------------------------------------------
  // 3. Dilation > 1 needs the general zero-stuff reassembly, not the strided
  //    insert used here.
  // --------------------------------------------------------------------------
  func.func @dilated(%ctx: !hip.context, %x: tensor<1x8x16x16xf32>)
      -> tensor<1x16x35x35xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<8x16x3x3xf32>}
        : tensor<8x16x3x3xf32>
    %init = tensor.empty() : tensor<1x16x35x35xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x8x16x16xf32>,
                                               tensor<8x16x3x3xf32>)
        outs(%init : tensor<1x16x35x35xf32>)
        {kernel_shape = [3, 3], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [2, 2], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x16x35x35xf32>
    return %y : tensor<1x16x35x35xf32>
  }

  // CHECK-LABEL: func.func @dilated
  // CHECK: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 4. stride > kernel leaves residues with no taps, so the strided inserts
  //    would not cover the destination.
  // --------------------------------------------------------------------------
  func.func @stride_gt_k(%ctx: !hip.context, %x: tensor<1x8x16x16xf32>)
      -> tensor<1x16x63x63xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<8x16x3x3xf32>}
        : tensor<8x16x3x3xf32>
    %init = tensor.empty() : tensor<1x16x63x63xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x8x16x16xf32>,
                                               tensor<8x16x3x3xf32>)
        outs(%init : tensor<1x16x63x63xf32>)
        {kernel_shape = [3, 3], strides = [4, 4], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x16x63x63xf32>
    return %y : tensor<1x16x63x63xf32>
  }

  // CHECK-LABEL: func.func @stride_gt_k
  // CHECK: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 5. tosa.conv2d has no grouped form, so hip.conv could not consume the
  //    residues even though the split itself is well defined.
  // --------------------------------------------------------------------------
  func.func @grouped(%ctx: !hip.context, %x: tensor<1x8x16x16xf32>)
      -> tensor<1x16x18x18xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<8x8x3x3xf32>}
        : tensor<8x8x3x3xf32>
    %init = tensor.empty() : tensor<1x16x18x18xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x8x16x16xf32>,
                                               tensor<8x8x3x3xf32>)
        outs(%init : tensor<1x16x18x18xf32>)
        {kernel_shape = [3, 3], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 2 : i64}
        : tensor<1x16x18x18xf32>
    return %y : tensor<1x16x18x18xf32>
  }

  // CHECK-LABEL: func.func @grouped
  // CHECK: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 6. One channel each way and a 3x3 filter numbered 0..8, so every residue's
  //    contents are forced. Residue (i, j) takes taps {i, i+2} x {j, j+2} and
  //    position a holds tap (taps - 1 - a), i.e. the taps run backwards:
  //
  //      w = 0 1 2     (0,0) -> 8 6   (0,1) -> 7   (1,0) -> 5 3   (1,1) -> 4
  //          3 4 5              2 0            1
  //          6 7 8
  //
  //    A forgotten reversal would emit [[0, 2], [6, 8]] for residue (0, 0);
  //    an off-by-one tap phase would swap the (0,1) and (1,0) sub-filters.
  // --------------------------------------------------------------------------
  func.func @residue_taps(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x1x9x9xf32> {
    %w = hip.constant {value = dense<[[[[0.0, 1.0, 2.0],
                                        [3.0, 4.0, 5.0],
                                        [6.0, 7.0, 8.0]]]]>
        : tensor<1x1x3x3xf32>} : tensor<1x1x3x3xf32>
    %init = tensor.empty() : tensor<1x1x9x9xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x1x3x3xf32>)
        outs(%init : tensor<1x1x9x9xf32>)
        {kernel_shape = [3, 3], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x1x9x9xf32>
    return %y : tensor<1x1x9x9xf32>
  }

  // CHECK-LABEL: func.func @residue_taps
  // CHECK-NOT: hip.conv_transpose
  // CHECK-DAG: hip.constant {value = dense<{{\[}}[{{\[}}[8.000000e+00, 6.000000e+00], [2.000000e+00, 0.000000e+00]]]]> : tensor<1x1x2x2xf32>}
  // CHECK-DAG: hip.constant {value = dense<{{\[}}[{{\[}}[7.000000e+00], [1.000000e+00]]]]> : tensor<1x1x2x1xf32>}
  // CHECK-DAG: hip.constant {value = dense<{{\[}}[{{\[}}[5.000000e+00, 3.000000e+00]]]]> : tensor<1x1x1x2xf32>}
  // CHECK-DAG: hip.constant {value = dense<4.000000e+00> : tensor<1x1x1x1xf32>}

  // --------------------------------------------------------------------------
  // 7. ConvTranspose weights are [C, M, ...] but hip.conv wants [M, C, ...].
  //    A 1x1 filter removes the spatial dimension from the picture, so the
  //    only thing under test is the channel transpose: with C = 2, M = 3 and
  //    w[c][m] = 1 + 3c + m, the residue filter must read column-first.
  // --------------------------------------------------------------------------
  func.func @channel_swap(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
      -> tensor<1x3x4x4xf32> {
    %w = hip.constant {value = dense<[[[[1.0]], [[2.0]], [[3.0]]],
                                      [[[4.0]], [[5.0]], [[6.0]]]]>
        : tensor<2x3x1x1xf32>} : tensor<2x3x1x1xf32>
    %init = tensor.empty() : tensor<1x3x4x4xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x2x4x4xf32>,
                                               tensor<2x3x1x1xf32>)
        outs(%init : tensor<1x3x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x3x4x4xf32>
    return %y : tensor<1x3x4x4xf32>
  }

  // CHECK-LABEL: func.func @channel_swap
  // CHECK-NOT: hip.conv_transpose
  // CHECK: hip.constant {value = dense<{{\[}}[{{\[}}[1.000000e+00]], [{{\[}}4.000000e+00]]], [{{\[}}[2.000000e+00]], [{{\[}}5.000000e+00]]], [{{\[}}[3.000000e+00]], [{{\[}}6.000000e+00]]]]> : tensor<3x2x1x1xf32>}

  // --------------------------------------------------------------------------
  // 8. Bias is applied once to the reassembled result, broadcast over N/H/W.
  // --------------------------------------------------------------------------
  func.func @with_bias(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %b = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>}
        : tensor<2xf32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w, %b : tensor<1x1x4x4xf32>,
                                                   tensor<1x2x2x2xf32>,
                                                   tensor<2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    return %y : tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @with_bias
  // CHECK-NOT: hip.conv_transpose
  // The bias is reshaped to [1, M, 1, 1] and added after the crop, once.
  // CHECK: tensor.expand_shape {{.*}} into tensor<1x2x1x1xf32>
  // CHECK: hip.add
  // CHECK-NOT: hip.add

  // --------------------------------------------------------------------------
  // 9. hip.conv lowers to tosa.conv2d only for f16/bf16/f32, so decomposing an
  //    f64 transpose would just move the failure to legalization.
  // --------------------------------------------------------------------------
  func.func @f64(%ctx: !hip.context, %x: tensor<1x1x4x4xf64>)
      -> tensor<1x1x5x5xf64> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x1x2x2xf64>}
        : tensor<1x1x2x2xf64>
    %init = tensor.empty() : tensor<1x1x5x5xf64>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf64>,
                                               tensor<1x1x2x2xf64>)
        outs(%init : tensor<1x1x5x5xf64>)
        {kernel_shape = [2, 2], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x1x5x5xf64>
    return %y : tensor<1x1x5x5xf64>
  }

  // CHECK-LABEL: func.func @f64
  // CHECK: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 10. The residue count is strideH * strideW, and each one becomes its own
  //     serially compiled kernel. 9x9 = 81 is past the cap, so this stays on
  //     the MIOpen path even though the split itself is well defined.
  // --------------------------------------------------------------------------
  func.func @many_residues(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x1x36x36xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x1x9x9xf32>}
        : tensor<1x1x9x9xf32>
    %init = tensor.empty() : tensor<1x1x36x36xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x1x9x9xf32>)
        outs(%init : tensor<1x1x36x36xf32>)
        {kernel_shape = [9, 9], strides = [9, 9], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x1x36x36xf32>
    return %y : tensor<1x1x36x36xf32>
  }

  // CHECK-LABEL: func.func @many_residues
  // CHECK: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 11. Nothing verifies the result rank against the operands -- it comes from
  //     the `outs` operand, which is only constrained to be a tensor or memref.
  //     A rank-3 result must bail out rather than index dim 3 of a 3-D type.
  // --------------------------------------------------------------------------
  func.func @result_rank_mismatch(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x1x2x2xf32>}
        : tensor<1x1x2x2xf32>
    %init = tensor.empty() : tensor<1x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x1x2x2xf32>)
        outs(%init : tensor<1x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x8x8xf32>
    return %y : tensor<1x8x8xf32>
  }

  // CHECK-LABEL: func.func @result_rank_mismatch
  // CHECK: hip.conv_transpose
}
