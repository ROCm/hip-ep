// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.conv_transpose is split into stride-1 hip.conv residues that
// rocMLIR can anchor on, and that unsupported cases are left for MIOpen.
//
// Test cases:
// 1. stride2      - 2x upsample, 3x3 kernel -> residues padded to one 2x2 conv
// 2. stride1      - no upsampling -> a single conv, no reassembly
// 3. dilated      - dilation > 1 is out of scope, op survives
// 4. stride_gt_k  - an empty residue would leave holes, op survives
// 5. grouped      - tosa.conv2d has no grouped form, op survives
// 6. residue_taps - exact padded filter values: tap phase, flip, zero taps
// 7. channel_swap - exact sub-filter values: [C, M] -> [M, C]
// 8. with_bias    - a constant bias becomes the wide conv's bias, tiled
// 9. f64          - hip.conv has no f64 lowering, op survives
// 10. many_residues - stride^2 past the kernel-count cap, op survives
// 11. result_rank_mismatch - rank-3 result, op survives
// 12. batch_mismatch       - result batch != input batch, op survives
// 13. out_pad_no_slack     - equal taps, so interleave; grid still grows
// 14. result_too_large     - result past the output_padding bound, op survives
// 15. equal_residue_values - exact wide filter: residue, channel and tap order
// 16. dynamic_1d            - (see the 1-D pattern) dynamic width
// 17. unequal_residues      - padding would cost > 2x taps: per-residue convs,
//                             run-time bias on each of them, no add
// 18. runtime_bias_equal    - run-time bias on the wide conv stays one add
// 19. add_reshaped_bias     - Add(y, Reshape([1,1,1,M] -> [1,M,1,1])) folds
//                             into the wide conv's bias; the add is gone
// 20. add_and_own_bias      - the added constant sums with the op's own bias
// 21. add_scalar            - a scalar added to every channel folds too
// 22. add_transposed_bias   - a transposed [1,1,1,M] constant folds
// 23. add_unequal_residues  - the folded bias goes onto every residue conv
// 24. add_rank1_no_fold     - rank-1 [M] broadcasts on W, not on channels
// 25. add_full_size_no_fold - a constant with values per pixel is not a bias
// 26. add_runtime_no_fold   - a run-time operand stays an add
// 27. add_two_users_no_fold - the bias-free result has another reader
//
// Cases 1-5 use a splat filter and so only pin structure. Cases 6, 7 and 15
// use distinct per-tap values, which is what actually catches a wrong tap
// phase, a missing reversal, a transposed channel mapping, or a wide filter
// packed in the wrong order. Cases 11, 12 and 14
// cover operand/result shape disagreements that no verifier rejects, so the
// pattern has to reject them itself rather than build invalid IR.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-decompose-conv-transpose --canonicalize | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. Stride 2: residue (i, j) keeps taps {i, i+2} x {j, j+2}, so the filter
  //    splits 3x3 into 2x2 / 2x1 / 1x2 / 1x1 sub-filters with the channels
  //    swapped from [C, M, ...] to [M, C, ...]. 16 padded taps against 9 real
  //    ones is within the 2x bound, so the smaller residues get zero taps and
  //    all four share one convolution and the interleave.
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
  // CHECK: hip.conv({{.*}} tensor<64x8x2x2xf32>){{.*}} {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [1, 1]}
  // CHECK-SAME: tensor<1x64x17x17xf32>
  // CHECK-NOT: hip.conv(
  // CHECK: tensor.expand_shape {{.*}} into tensor<1x2x2x16x17x17xf32>
  // CHECK: hip.transpose
  // CHECK: tensor.collapse_shape {{.*}} into tensor<1x16x34x34xf32>
  // The leading pad is cropped back off.
  // CHECK: tensor.extract_slice {{.*}}[0, 0, 1, 1] [1, 16, 32, 32] [1, 1, 1, 1]
  // CHECK-NOT: tensor.insert_slice

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
  //    The three smaller residues are padded to 2x2 with their zero taps in
  //    front (low positions), where the grown leading pad reads nothing:
  //    (0,1) -> [[0, 7], [0, 1]], (1,0) -> [[0, 0], [5, 3]], (1,1) ->
  //    [[0, 0], [0, 4]]. Zeros at the back would shift the residue by a tap.
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
  // CHECK: hip.constant {value = dense<{{\[}}[{{\[}}[8.000000e+00, 6.000000e+00], [2.000000e+00, 0.000000e+00]]], [{{\[}}[0.000000e+00, 7.000000e+00], [0.000000e+00, 1.000000e+00]]], [{{\[}}[0.000000e+00, 0.000000e+00], [5.000000e+00, 3.000000e+00]]], [{{\[}}[0.000000e+00, 0.000000e+00], [0.000000e+00, 4.000000e+00]]]]> : tensor<4x1x2x2xf32>}
  // CHECK: hip.conv({{.*}} tensor<4x1x2x2xf32>){{.*}} kernel_shape = [2, 2], pads = [1, 1, 1, 1]

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
  // 8. A constant bias becomes the wide convolution's bias, repeated once per
  //    residue (residue-major, like the filter), so no add follows the crop.
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
  // CHECK: hip.constant {value = dense<[1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<8xf32>}
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>, tensor<8xf32>)
  // CHECK-NOT: hip.add
  // CHECK: return

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

  // --------------------------------------------------------------------------
  // 12. The batch is read off the input and used for the residue convs, the
  //     reassembly grid and the crop sizes, while the crop's result type comes
  //     from the op. Nothing verifies the two agree, so a mismatch has to be
  //     rejected here or the crop would contradict its own result type.
  // --------------------------------------------------------------------------
  func.func @batch_mismatch(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<2x1x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x1x2x2xf32>}
        : tensor<1x1x2x2xf32>
    %init = tensor.empty() : tensor<2x1x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x1x2x2xf32>)
        outs(%init : tensor<2x1x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<2x1x8x8xf32>
    return %y : tensor<2x1x8x8xf32>
  }

  // CHECK-LABEL: func.func @batch_mismatch
  // CHECK: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 13. The stride divides the kernel, so the residue grid has no trailing
  //     slack for output_padding to land in: 4 residues of 4 -> a 8x8 grid,
  //     against a 9x9 result. Each residue's trailing pad grows by one to
  //     stretch the grid to 10x10, and the extra positions convolve only
  //     past-the-end input, so they come out zero as ONNX requires.
  // --------------------------------------------------------------------------
  func.func @out_pad_no_slack(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x1x9x9xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x1x2x2xf32>}
        : tensor<1x1x2x2xf32>
    %init = tensor.empty() : tensor<1x1x9x9xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x1x2x2xf32>)
        outs(%init : tensor<1x1x9x9xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [1, 1], group = 1 : i64}
        : tensor<1x1x9x9xf32>
    return %y : tensor<1x1x9x9xf32>
  }

  // CHECK-LABEL: func.func @out_pad_no_slack
  // Each residue keeps a single tap and the same padding, so the residues
  // become one wide convolution (4 * M output channels) whose result is
  // interleaved by one transpose, rather than four strided inserts.
  // The trailing pad is 1, which stretches each residue to 5x5 and the
  // interleaved grid to 10x10 -- room for the output_padding.
  // CHECK-NOT: tensor.insert_slice
  // CHECK: hip.conv({{.*}} tensor<4x1x1x1xf32>)
  // CHECK-SAME: pads = [0, 0, 1, 1]
  // CHECK-SAME: tensor<1x4x5x5xf32>
  // CHECK-NOT: hip.conv(
  // CHECK-NOT: tensor.concat
  // CHECK: tensor.expand_shape {{.*}} {{\[}}[0], [1, 2, 3], [4], [5]] {{.*}} into tensor<1x2x2x1x5x5xf32>
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 3, 4, 1, 5, 2]
  // CHECK: tensor.extract_slice {{.*}}[0, 0, 0, 0] [1, 1, 9, 9] [1, 1, 1, 1]
  // CHECK-NOT: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 14. Growing the grid is bounded by ONNX requiring output_padding < stride,
  //     which is at most one extra position per axis. A result larger than
  //     that disagrees with the attributes, so the grid must not be sized off
  //     it -- here a 20x20 result against a 4x4 input at stride 2.
  // --------------------------------------------------------------------------
  func.func @result_too_large(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x1x20x20xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x1x3x3xf32>}
        : tensor<1x1x3x3xf32>
    %init = tensor.empty() : tensor<1x1x20x20xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x1x3x3xf32>)
        outs(%init : tensor<1x1x20x20xf32>)
        {kernel_shape = [3, 3], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x1x20x20xf32>
    return %y : tensor<1x1x20x20xf32>
  }

  // CHECK-LABEL: func.func @result_too_large
  // CHECK: hip.conv_transpose

  // --------------------------------------------------------------------------
  // 15. A 1-D transpose with dynamic width expands to a rank-4 [N, C, 1, W]
  // view, decomposes through 2-D hip.conv residues, and collapses the
  // artificial height dimension. The temporary width comes from tensor.dim.
  // --------------------------------------------------------------------------
  func.func @dynamic_1d(%ctx: !hip.context, %x: tensor<?x2x?xf32>,
                        %init: tensor<?x3x?xf32>)
      -> tensor<?x3x?xf32> {
    %w = hip.constant {value = dense<1.000000e+00> : tensor<2x3x16xf32>}
        : tensor<2x3x16xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<?x2x?xf32>,
                                              tensor<2x3x16xf32>)
        outs(%init : tensor<?x3x?xf32>)
        {kernel_shape = [16], strides = [8], pads = [4, 4],
         dilations = [1], output_padding = [0], group = 1 : i64}
        : tensor<?x3x?xf32>
    return %y : tensor<?x3x?xf32>
  }

  // CHECK-LABEL: func.func @dynamic_1d
  // CHECK-NOT: hip.conv_transpose
  // CHECK: tensor.expand_shape {{.*}} : tensor<?x2x?xf32> into tensor<?x2x1x?xf32>
  // CHECK: tensor.dim {{.*}}, %{{.*}} : tensor<?x2x?xf32>
  // CHECK-COUNT-8: hip.conv
  // CHECK: kernel_shape = [1, 2]
  // CHECK: tensor.collapse_shape {{.*}} : tensor<?x3x1x?xf32> into tensor<?x3x?xf32>

  // --------------------------------------------------------------------------
  // 16. Equal residues with several channels each way and distinct values:
  //     the wide filter must be residue-major, then output channel, then
  //     input channel, with each residue's taps reversed. A 4x2 kernel at
  //     stride 2 gives four residues of 2x1 taps, and
  //     w[c][m][y0][y1] = 24c + 8m + 2y0 + y1, so wide output channel
  //     (2 * it0 + it1) * 3 + m holds, for each c, the taps
  //     w[c][m][it0 + 2][it1] then w[c][m][it0][it1]. Residue (0, 0), m = 0
  //     reads [[4], [0]] for c = 0 and [[28], [24]] for c = 1; a
  //     channel-major packing, a missing [C, M] swap, or a missing reversal
  //     each changes those values.
  // --------------------------------------------------------------------------
  func.func @equal_residue_values(%ctx: !hip.context, %x: tensor<1x2x3x3xf32>)
      -> tensor<1x3x8x6xf32> {
    %w = hip.constant {value = dense<[[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0]], [[16.0, 17.0], [18.0, 19.0], [20.0, 21.0], [22.0, 23.0]]], [[[24.0, 25.0], [26.0, 27.0], [28.0, 29.0], [30.0, 31.0]], [[32.0, 33.0], [34.0, 35.0], [36.0, 37.0], [38.0, 39.0]], [[40.0, 41.0], [42.0, 43.0], [44.0, 45.0], [46.0, 47.0]]]]>
        : tensor<2x3x4x2xf32>} : tensor<2x3x4x2xf32>
    %init = tensor.empty() : tensor<1x3x8x6xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x2x3x3xf32>,
                                               tensor<2x3x4x2xf32>)
        outs(%init : tensor<1x3x8x6xf32>)
        {kernel_shape = [4, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x3x8x6xf32>
    return %y : tensor<1x3x8x6xf32>
  }

  // CHECK-LABEL: func.func @equal_residue_values
  // CHECK-NOT: hip.conv_transpose
  // CHECK: hip.constant {value = dense<{{\[}}
  // CHECK-SAME: {{\[}}{{\[}}[4.000000e+00], [0.000000e+00]], {{\[}}[2.800000e+01], [2.400000e+01]]], {{\[}}{{\[}}[1.200000e+01], [8.000000e+00]], {{\[}}[3.600000e+01], [3.200000e+01]]], {{\[}}{{\[}}[2.000000e+01], [1.600000e+01]], {{\[}}[4.400000e+01], [4.000000e+01]]],
  // CHECK-SAME: {{\[}}{{\[}}[5.000000e+00], [1.000000e+00]], {{\[}}[2.900000e+01], [2.500000e+01]]], {{\[}}{{\[}}[1.300000e+01], [9.000000e+00]], {{\[}}[3.700000e+01], [3.300000e+01]]], {{\[}}{{\[}}[2.100000e+01], [1.700000e+01]], {{\[}}[4.500000e+01], [4.100000e+01]]],
  // CHECK-SAME: {{\[}}{{\[}}[6.000000e+00], [2.000000e+00]], {{\[}}[3.000000e+01], [2.600000e+01]]], {{\[}}{{\[}}[1.400000e+01], [1.000000e+01]], {{\[}}[3.800000e+01], [3.400000e+01]]], {{\[}}{{\[}}[2.200000e+01], [1.800000e+01]], {{\[}}[4.600000e+01], [4.200000e+01]]],
  // CHECK-SAME: {{\[}}{{\[}}[7.000000e+00], [3.000000e+00]], {{\[}}[3.100000e+01], [2.700000e+01]]], {{\[}}{{\[}}[1.500000e+01], [1.100000e+01]], {{\[}}[3.900000e+01], [3.500000e+01]]], {{\[}}{{\[}}[2.300000e+01], [1.900000e+01]], {{\[}}[4.700000e+01], [4.300000e+01]]]]>
  // CHECK-SAME: : tensor<12x2x2x1xf32>
  // CHECK: hip.conv({{.*}} tensor<12x2x2x1xf32>)
  // CHECK-SAME: kernel_shape = [2, 1], pads = [1, 0, 1, 0]
  // CHECK-SAME: tensor<1x12x4x3xf32>
  // CHECK: tensor.expand_shape {{.*}} into tensor<1x2x2x3x4x3xf32>
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 3, 4, 1, 5, 2]
  // CHECK-SAME: tensor<1x3x4x2x3x2xf32>
  // CHECK: tensor.collapse_shape {{.*}} into tensor<1x3x8x6xf32>
  // CHECK-NOT: tensor.insert_slice

  // --------------------------------------------------------------------------
  // 17. A 5x5 filter at stride 4 keeps 2 or 1 taps per axis: 25 real taps,
  //     64 once padded, past the 2x bound. The 16 residues stay separate
  //     convolutions with strided inserts, and the run-time bias goes onto
  //     each of them: the residues tile the grid, so no add follows.
  // --------------------------------------------------------------------------
  func.func @unequal_residues(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>,
                              %b: tensor<3xf32>) -> tensor<1x3x17x17xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<2x3x5x5xf32>}
        : tensor<2x3x5x5xf32>
    %init = tensor.empty() : tensor<1x3x17x17xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w, %b : tensor<1x2x4x4xf32>,
                                                   tensor<2x3x5x5xf32>,
                                                   tensor<3xf32>)
        outs(%init : tensor<1x3x17x17xf32>)
        {kernel_shape = [5, 5], strides = [4, 4], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x3x17x17xf32>
    return %y : tensor<1x3x17x17xf32>
  }

  // CHECK-LABEL: func.func @unequal_residues
  // CHECK-NOT: hip.conv_transpose
  // CHECK: hip.conv({{.*}} tensor<3x2x2x2xf32>, tensor<3xf32>)
  // CHECK-COUNT-16: tensor.insert_slice
  // CHECK-NOT: hip.add
  // CHECK-NOT: hip.transpose
  // CHECK: return

  // --------------------------------------------------------------------------
  // 18. A run-time bias cannot be repeated per residue at compile time, so on
  //     the wide convolution it stays one broadcast add after the crop.
  // --------------------------------------------------------------------------
  func.func @runtime_bias_equal(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>,
                                %b: tensor<2xf32>) -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
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

  // CHECK-LABEL: func.func @runtime_bias_equal
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>)
  // CHECK: hip.transpose
  // CHECK: tensor.expand_shape {{.*}} into tensor<1x2x1x1xf32>
  // CHECK: hip.add
  // CHECK-NOT: hip.add

  // --------------------------------------------------------------------------
  // 19. The exporter's form of a bias: a bias-free ConvTranspose, then an add
  //     of a [1,1,1,M] constant reshaped to [1,M,1,1]. The add folds into the
  //     wide convolution's bias like case 8, and the activation behind it
  //     now follows the crop directly.
  // --------------------------------------------------------------------------
  func.func @add_reshaped_bias(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %c = hip.constant {value = dense<[[[[3.0, 4.0]]]]> : tensor<1x1x1x2xf32>}
        : tensor<1x1x1x2xf32>
    %flat = tensor.collapse_shape %c [[0, 1, 2, 3]]
        : tensor<1x1x1x2xf32> into tensor<2xf32>
    %cb = tensor.expand_shape %flat [[0, 1, 2, 3]] output_shape [1, 2, 1, 1]
        : tensor<2xf32> into tensor<1x2x1x1xf32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x2x2x2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    %ae = tensor.empty() : tensor<1x2x8x8xf32>
    %a = hip.add(%ctx) ins(%y, %cb : tensor<1x2x8x8xf32>, tensor<1x2x1x1xf32>)
        outs(%ae : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    %le = tensor.empty() : tensor<1x2x8x8xf32>
    %l = hip.leaky_relu(%ctx) ins(%a : tensor<1x2x8x8xf32>)
        outs(%le : tensor<1x2x8x8xf32>) {alpha = 1.000000e-01 : f64}
        : tensor<1x2x8x8xf32>
    return %l : tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @add_reshaped_bias
  // CHECK-NOT: hip.conv_transpose
  // CHECK: hip.constant {value = dense<[3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<8xf32>}
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>, tensor<8xf32>)
  // With no padding the interleave is the whole result, so no crop follows.
  // CHECK: %[[GRID:.*]] = tensor.collapse_shape
  // CHECK-NOT: hip.add
  // CHECK: hip.leaky_relu({{.*}}) ins(%[[GRID]] : tensor<1x2x8x8xf32>)
  // CHECK: return

  // --------------------------------------------------------------------------
  // 20. With a bias of its own, the added constant is summed into it.
  // --------------------------------------------------------------------------
  func.func @add_and_own_bias(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %b = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>}
        : tensor<2xf32>
    %c = hip.constant {value = dense<[[[[10.0]], [[20.0]]]]> : tensor<1x2x1x1xf32>}
        : tensor<1x2x1x1xf32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w, %b : tensor<1x1x4x4xf32>,
                                                   tensor<1x2x2x2xf32>,
                                                   tensor<2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    %ae = tensor.empty() : tensor<1x2x8x8xf32>
    %a = hip.add(%ctx) ins(%c, %y : tensor<1x2x1x1xf32>, tensor<1x2x8x8xf32>)
        outs(%ae : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    return %a : tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @add_and_own_bias
  // CHECK: hip.constant {value = dense<[1.100000e+01, 2.200000e+01, 1.100000e+01, 2.200000e+01, 1.100000e+01, 2.200000e+01, 1.100000e+01, 2.200000e+01]> : tensor<8xf32>}
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>, tensor<8xf32>)
  // CHECK-NOT: hip.add
  // CHECK: return

  // --------------------------------------------------------------------------
  // 21. A scalar broadcasts to every channel, so it folds as a uniform bias.
  // --------------------------------------------------------------------------
  func.func @add_scalar(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %c = hip.constant {value = dense<5.0> : tensor<f32>} : tensor<f32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x2x2x2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    %ae = tensor.empty() : tensor<1x2x8x8xf32>
    %a = hip.add(%ctx) ins(%y, %c : tensor<1x2x8x8xf32>, tensor<f32>)
        outs(%ae : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    return %a : tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @add_scalar
  // CHECK: hip.constant {value = dense<5.000000e+00> : tensor<8xf32>}
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>, tensor<8xf32>)
  // CHECK-NOT: hip.add
  // CHECK: return

  // --------------------------------------------------------------------------
  // 22. A transpose that only moves unit dimensions keeps the value order.
  // --------------------------------------------------------------------------
  func.func @add_transposed_bias(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %c = hip.constant {value = dense<[[[[3.0, 4.0]]]]> : tensor<1x1x1x2xf32>}
        : tensor<1x1x1x2xf32>
    %te = tensor.empty() : tensor<1x2x1x1xf32>
    %ct = hip.transpose(%ctx) ins(%c : tensor<1x1x1x2xf32>)
        outs(%te : tensor<1x2x1x1xf32>) {perm = [0, 3, 1, 2]}
        : tensor<1x2x1x1xf32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x2x2x2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    %ae = tensor.empty() : tensor<1x2x8x8xf32>
    %a = hip.add(%ctx) ins(%y, %ct : tensor<1x2x8x8xf32>, tensor<1x2x1x1xf32>)
        outs(%ae : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    return %a : tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @add_transposed_bias
  // CHECK: hip.constant {value = dense<[3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<8xf32>}
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>, tensor<8xf32>)
  // CHECK-NOT: hip.add
  // CHECK: return

  // --------------------------------------------------------------------------
  // 23. On the unequal-residue path (case 17) the folded bias goes onto each
  //     residue convolution.
  // --------------------------------------------------------------------------
  func.func @add_unequal_residues(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
      -> tensor<1x3x17x17xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<2x3x5x5xf32>}
        : tensor<2x3x5x5xf32>
    %c = hip.constant
        {value = dense<[[[[1.0]], [[2.0]], [[3.0]]]]> : tensor<1x3x1x1xf32>}
        : tensor<1x3x1x1xf32>
    %init = tensor.empty() : tensor<1x3x17x17xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x2x4x4xf32>,
                                               tensor<2x3x5x5xf32>)
        outs(%init : tensor<1x3x17x17xf32>)
        {kernel_shape = [5, 5], strides = [4, 4], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x3x17x17xf32>
    %ae = tensor.empty() : tensor<1x3x17x17xf32>
    %a = hip.add(%ctx) ins(%y, %c : tensor<1x3x17x17xf32>, tensor<1x3x1x1xf32>)
        outs(%ae : tensor<1x3x17x17xf32>) -> tensor<1x3x17x17xf32>
    return %a : tensor<1x3x17x17xf32>
  }

  // CHECK-LABEL: func.func @add_unequal_residues
  // CHECK: hip.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>}
  // CHECK: hip.conv({{.*}} tensor<3x2x2x2xf32>, tensor<3xf32>)
  // CHECK-COUNT-16: tensor.insert_slice
  // CHECK-NOT: hip.add
  // CHECK: return

  // --------------------------------------------------------------------------
  // 24. A rank-1 [M] operand right-aligns onto the width, not the channels:
  //     here W == M == 2, so the add is legal but is not a bias.
  // --------------------------------------------------------------------------
  func.func @add_rank1_no_fold(%ctx: !hip.context, %x: tensor<1x1x1x1xf32>)
      -> tensor<1x2x2x2xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %c = hip.constant {value = dense<[3.0, 4.0]> : tensor<2xf32>}
        : tensor<2xf32>
    %init = tensor.empty() : tensor<1x2x2x2xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x1x1xf32>,
                                               tensor<1x2x2x2xf32>)
        outs(%init : tensor<1x2x2x2xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x2x2xf32>
    %ae = tensor.empty() : tensor<1x2x2x2xf32>
    %a = hip.add(%ctx) ins(%y, %c : tensor<1x2x2x2xf32>, tensor<2xf32>)
        outs(%ae : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %a : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @add_rank1_no_fold
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>)
  // CHECK: hip.add
  // CHECK: return

  // --------------------------------------------------------------------------
  // 25. A constant with a value per pixel is not a per-channel bias.
  // --------------------------------------------------------------------------
  func.func @add_full_size_no_fold(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %c = hip.constant {value = dense<1.0> : tensor<1x2x8x8xf32>}
        : tensor<1x2x8x8xf32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x2x2x2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    %ae = tensor.empty() : tensor<1x2x8x8xf32>
    %a = hip.add(%ctx) ins(%y, %c : tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>)
        outs(%ae : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    return %a : tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @add_full_size_no_fold
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>)
  // CHECK: hip.add
  // CHECK: return

  // --------------------------------------------------------------------------
  // 26. A run-time operand has no values to fold.
  // --------------------------------------------------------------------------
  func.func @add_runtime_no_fold(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>,
                                 %c: tensor<1x2x1x1xf32>)
      -> tensor<1x2x8x8xf32> {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x2x2x2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    %ae = tensor.empty() : tensor<1x2x8x8xf32>
    %a = hip.add(%ctx) ins(%y, %c : tensor<1x2x8x8xf32>, tensor<1x2x1x1xf32>)
        outs(%ae : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    return %a : tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @add_runtime_no_fold
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>)
  // CHECK: hip.add
  // CHECK: return

  // --------------------------------------------------------------------------
  // 27. Another reader needs the result without the bias.
  // --------------------------------------------------------------------------
  func.func @add_two_users_no_fold(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> (tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>) {
    %w = hip.constant {value = dense<1.000000e-02> : tensor<1x2x2x2xf32>}
        : tensor<1x2x2x2xf32>
    %c = hip.constant {value = dense<[[[[3.0]], [[4.0]]]]> : tensor<1x2x1x1xf32>}
        : tensor<1x2x1x1xf32>
    %init = tensor.empty() : tensor<1x2x8x8xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                               tensor<1x2x2x2xf32>)
        outs(%init : tensor<1x2x8x8xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x2x8x8xf32>
    %ae = tensor.empty() : tensor<1x2x8x8xf32>
    %a = hip.add(%ctx) ins(%y, %c : tensor<1x2x8x8xf32>, tensor<1x2x1x1xf32>)
        outs(%ae : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    return %a, %y : tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>
  }

  // CHECK-LABEL: func.func @add_two_users_no_fold
  // CHECK: hip.conv({{.*}} tensor<8x1x1x1xf32>)
  // CHECK: hip.add
  // CHECK: return
}
