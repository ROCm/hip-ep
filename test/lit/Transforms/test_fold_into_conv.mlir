// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a constant per-channel affine map applied to a convolution's output
// (inference hip.batch_norm, hip.mul / hip.add by a channel-broadcast
// constant) is folded into the filter and bias of the convolution(s)
// producing it, also with a dynamic batch, and that the cases where the fold
// would be wrong or unreadable are left alone.
//
// Every batch-norm case uses scale = [1, 2], B = [0.5, 1], mean = [1, 0],
// var = [3, 0], epsilon = 1, so s = scale / sqrt(var + eps) = [0.5, 2] and
// t = B - mean * s = [0, 1]. The filter is [2, 3] per output channel.
//
// Test cases:
// 1. conv_no_bias        - filter scaled per output channel, t becomes the bias
// 2. conv_with_bias      - existing bias b becomes s * b + t
// 3. add_conv_conv       - both filters scaled, t added to the first conv only
// 4. grouped             - grouped conv folds the same way (per output channel)
// 5. f16                 - folded values are rounded to the filter type
// 6. multi_use           - conv result has a second consumer, BN survives
// 7. dynamic_param       - non-constant BN parameter, BN survives
// 8. memory_filter       - memory-address filter is never read, BN survives
// 9. bn_dynamic_batch    - dynamic batch folds; tensor.dim users of the conv
//                          move to the folded conv
// 10. mul_layer_scale    - conv * [C,1,1] scales filter and bias
// 11. mul_add            - (conv * [1,C,1,1]) + [C,1,1] folds to one conv
// 12. mul_f16_dynamic    - f16, dynamic batch, constant on the lhs, rounding
// 13. mul_scalar         - a scalar scale applies to every channel
// 14. add_memory_filter  - a shift-only fold leaves the filter unread
// 15. mul_add_conv_conv  - a scale through an add of two convs scales both
// 16. mul_multi_use      - conv result has a second consumer, mul survives
// 17. mul_dynamic_scale  - non-constant scale, mul survives
// 18. mul_spatial        - [1,1,H,W] and [W] constants are not per channel
// 19. dynamic_channel    - dynamic channel dim, BN and mul survive
// 20. mul_f16_overflow   - folded filter would round to inf, mul survives
// 21. add_init_alias     - the add writes into a conv result, mul survives
// 22. add_same_conv      - add of one conv result to itself, mul survives
// 23. conv1d             - rank 3: [C,1] folds, [C] (along L) survives
// 24. add_lhs_constant   - t + conv, dynamic batch: t goes into the bias
// 25. mul_identity       - a scale of ones drops the mul, conv unchanged
// 26. bn_add_conv_conv_dynamic - dynamic batch through an add; tensor.dim
//                          users of both convs move to the folded convs
// 27. mul_mixed_type     - an f32 scale on an f16 conv is not folded
// ============================================================================

// RUN: hip-mlir-opt %s --hip-fold-into-conv --canonicalize | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. BN(conv), no conv bias.
  // --------------------------------------------------------------------------
  func.func @conv_no_bias(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i1 : tensor<1x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @conv_no_bias
  // CHECK-NOT: hip.batch_norm
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] :
  // CHECK-NOT: hip.batch_norm
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 2. BN(conv) with an existing bias b = [1, 1] -> s * b + t = [0.5, 3].
  // --------------------------------------------------------------------------
  func.func @conv_with_bias(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %cb = hip.constant {value = dense<[1.0, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w, %cb : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i1 : tensor<1x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @conv_with_bias
  // CHECK-NOT: hip.batch_norm
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[5.000000e-01, 3.000000e+00]> : tensor<2xf32>}
  // CHECK: hip.conv(%{{.*}}) ins(%{{.*}}, %{{.*}}, %[[B]] :
  // CHECK-NOT: hip.batch_norm

  // --------------------------------------------------------------------------
  // 3. BN(add(conv, conv)): both filters scaled by s; t lands on one bias only.
  // --------------------------------------------------------------------------
  func.func @add_conv_conv(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>,
                           %z: tensor<1x1x4x4xf32>) -> tensor<1x2x4x4xf32> {
    %wa = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %wb = hip.constant {value = dense<[[[[4.0]]], [[[5.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %ca = hip.conv(%ctx) ins(%x, %wa : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %cb = hip.conv(%ctx) ins(%z, %wb : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i2 = tensor.empty() : tensor<1x2x4x4xf32>
    %s = hip.add(%ctx) ins(%ca, %cb : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
        outs(%i2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %i3 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%s, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i3 : tensor<1x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @add_conv_conv
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<1x1x4x4xf32>, %[[Z:.*]]: tensor<1x1x4x4xf32>)
  // CHECK-NOT: hip.batch_norm
  // CHECK-DAG: %[[WA:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[WB:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[2.000000e+00]]], {{\[}}{{\[}}[1.000000e+01]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[T:.*]] = hip.constant {value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>}
  // CHECK-DAG: %[[CA:.*]] = hip.conv(%{{.*}}) ins(%[[X]], %[[WA]], %[[T]] :
  // CHECK-DAG: %[[CB:.*]] = hip.conv(%{{.*}}) ins(%[[Z]], %[[WB]] : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
  // CHECK: %[[S:.*]] = hip.add(%{{.*}}) ins(%[[CA]], %[[CB]] :
  // CHECK-NOT: hip.batch_norm
  // CHECK: return %[[S]]

  // --------------------------------------------------------------------------
  // 4. Grouped conv: the filter is still [M, C/group, ...], scaled per M.
  // --------------------------------------------------------------------------
  func.func @grouped(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 2 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i1 : tensor<1x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @grouped
  // CHECK-NOT: hip.batch_norm
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK: hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %{{.*}} :{{.*}}group = 2 : i64
  // CHECK-NOT: hip.batch_norm

  // --------------------------------------------------------------------------
  // 5. f16: the folded filter and bias are f16.
  // --------------------------------------------------------------------------
  func.func @f16(%ctx: !hip.context, %x: tensor<1x1x4x4xf16>)
      -> tensor<1x2x4x4xf16> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf16>} : tensor<2x1x1x1xf16>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf16>} : tensor<2xf16>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf16>} : tensor<2xf16>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf16>} : tensor<2xf16>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf16>} : tensor<2xf16>
    %i0 = tensor.empty() : tensor<1x2x4x4xf16>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf16>, tensor<2x1x1x1xf16>)
        outs(%i0 : tensor<1x2x4x4xf16>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf16>
    %i1 = tensor.empty() : tensor<1x2x4x4xf16>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf16>, tensor<2xf16>, tensor<2xf16>, tensor<2xf16>,
        tensor<2xf16>) outs(%i1 : tensor<1x2x4x4xf16>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf16>
    return %y : tensor<1x2x4x4xf16>
  }

  // CHECK-LABEL: func.func @f16
  // CHECK-NOT: hip.batch_norm
  // CHECK-DAG: hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf16>}
  // CHECK-DAG: hip.constant {value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf16>}
  // CHECK: hip.conv
  // CHECK-NOT: hip.batch_norm

  // --------------------------------------------------------------------------
  // 6. The conv result is also returned: rescaling it would change that
  //    consumer, so the batch norm stays.
  // --------------------------------------------------------------------------
  func.func @multi_use(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i1 : tensor<1x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf32>
    return %y, %c : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @multi_use
  // CHECK: hip.batch_norm

  // --------------------------------------------------------------------------
  // 7. A runtime BN parameter cannot be folded at compile time.
  // --------------------------------------------------------------------------
  func.func @dynamic_param(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>,
                           %scale: tensor<2xf32>) -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i1 : tensor<1x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @dynamic_param
  // CHECK: hip.batch_norm

  // --------------------------------------------------------------------------
  // 8. A memory-address filter is process-local; the pass must not
  //    dereference it.
  // --------------------------------------------------------------------------
  func.func @memory_filter(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {memory_address = 4096 : i64, size = 8 : i64} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i1 : tensor<1x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @memory_filter
  // CHECK: hip.constant {memory_address = 4096 : i64
  // CHECK: hip.batch_norm

  // --------------------------------------------------------------------------
  // 9. Dynamic batch: only the channel dim must be static. The conv result is
  //    also read by tensor.dim (for the BN init and as a returned size); the
  //    fold keeps every shape, so those users move to the folded conv.
  // --------------------------------------------------------------------------
  func.func @bn_dynamic_batch(%ctx: !hip.context, %x: tensor<?x1x4x4xf32>)
      -> (tensor<?x2x4x4xf32>, index) {
    %c0 = arith.constant 0 : index
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %n = tensor.dim %x, %c0 : tensor<?x1x4x4xf32>
    %i0 = tensor.empty(%n) : tensor<?x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<?x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<?x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<?x2x4x4xf32>
    %d = tensor.dim %c, %c0 : tensor<?x2x4x4xf32>
    %i1 = tensor.empty(%d) : tensor<?x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%c, %scale, %b, %mean, %var :
        tensor<?x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i1 : tensor<?x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<?x2x4x4xf32>
    return %y, %d : tensor<?x2x4x4xf32>, index
  }

  // CHECK-LABEL: func.func @bn_dynamic_batch
  // CHECK-NOT: hip.batch_norm
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] : tensor<?x1x4x4xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>) outs(%{{.*}} : tensor<?x2x4x4xf32>) {{.*}} : tensor<?x2x4x4xf32>
  // CHECK-NOT: hip.batch_norm
  // CHECK: return %[[Y]], %{{.*}} : tensor<?x2x4x4xf32>, index

  // --------------------------------------------------------------------------
  // 10. Layer scale: conv * s with s : [C, 1, 1] = [0.5, 2] scales the filter
  //     and the bias b = [1, 1] -> [0.5, 2].
  // --------------------------------------------------------------------------
  func.func @mul_layer_scale(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %cb = hip.constant {value = dense<[1.0, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %s = hip.constant {value = dense<[[[0.5]], [[2.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w, %cb : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_layer_scale
  // CHECK-NOT: hip.mul
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[5.000000e-01, 2.000000e+00]> : tensor<2xf32>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] :
  // CHECK-NOT: hip.mul
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 11. (conv * s) + t with s : [1, C, 1, 1] = [0.5, 2] and t : [C, 1, 1] =
  //     [1, -1]: the mul folds first, then the add into the same conv.
  // --------------------------------------------------------------------------
  func.func @mul_add(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %s = hip.constant {value = dense<[[[[0.5]], [[2.0]]]]> : tensor<1x2x1x1xf32>} : tensor<1x2x1x1xf32>
    %t = hip.constant {value = dense<[[[1.0]], [[-1.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %m = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf32>, tensor<1x2x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %i2 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.add(%ctx) ins(%m, %t : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_add
  // CHECK-NOT: hip.mul
  // CHECK-NOT: hip.add
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[1.000000e+00, -1.000000e+00]> : tensor<2xf32>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] :
  // CHECK-NOT: hip.mul
  // CHECK-NOT: hip.add
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 12. f16 with a dynamic batch and the constant as the lhs: s = [3, 0.1]
  //     gives [6, 3 * 0.0999755859375 = 0.2999267578125], which rounds to the
  //     f16 0.2998046875 (ties to even); the bias b = [1, 1] -> [3, 0.1].
  // --------------------------------------------------------------------------
  func.func @mul_f16_dynamic(%ctx: !hip.context, %x: tensor<?x1x4x4xf16>)
      -> tensor<?x2x4x4xf16> {
    %c0 = arith.constant 0 : index
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf16>} : tensor<2x1x1x1xf16>
    %cb = hip.constant {value = dense<[1.0, 1.0]> : tensor<2xf16>} : tensor<2xf16>
    %s = hip.constant {value = dense<[[[3.0]], [[0.1]]]> : tensor<2x1x1xf16>} : tensor<2x1x1xf16>
    %n = tensor.dim %x, %c0 : tensor<?x1x4x4xf16>
    %i0 = tensor.empty(%n) : tensor<?x2x4x4xf16>
    %c = hip.conv(%ctx) ins(%x, %w, %cb : tensor<?x1x4x4xf16>, tensor<2x1x1x1xf16>, tensor<2xf16>)
        outs(%i0 : tensor<?x2x4x4xf16>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<?x2x4x4xf16>
    %d = tensor.dim %c, %c0 : tensor<?x2x4x4xf16>
    %i1 = tensor.empty(%d) : tensor<?x2x4x4xf16>
    %y = hip.mul(%ctx) ins(%s, %c : tensor<2x1x1xf16>, tensor<?x2x4x4xf16>)
        outs(%i1 : tensor<?x2x4x4xf16>) -> tensor<?x2x4x4xf16>
    return %y : tensor<?x2x4x4xf16>
  }

  // CHECK-LABEL: func.func @mul_f16_dynamic
  // CHECK-NOT: hip.mul
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[6.000000e+00]]], {{\[}}{{\[}}[2.998050e-01]]]]> : tensor<2x1x1x1xf16>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[3.000000e+00, 9.997550e-02]> : tensor<2xf16>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] : tensor<?x1x4x4xf16>, tensor<2x1x1x1xf16>, tensor<2xf16>) outs(%{{.*}} : tensor<?x2x4x4xf16>)
  // CHECK-NOT: hip.mul
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 13. A scalar scale broadcasts to every channel; with no bias and no shift
  //     the conv gets no bias.
  // --------------------------------------------------------------------------
  func.func @mul_scalar(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf32>, tensor<f32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_scalar
  // CHECK-NOT: hip.mul
  // CHECK: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[1.500000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]] : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
  // CHECK-NOT: hip.mul
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 14. conv + t only moves t into the bias, so the filter is never read: the
  //     memory-address filter stays as it is.
  // --------------------------------------------------------------------------
  func.func @add_memory_filter(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {memory_address = 4096 : i64, size = 8 : i64} : tensor<2x1x1x1xf32>
    %t = hip.constant {value = dense<[[[[1.0]], [[2.0]]]]> : tensor<1x2x1x1xf32>} : tensor<1x2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.add(%ctx) ins(%c, %t : tensor<1x2x4x4xf32>, tensor<1x2x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @add_memory_filter
  // CHECK-NOT: hip.add
  // CHECK-DAG: %[[W:.*]] = hip.constant {memory_address = 4096 : i64
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] :
  // CHECK-NOT: hip.add
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 15. (conv_a + conv_b) * s: both filters and conv_b's bias b = [1, 1] are
  //     scaled; with no shift, conv_a gets no bias.
  // --------------------------------------------------------------------------
  func.func @mul_add_conv_conv(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>,
                               %z: tensor<1x1x4x4xf32>) -> tensor<1x2x4x4xf32> {
    %wa = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %wb = hip.constant {value = dense<[[[[4.0]]], [[[5.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %cb = hip.constant {value = dense<[1.0, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %s = hip.constant {value = dense<[[[0.5]], [[2.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %ca = hip.conv(%ctx) ins(%x, %wa : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %cbv = hip.conv(%ctx) ins(%z, %wb, %cb : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i2 = tensor.empty() : tensor<1x2x4x4xf32>
    %sum = hip.add(%ctx) ins(%ca, %cbv : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
        outs(%i2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %i3 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%sum, %s : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i3 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_add_conv_conv
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<1x1x4x4xf32>, %[[Z:.*]]: tensor<1x1x4x4xf32>)
  // CHECK-NOT: hip.mul
  // CHECK-DAG: %[[WA:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[WB:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[2.000000e+00]]], {{\[}}{{\[}}[1.000000e+01]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[5.000000e-01, 2.000000e+00]> : tensor<2xf32>}
  // CHECK-DAG: %[[CA:.*]] = hip.conv(%{{.*}}) ins(%[[X]], %[[WA]] : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
  // CHECK-DAG: %[[CB:.*]] = hip.conv(%{{.*}}) ins(%[[Z]], %[[WB]], %[[B]] :
  // CHECK: %[[S:.*]] = hip.add(%{{.*}}) ins(%[[CA]], %[[CB]] :
  // CHECK-NOT: hip.mul
  // CHECK: return %[[S]]

  // --------------------------------------------------------------------------
  // 16. The conv result is also returned, so the mul stays.
  // --------------------------------------------------------------------------
  func.func @mul_multi_use(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %s = hip.constant {value = dense<[[[0.5]], [[2.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y, %c : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_multi_use
  // CHECK: hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[2.000000e+00]]], {{\[}}{{\[}}[3.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK: hip.mul

  // --------------------------------------------------------------------------
  // 17. A runtime scale cannot be folded at compile time.
  // --------------------------------------------------------------------------
  func.func @mul_dynamic_scale(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>,
                               %s: tensor<2x1x1xf32>) -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_dynamic_scale
  // CHECK: hip.mul

  // --------------------------------------------------------------------------
  // 18. Constants that vary along H / W are not per channel: [1, 1, H, W]
  //     and [W] (which broadcasts along W, not C) both stay.
  // --------------------------------------------------------------------------
  func.func @mul_spatial(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %hw = hip.constant {value = dense<2.0> : tensor<1x1x4x4xf32>} : tensor<1x1x4x4xf32>
    %wv = hip.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %ca = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %cb = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i2 = tensor.empty() : tensor<1x2x4x4xf32>
    %ya = hip.mul(%ctx) ins(%ca, %hw : tensor<1x2x4x4xf32>, tensor<1x1x4x4xf32>)
        outs(%i2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %i3 = tensor.empty() : tensor<1x2x4x4xf32>
    %yb = hip.mul(%ctx) ins(%cb, %wv : tensor<1x2x4x4xf32>, tensor<4xf32>)
        outs(%i3 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %ya, %yb : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_spatial
  // CHECK-COUNT-2: hip.mul
  // CHECK: return

  // --------------------------------------------------------------------------
  // 19. A dynamic channel dim leaves no static per-channel layout to fold
  //     into: the batch norm and the mul both stay.
  // --------------------------------------------------------------------------
  func.func @dynamic_channel(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>,
                             %k: index)
      -> (tensor<1x?x4x4xf32>, tensor<1x?x4x4xf32>) {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %s = hip.constant {value = dense<[[[0.5]], [[2.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty(%k) : tensor<1x?x4x4xf32>
    %ca = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x?x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x?x4x4xf32>
    %i1 = tensor.empty(%k) : tensor<1x?x4x4xf32>
    %cb = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i1 : tensor<1x?x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x?x4x4xf32>
    %i2 = tensor.empty(%k) : tensor<1x?x4x4xf32>
    %ya = hip.batch_norm(%ctx) ins(%ca, %scale, %b, %mean, %var :
        tensor<1x?x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i2 : tensor<1x?x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<1x?x4x4xf32>
    %i3 = tensor.empty(%k) : tensor<1x?x4x4xf32>
    %yb = hip.mul(%ctx) ins(%cb, %s : tensor<1x?x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i3 : tensor<1x?x4x4xf32>) -> tensor<1x?x4x4xf32>
    return %ya, %yb : tensor<1x?x4x4xf32>, tensor<1x?x4x4xf32>
  }

  // CHECK-LABEL: func.func @dynamic_channel
  // CHECK: hip.batch_norm
  // CHECK: hip.mul

  // --------------------------------------------------------------------------
  // 20. 1000 * 100 overflows f16 (max 65504): folding would turn a finite
  //     filter into inf, so the mul stays.
  // --------------------------------------------------------------------------
  func.func @mul_f16_overflow(%ctx: !hip.context, %x: tensor<1x1x4x4xf16>)
      -> tensor<1x2x4x4xf16> {
    %w = hip.constant {value = dense<[[[[1000.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf16>} : tensor<2x1x1x1xf16>
    %s = hip.constant {value = dense<[[[100.0]], [[1.0]]]> : tensor<2x1x1xf16>} : tensor<2x1x1xf16>
    %i0 = tensor.empty() : tensor<1x2x4x4xf16>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf16>, tensor<2x1x1x1xf16>)
        outs(%i0 : tensor<1x2x4x4xf16>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf16>
    %i1 = tensor.empty() : tensor<1x2x4x4xf16>
    %y = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf16>, tensor<2x1x1xf16>)
        outs(%i1 : tensor<1x2x4x4xf16>) -> tensor<1x2x4x4xf16>
    return %y : tensor<1x2x4x4xf16>
  }

  // CHECK-LABEL: func.func @mul_f16_overflow
  // CHECK: hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+03]]], {{\[}}{{\[}}[3.000000e+00]]]]> : tensor<2x1x1x1xf16>}
  // CHECK: hip.mul

  // --------------------------------------------------------------------------
  // 21. The add writes into conv_a's result (its DPS init): rewriting conv_a
  //     would leave that use on the old value, so the mul stays.
  // --------------------------------------------------------------------------
  func.func @add_init_alias(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %s = hip.constant {value = dense<[[[0.5]], [[2.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %ca = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %cb = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %sum = hip.add(%ctx) ins(%ca, %cb : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
        outs(%ca : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %i2 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%sum, %s : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @add_init_alias
  // CHECK-NOT: dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]]
  // CHECK: hip.mul
  // CHECK: return

  // --------------------------------------------------------------------------
  // 22. conv + conv of one conv result: the add has no second conv to take
  //     the scale, so the mul stays.
  // --------------------------------------------------------------------------
  func.func @add_same_conv(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %s = hip.constant {value = dense<[[[0.5]], [[2.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %sum = hip.add(%ctx) ins(%c, %c : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %i2 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%sum, %s : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>)
        outs(%i2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @add_same_conv
  // CHECK: hip.mul
  // CHECK: return

  // --------------------------------------------------------------------------
  // 23. 1-D conv ([N, C, L]): [C, 1] is per channel and folds; [C] lines up
  //     with L, not C, and stays.
  // --------------------------------------------------------------------------
  func.func @conv1d(%ctx: !hip.context, %x: tensor<1x1x2xf32>)
      -> (tensor<1x2x2xf32>, tensor<1x2x2xf32>) {
    %w = hip.constant {value = dense<[[[2.0]], [[3.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %sc = hip.constant {value = dense<[[0.5], [2.0]]> : tensor<2x1xf32>} : tensor<2x1xf32>
    %sl = hip.constant {value = dense<[0.5, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %i0 = tensor.empty() : tensor<1x2x2xf32>
    %ca = hip.conv(%ctx) ins(%x, %w : tensor<1x1x2xf32>, tensor<2x1x1xf32>)
        outs(%i0 : tensor<1x2x2xf32>)
        {kernel_shape = [1], strides = [1], pads = [0, 0], dilations = [1],
         group = 1 : i64} : tensor<1x2x2xf32>
    %i1 = tensor.empty() : tensor<1x2x2xf32>
    %cb = hip.conv(%ctx) ins(%x, %w : tensor<1x1x2xf32>, tensor<2x1x1xf32>)
        outs(%i1 : tensor<1x2x2xf32>)
        {kernel_shape = [1], strides = [1], pads = [0, 0], dilations = [1],
         group = 1 : i64} : tensor<1x2x2xf32>
    %i2 = tensor.empty() : tensor<1x2x2xf32>
    %ya = hip.mul(%ctx) ins(%ca, %sc : tensor<1x2x2xf32>, tensor<2x1xf32>)
        outs(%i2 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %i3 = tensor.empty() : tensor<1x2x2xf32>
    %yb = hip.mul(%ctx) ins(%cb, %sl : tensor<1x2x2xf32>, tensor<2xf32>)
        outs(%i3 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    return %ya, %yb : tensor<1x2x2xf32>, tensor<1x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv1d
  // CHECK-DAG: %[[WA:.*]] = hip.constant {value = dense<{{\[}}{{\[}}[1.000000e+00]], {{\[}}[6.000000e+00]]]> : tensor<2x1x1xf32>}
  // CHECK-DAG: %[[YA:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[WA]] : tensor<1x1x2xf32>, tensor<2x1x1xf32>)
  // CHECK-DAG: %[[YB:.*]] = hip.mul
  // CHECK: return %[[YA]], %[[YB]]

  // --------------------------------------------------------------------------
  // 24. t + conv (constant on the lhs) with a dynamic batch: t goes into the
  //     bias; the filter is unchanged.
  // --------------------------------------------------------------------------
  func.func @add_lhs_constant(%ctx: !hip.context, %x: tensor<?x1x4x4xf16>)
      -> tensor<?x2x4x4xf16> {
    %c0 = arith.constant 0 : index
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf16>} : tensor<2x1x1x1xf16>
    %cb = hip.constant {value = dense<[1.0, 1.0]> : tensor<2xf16>} : tensor<2xf16>
    %t = hip.constant {value = dense<[[[0.5]], [[-2.0]]]> : tensor<2x1x1xf16>} : tensor<2x1x1xf16>
    %n = tensor.dim %x, %c0 : tensor<?x1x4x4xf16>
    %i0 = tensor.empty(%n) : tensor<?x2x4x4xf16>
    %c = hip.conv(%ctx) ins(%x, %w, %cb : tensor<?x1x4x4xf16>, tensor<2x1x1x1xf16>, tensor<2xf16>)
        outs(%i0 : tensor<?x2x4x4xf16>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<?x2x4x4xf16>
    %d = tensor.dim %c, %c0 : tensor<?x2x4x4xf16>
    %i1 = tensor.empty(%d) : tensor<?x2x4x4xf16>
    %y = hip.add(%ctx) ins(%t, %c : tensor<2x1x1xf16>, tensor<?x2x4x4xf16>)
        outs(%i1 : tensor<?x2x4x4xf16>) -> tensor<?x2x4x4xf16>
    return %y : tensor<?x2x4x4xf16>
  }

  // CHECK-LABEL: func.func @add_lhs_constant
  // CHECK-NOT: hip.add
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[2.000000e+00]]], {{\[}}{{\[}}[3.000000e+00]]]]> : tensor<2x1x1x1xf16>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[1.500000e+00, -1.000000e+00]> : tensor<2xf16>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] :
  // CHECK-NOT: hip.add
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 25. A scale of all ones: the mul goes and the conv is left as it is.
  // --------------------------------------------------------------------------
  func.func @mul_identity(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
      -> tensor<1x2x4x4xf32> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %s = hip.constant {value = dense<1.0> : tensor<1x2x1x1xf32>} : tensor<1x2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf32>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<1x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf32>
    %i1 = tensor.empty() : tensor<1x2x4x4xf32>
    %y = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf32>, tensor<1x2x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %y : tensor<1x2x4x4xf32>
  }

  // CHECK-LABEL: func.func @mul_identity
  // CHECK-NOT: hip.mul
  // CHECK: %[[W:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[2.000000e+00]]], {{\[}}{{\[}}[3.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK: %[[Y:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[W]] : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
  // CHECK-NOT: hip.mul
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // 26. BN of (conv_a + conv_b) with a dynamic batch, both convs also read by
  //     tensor.dim: those move to the folded convs.
  // --------------------------------------------------------------------------
  func.func @bn_add_conv_conv_dynamic(%ctx: !hip.context, %x: tensor<?x1x4x4xf32>)
      -> (tensor<?x2x4x4xf32>, index, index) {
    %c0 = arith.constant 0 : index
    %wa = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %wb = hip.constant {value = dense<[[[[4.0]]], [[[5.0]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
    %scale = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
    %b = hip.constant {value = dense<[0.5, 1.0]> : tensor<2xf32>} : tensor<2xf32>
    %mean = hip.constant {value = dense<[1.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %var = hip.constant {value = dense<[3.0, 0.0]> : tensor<2xf32>} : tensor<2xf32>
    %n = tensor.dim %x, %c0 : tensor<?x1x4x4xf32>
    %i0 = tensor.empty(%n) : tensor<?x2x4x4xf32>
    %ca = hip.conv(%ctx) ins(%x, %wa : tensor<?x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i0 : tensor<?x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<?x2x4x4xf32>
    %i1 = tensor.empty(%n) : tensor<?x2x4x4xf32>
    %cb = hip.conv(%ctx) ins(%x, %wb : tensor<?x1x4x4xf32>, tensor<2x1x1x1xf32>)
        outs(%i1 : tensor<?x2x4x4xf32>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<?x2x4x4xf32>
    %da = tensor.dim %ca, %c0 : tensor<?x2x4x4xf32>
    %db = tensor.dim %cb, %c0 : tensor<?x2x4x4xf32>
    %i2 = tensor.empty(%da) : tensor<?x2x4x4xf32>
    %s = hip.add(%ctx) ins(%ca, %cb : tensor<?x2x4x4xf32>, tensor<?x2x4x4xf32>)
        outs(%i2 : tensor<?x2x4x4xf32>) -> tensor<?x2x4x4xf32>
    %ds = tensor.dim %s, %c0 : tensor<?x2x4x4xf32>
    %i3 = tensor.empty(%ds) : tensor<?x2x4x4xf32>
    %y = hip.batch_norm(%ctx) ins(%s, %scale, %b, %mean, %var :
        tensor<?x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>) outs(%i3 : tensor<?x2x4x4xf32>)
        {epsilon = 1.0 : f32} : tensor<?x2x4x4xf32>
    return %y, %da, %db : tensor<?x2x4x4xf32>, index, index
  }

  // CHECK-LABEL: func.func @bn_add_conv_conv_dynamic
  // CHECK-NOT: hip.batch_norm
  // CHECK-DAG: %[[WA:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]]], {{\[}}{{\[}}[6.000000e+00]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[WB:.*]] = hip.constant {value = dense<{{\[}}{{\[}}{{\[}}[2.000000e+00]]], {{\[}}{{\[}}[1.000000e+01]]]]> : tensor<2x1x1x1xf32>}
  // CHECK-DAG: %[[CA:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[WA]], %{{.*}} : tensor<?x1x4x4xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>)
  // CHECK-DAG: %[[CB:.*]] = hip.conv(%{{.*}}) ins(%{{.*}}, %[[WB]] : tensor<?x1x4x4xf32>, tensor<2x1x1x1xf32>)
  // CHECK: %[[S:.*]] = hip.add(%{{.*}}) ins(%[[CA]], %[[CB]] :
  // CHECK-NOT: hip.batch_norm
  // CHECK: return %[[S]], %{{.*}}, %{{.*}} : tensor<?x2x4x4xf32>, index, index

  // --------------------------------------------------------------------------
  // 27. The scale's element type differs from the conv's: the pass does not
  //     guess the mixed-type mul's rounding, so it stays.
  // --------------------------------------------------------------------------
  func.func @mul_mixed_type(%ctx: !hip.context, %x: tensor<1x1x4x4xf16>)
      -> tensor<1x2x4x4xf16> {
    %w = hip.constant {value = dense<[[[[2.0]]], [[[3.0]]]]> : tensor<2x1x1x1xf16>} : tensor<2x1x1x1xf16>
    %s = hip.constant {value = dense<[[[0.5]], [[2.0]]]> : tensor<2x1x1xf32>} : tensor<2x1x1xf32>
    %i0 = tensor.empty() : tensor<1x2x4x4xf16>
    %c = hip.conv(%ctx) ins(%x, %w : tensor<1x1x4x4xf16>, tensor<2x1x1x1xf16>)
        outs(%i0 : tensor<1x2x4x4xf16>)
        {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
         dilations = [1, 1], group = 1 : i64} : tensor<1x2x4x4xf16>
    %i1 = tensor.empty() : tensor<1x2x4x4xf16>
    %y = hip.mul(%ctx) ins(%c, %s : tensor<1x2x4x4xf16>, tensor<2x1x1xf32>)
        outs(%i1 : tensor<1x2x4x4xf16>) -> tensor<1x2x4x4xf16>
    return %y : tensor<1x2x4x4xf16>
  }

  // CHECK-LABEL: func.func @mul_mixed_type
  // CHECK: hip.mul
}
