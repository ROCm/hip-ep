// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify an inference hip.batch_norm with constant parameters is folded into
// the filter and bias of the convolution(s) producing its input, and that the
// cases where the fold would be wrong or unreadable are left alone.
//
// Every positive case uses scale = [1, 2], B = [0.5, 1], mean = [1, 0],
// var = [3, 0], epsilon = 1, so s = scale / sqrt(var + eps) = [0.5, 2] and
// t = B - mean * s = [0, 1].
//
// Test cases:
// 1. conv_no_bias     - filter scaled per output channel, t becomes the bias
// 2. conv_with_bias   - existing bias b becomes s * b + t
// 3. add_conv_conv    - both filters scaled, t added to the first conv only
// 4. grouped          - grouped conv folds the same way (per output channel)
// 5. f16              - folded values are rounded to the filter type
// 6. multi_use        - conv result has a second consumer, BN survives
// 7. dynamic_param    - non-constant BN parameter, BN survives
// 8. memory_filter    - memory-address filter is never read, BN survives
// ============================================================================

// RUN: hip-mlir-opt %s --hip-fold-batchnorm --canonicalize | FileCheck %s

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
}
