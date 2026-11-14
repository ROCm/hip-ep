// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// A bias-free ConvTranspose followed by Add(Reshape([1,1,1,M] -> [1,M,1,1]))
// and LeakyRelu, as exporters write a ConvTranspose with bias, ends in one
// kernel: hip-decompose-conv-transpose folds the add into the wide
// convolution's bias, hip-hoist-pointwise-over-views moves the leaky_relu
// above the interleave and crop, and hip-fuse-rocmlir-anchor takes it into
// the convolution's epilogue. The next 3x3 convolution then reads its input
// with no pointwise prologue, instead of recomputing the add and the
// leaky_relu for each of its nine taps.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-decompose-conv-transpose --canonicalize \
// RUN:   --hip-hoist-pointwise-over-views --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor \
// RUN:   | FileCheck %s

// The decomposed ConvTranspose: wide conv with the tiled bias, leaky_relu and
// the interleave in one kernel.
// CHECK: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[WIDE:.*]] = hip.conv({{.*}} tensor<32x8x2x2xf16>, tensor<32xf16>)
// CHECK-NOT: return
// CHECK: hip.leaky_relu({{.*}}) ins(%[[WIDE]] : tensor<1x32x9x9xf16>)
// CHECK-NOT: return
// CHECK: hip.transpose
// CHECK: return

// The next 3x3 convolution reads the interleave and crop as views, with no
// pointwise op of its own.
// CHECK: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-NOT: hip.add
// CHECK-NOT: hip.leaky_relu
// CHECK: tensor.extract_slice
// CHECK-NOT: hip.add
// CHECK-NOT: hip.leaky_relu
// CHECK: hip.conv({{.*}} tensor<8x8x3x3xf16>)
// CHECK: return

// CHECK-LABEL: func.func @main_graph
// CHECK-COUNT-2: hip.rocmlir
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.add
// CHECK-NOT: hip.leaky_relu
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x8x8xf16>)
    -> tensor<1x8x16x16xf16> {
  %w = hip.constant {value = dense<1.000000e-02> : tensor<8x8x4x4xf16>}
      : tensor<8x8x4x4xf16>
  %c = hip.constant {value = dense<5.000000e-01> : tensor<1x1x1x8xf16>}
      : tensor<1x1x1x8xf16>
  %flat = tensor.collapse_shape %c [[0, 1, 2, 3]]
      : tensor<1x1x1x8xf16> into tensor<8xf16>
  %cb = tensor.expand_shape %flat [[0, 1, 2, 3]] output_shape [1, 8, 1, 1]
      : tensor<8xf16> into tensor<1x8x1x1xf16>
  %init = tensor.empty() : tensor<1x8x16x16xf16>
  %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x8x8x8xf16>,
                                             tensor<8x8x4x4xf16>)
      outs(%init : tensor<1x8x16x16xf16>)
      {kernel_shape = [4, 4], strides = [2, 2], pads = [1, 1, 1, 1],
       dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
      : tensor<1x8x16x16xf16>
  %ae = tensor.empty() : tensor<1x8x16x16xf16>
  %a = hip.add(%ctx) ins(%y, %cb : tensor<1x8x16x16xf16>, tensor<1x8x1x1xf16>)
      outs(%ae : tensor<1x8x16x16xf16>) -> tensor<1x8x16x16xf16>
  %le = tensor.empty() : tensor<1x8x16x16xf16>
  %l = hip.leaky_relu(%ctx) ins(%a : tensor<1x8x16x16xf16>)
      outs(%le : tensor<1x8x16x16xf16>) {alpha = 1.000000e-01 : f64}
      : tensor<1x8x16x16xf16>
  %w2 = hip.constant {value = dense<1.000000e-02> : tensor<8x8x3x3xf16>}
      : tensor<8x8x3x3xf16>
  %oe = tensor.empty() : tensor<1x8x16x16xf16>
  %o = hip.conv(%ctx) ins(%l, %w2 : tensor<1x8x16x16xf16>,
                                    tensor<8x8x3x3xf16>)
      outs(%oe : tensor<1x8x16x16xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x8x16x16xf16>
  return %o : tensor<1x8x16x16xf16>
}
