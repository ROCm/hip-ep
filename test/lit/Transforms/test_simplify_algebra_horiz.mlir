// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Two matching convolutions of one activation merge even when a third
// convolution of that activation has a different kernel. The third stays.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-simplify-algebra | FileCheck %s

// CHECK-LABEL: func.func @siblings
// CHECK: hip.conv
// CHECK-SAME: tensor<8x4x2x2xf32>
// CHECK: tensor.extract_slice {{.*}}[0, 0, 0, 0] [1, 4, 4, 4]
// CHECK: tensor.extract_slice {{.*}}[0, 4, 0, 0] [1, 4, 4, 4]
// CHECK: hip.conv
// CHECK-SAME: tensor<2x4x3x3xf32>
// CHECK-NOT: hip.conv
func.func @siblings(%ctx: !hip.context, %x: tensor<1x4x4x4xf32>)
    -> (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x2x4x4xf32>) {
  %w0 = hip.constant {value = dense<1.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %w1 = hip.constant {value = dense<2.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %w2 = hip.constant {value = dense<3.000000e-02> : tensor<2x4x3x3xf32>}
      : tensor<2x4x3x3xf32>
  %e0 = tensor.empty() : tensor<1x4x4x4xf32>
  %e1 = tensor.empty() : tensor<1x4x4x4xf32>
  %e2 = tensor.empty() : tensor<1x2x4x4xf32>
  %c0 = hip.conv(%ctx) ins(%x, %w0 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
      outs(%e0 : tensor<1x4x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
  %c1 = hip.conv(%ctx) ins(%x, %w1 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
      outs(%e1 : tensor<1x4x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
  %c2 = hip.conv(%ctx) ins(%x, %w2 : tensor<1x4x4x4xf32>, tensor<2x4x3x3xf32>)
      outs(%e2 : tensor<1x2x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xf32>
  return %c0, %c1, %c2 : tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>, tensor<1x2x4x4xf32>
}
