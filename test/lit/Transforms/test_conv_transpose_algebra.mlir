// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// A stride-2 4x4 transposed convolution has four equal 2x2 residues. Decompose
// emits them as one plain convolution whose output channels are 4 * M,
// followed by the DepthToSpace interleave. Fuse-rocmlir then outlines the
// convolution with that interleave, so the kernel stores the interleaved
// layout directly. There are no channel slices and no concat.
// ============================================================================

// RUN: hip-mlir-opt --split-input-file %s \
// RUN:   --hip-decompose-conv-transpose --hip-simplify-algebra --hip-fuse-rocmlir \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.conv
// CHECK-SAME: tensor<4x1x2x2xf32>
// CHECK-SAME: tensor<1x4x5x5xf32>
// CHECK: tensor.expand_shape {{.*}} into tensor<1x2x2x1x5x5xf32>
// CHECK: hip.transpose
// CHECK-SAME: perm = [0, 3, 4, 1, 5, 2]
// CHECK: return

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.conv_transpose
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor.concat
// CHECK-NOT: hip.transpose
// CHECK: hip.rocmlir
// CHECK-SAME: tensor<1x1x5x2x5x2xf32>
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract_slice {{.*}}[0, 0, 1, 1] [1, 1, 8, 8]
// CHECK-NOT: hip.conv({{.*}})
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x1x4x4xf32>)
    -> tensor<1x1x8x8xf32> {
  %w = hip.constant {value = dense<1.000000e-02> : tensor<1x1x4x4xf32>}
      : tensor<1x1x4x4xf32>
  %init = tensor.empty() : tensor<1x1x8x8xf32>
  %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x4x4xf32>,
                                             tensor<1x1x4x4xf32>)
      outs(%init : tensor<1x1x8x8xf32>)
      {kernel_shape = [4, 4], strides = [2, 2], pads = [1, 1, 1, 1],
       dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
      : tensor<1x1x8x8xf32>
  return %y : tensor<1x1x8x8xf32>
}
