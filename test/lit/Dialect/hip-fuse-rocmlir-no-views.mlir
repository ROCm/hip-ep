// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// fuse-views=false keeps the forward-only kernel boundaries: the layout
// transposes around an NHWC convolution stay in main_graph, and the kernel
// holds only the convolution.

// RUN: hip-mlir-opt --hip-fuse-rocmlir="fuse-views=false" %s \
// RUN:   | hip-mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<1x8x4x4xf16>, %{{[a-z0-9]+}}: tensor<16x8x3x3xf16>, %{{[a-z0-9]+}}: tensor<16xf16>) -> tensor<1x16x4x4xf16>
// CHECK-NOT: hip.transpose
// CHECK: hip.conv
// CHECK-NOT: hip.transpose
// CHECK: return

// CHECK-LABEL: func.func @main_graph
// CHECK-COUNT-2: hip.transpose
// CHECK: hip.rocmlir
// CHECK-COUNT-2: hip.transpose
// CHECK: hip.add
// CHECK: hip.transpose
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x4x8xf16>,
                      %w: tensor<16x3x3x8xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x16x4x4xf16>) -> tensor<1x4x4x16xf16> {
  %xe = tensor.empty() : tensor<1x8x4x4xf16>
  %xn = hip.transpose(%ctx) ins(%x : tensor<1x4x4x8xf16>)
      outs(%xe : tensor<1x8x4x4xf16>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf16>
  %we = tensor.empty() : tensor<16x8x3x3xf16>
  %wn = hip.transpose(%ctx) ins(%w : tensor<16x3x3x8xf16>)
      outs(%we : tensor<16x8x3x3xf16>) {perm = [0, 3, 1, 2]} : tensor<16x8x3x3xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%xn, %wn, %b : tensor<1x8x4x4xf16>,
                                         tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ce = tensor.empty() : tensor<1x4x4x16xf16>
  %cl = hip.transpose(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%ce : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]} : tensor<1x4x4x16xf16>
  %te = tensor.empty() : tensor<1x16x4x4xf16>
  %t = hip.transpose(%ctx) ins(%cl : tensor<1x4x4x16xf16>)
      outs(%te : tensor<1x16x4x4xf16>) {perm = [0, 3, 1, 2]} : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%t, %r : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %oe = tensor.empty() : tensor<1x4x4x16xf16>
  %o = hip.transpose(%ctx) ins(%a : tensor<1x16x4x4xf16>)
      outs(%oe : tensor<1x4x4x16xf16>) {perm = [0, 2, 3, 1]} : tensor<1x4x4x16xf16>
  return %o : tensor<1x4x4x16xf16>
}
