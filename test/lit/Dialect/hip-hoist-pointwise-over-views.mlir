// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// hip-hoist-pointwise-over-views moves a pointwise op whose other inputs are
// constant scalars above the single-use views between it and a gemm or
// convolution anchor. The views here are the DepthToSpace interleave and crop
// after a decomposed ConvTranspose. Once moved, the op runs in the anchor's
// epilogue: one kernel holds the conv, the op and the transpose.
//
// 1. leaky_relu over the interleave and crop: moved, then fused (the fusion
//    passes run on main_graph only).
// 2. max with a scalar, then leaky_relu: both move, in order.
// 3. A per-channel add is not a scalar operand: stays.
// 4. The crop has a second user: stays.
// 5. The crop keeps too little of the grid: stays.
// 6. The views start at a function argument, not an anchor: stays.
// ============================================================================

// RUN: hip-mlir-opt --split-input-file --hip-hoist-pointwise-over-views %s \
// RUN:   | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-hoist-pointwise-over-views \
// RUN:   --hip-rocmlir-prepare-quant --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-fuse-rocmlir-anchor %s | FileCheck %s --check-prefix=FUSE

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[L:.*]] = hip.leaky_relu({{.*}}) ins(%[[C]] : tensor<1x32x9x9xf16>)
// CHECK: tensor.expand_shape %[[L]]
// CHECK: hip.transpose
// CHECK: tensor.collapse_shape
// CHECK: %[[R:.*]] = tensor.extract_slice
// CHECK-NOT: hip.leaky_relu
// CHECK: return %[[R]]

// FUSE-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// FUSE: hip.conv
// FUSE: hip.leaky_relu
// FUSE: hip.transpose
// FUSE-LABEL: func.func @main_graph
// FUSE: hip.rocmlir
// FUSE-NOT: hip.rocmlir
// FUSE-NOT: hip.leaky_relu
// FUSE: return
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x8x8xf16>)
    -> tensor<1x8x16x16xf16> {
  %w = hip.constant {value = dense<1.000000e+00> : tensor<32x8x2x2xf16>}
      : tensor<32x8x2x2xf16>
  %e = tensor.empty() : tensor<1x32x9x9xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x8x8x8xf16>, tensor<32x8x2x2xf16>)
      outs(%e : tensor<1x32x9x9xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x32x9x9xf16>
  %s = tensor.expand_shape %c [[0], [1, 2, 3], [4], [5]]
      output_shape [1, 2, 2, 8, 9, 9]
      : tensor<1x32x9x9xf16> into tensor<1x2x2x8x9x9xf16>
  %te = tensor.empty() : tensor<1x8x9x2x9x2xf16>
  %t = hip.transpose(%ctx) ins(%s : tensor<1x2x2x8x9x9xf16>)
      outs(%te : tensor<1x8x9x2x9x2xf16>) {perm = [0, 3, 4, 1, 5, 2]}
      : tensor<1x8x9x2x9x2xf16>
  %g = tensor.collapse_shape %t [[0], [1], [2, 3], [4, 5]]
      : tensor<1x8x9x2x9x2xf16> into tensor<1x8x18x18xf16>
  %r = tensor.extract_slice %g[0, 0, 1, 1] [1, 8, 16, 16] [1, 1, 1, 1]
      : tensor<1x8x18x18xf16> to tensor<1x8x16x16xf16>
  %le = tensor.empty() : tensor<1x8x16x16xf16>
  %l = hip.leaky_relu(%ctx) ins(%r : tensor<1x8x16x16xf16>)
      outs(%le : tensor<1x8x16x16xf16>) {alpha = 1.000000e-01 : f64}
      : tensor<1x8x16x16xf16>
  return %l : tensor<1x8x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @scalar_then_unary
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[M:.*]] = hip.max({{.*}}) ins(%[[C]], %{{.*}} : tensor<1x32x9x9xf16>, tensor<f16>)
// CHECK: %[[L:.*]] = hip.leaky_relu({{.*}}) ins(%[[M]] : tensor<1x32x9x9xf16>)
// CHECK: tensor.expand_shape %[[L]]
// CHECK-NOT: hip.max
// CHECK-NOT: hip.leaky_relu
// CHECK: return
func.func @scalar_then_unary(%ctx: !hip.context, %x: tensor<1x8x8x8xf16>)
    -> tensor<1x8x16x16xf16> {
  %w = hip.constant {value = dense<1.000000e+00> : tensor<32x8x2x2xf16>}
      : tensor<32x8x2x2xf16>
  %z = hip.constant {value = dense<0.000000e+00> : tensor<f16>} : tensor<f16>
  %e = tensor.empty() : tensor<1x32x9x9xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x8x8x8xf16>, tensor<32x8x2x2xf16>)
      outs(%e : tensor<1x32x9x9xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x32x9x9xf16>
  %s = tensor.expand_shape %c [[0], [1, 2, 3], [4], [5]]
      output_shape [1, 2, 2, 8, 9, 9]
      : tensor<1x32x9x9xf16> into tensor<1x2x2x8x9x9xf16>
  %te = tensor.empty() : tensor<1x8x9x2x9x2xf16>
  %t = hip.transpose(%ctx) ins(%s : tensor<1x2x2x8x9x9xf16>)
      outs(%te : tensor<1x8x9x2x9x2xf16>) {perm = [0, 3, 4, 1, 5, 2]}
      : tensor<1x8x9x2x9x2xf16>
  %g = tensor.collapse_shape %t [[0], [1], [2, 3], [4, 5]]
      : tensor<1x8x9x2x9x2xf16> into tensor<1x8x18x18xf16>
  %r = tensor.extract_slice %g[0, 0, 1, 1] [1, 8, 16, 16] [1, 1, 1, 1]
      : tensor<1x8x18x18xf16> to tensor<1x8x16x16xf16>
  %me = tensor.empty() : tensor<1x8x16x16xf16>
  %m = hip.max(%ctx) ins(%r, %z : tensor<1x8x16x16xf16>, tensor<f16>)
      outs(%me : tensor<1x8x16x16xf16>) : tensor<1x8x16x16xf16>
  %le = tensor.empty() : tensor<1x8x16x16xf16>
  %l = hip.leaky_relu(%ctx) ins(%m : tensor<1x8x16x16xf16>)
      outs(%le : tensor<1x8x16x16xf16>) {alpha = 1.000000e-01 : f64}
      : tensor<1x8x16x16xf16>
  return %l : tensor<1x8x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @per_channel_add
// CHECK: tensor.extract_slice
// CHECK: hip.add
func.func @per_channel_add(%ctx: !hip.context, %x: tensor<1x8x8x8xf16>,
                           %b: tensor<1x8x1x1xf16>) -> tensor<1x8x16x16xf16> {
  %w = hip.constant {value = dense<1.000000e+00> : tensor<32x8x2x2xf16>}
      : tensor<32x8x2x2xf16>
  %e = tensor.empty() : tensor<1x32x9x9xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x8x8x8xf16>, tensor<32x8x2x2xf16>)
      outs(%e : tensor<1x32x9x9xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x32x9x9xf16>
  %s = tensor.expand_shape %c [[0], [1, 2, 3], [4], [5]]
      output_shape [1, 2, 2, 8, 9, 9]
      : tensor<1x32x9x9xf16> into tensor<1x2x2x8x9x9xf16>
  %te = tensor.empty() : tensor<1x8x9x2x9x2xf16>
  %t = hip.transpose(%ctx) ins(%s : tensor<1x2x2x8x9x9xf16>)
      outs(%te : tensor<1x8x9x2x9x2xf16>) {perm = [0, 3, 4, 1, 5, 2]}
      : tensor<1x8x9x2x9x2xf16>
  %g = tensor.collapse_shape %t [[0], [1], [2, 3], [4, 5]]
      : tensor<1x8x9x2x9x2xf16> into tensor<1x8x18x18xf16>
  %r = tensor.extract_slice %g[0, 0, 1, 1] [1, 8, 16, 16] [1, 1, 1, 1]
      : tensor<1x8x18x18xf16> to tensor<1x8x16x16xf16>
  %ae = tensor.empty() : tensor<1x8x16x16xf16>
  %a = hip.add(%ctx) ins(%r, %b : tensor<1x8x16x16xf16>, tensor<1x8x1x1xf16>)
      outs(%ae : tensor<1x8x16x16xf16>) -> tensor<1x8x16x16xf16>
  return %a : tensor<1x8x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @second_user
// CHECK: tensor.extract_slice
// CHECK: hip.leaky_relu
func.func @second_user(%ctx: !hip.context, %x: tensor<1x8x8x8xf16>)
    -> (tensor<1x8x16x16xf16>, tensor<1x8x16x16xf16>) {
  %w = hip.constant {value = dense<1.000000e+00> : tensor<32x8x2x2xf16>}
      : tensor<32x8x2x2xf16>
  %e = tensor.empty() : tensor<1x32x9x9xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x8x8x8xf16>, tensor<32x8x2x2xf16>)
      outs(%e : tensor<1x32x9x9xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x32x9x9xf16>
  %s = tensor.expand_shape %c [[0], [1, 2, 3], [4], [5]]
      output_shape [1, 2, 2, 8, 9, 9]
      : tensor<1x32x9x9xf16> into tensor<1x2x2x8x9x9xf16>
  %te = tensor.empty() : tensor<1x8x9x2x9x2xf16>
  %t = hip.transpose(%ctx) ins(%s : tensor<1x2x2x8x9x9xf16>)
      outs(%te : tensor<1x8x9x2x9x2xf16>) {perm = [0, 3, 4, 1, 5, 2]}
      : tensor<1x8x9x2x9x2xf16>
  %g = tensor.collapse_shape %t [[0], [1], [2, 3], [4, 5]]
      : tensor<1x8x9x2x9x2xf16> into tensor<1x8x18x18xf16>
  %r = tensor.extract_slice %g[0, 0, 1, 1] [1, 8, 16, 16] [1, 1, 1, 1]
      : tensor<1x8x18x18xf16> to tensor<1x8x16x16xf16>
  %le = tensor.empty() : tensor<1x8x16x16xf16>
  %l = hip.leaky_relu(%ctx) ins(%r : tensor<1x8x16x16xf16>)
      outs(%le : tensor<1x8x16x16xf16>) {alpha = 1.000000e-01 : f64}
      : tensor<1x8x16x16xf16>
  return %l, %r : tensor<1x8x16x16xf16>, tensor<1x8x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @small_crop
// CHECK: tensor.extract_slice
// CHECK: hip.leaky_relu({{.*}}) ins(%{{.*}} : tensor<1x8x8x8xf16>)
func.func @small_crop(%ctx: !hip.context, %x: tensor<1x8x8x8xf16>)
    -> tensor<1x8x8x8xf16> {
  %w = hip.constant {value = dense<1.000000e+00> : tensor<32x8x2x2xf16>}
      : tensor<32x8x2x2xf16>
  %e = tensor.empty() : tensor<1x32x9x9xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x8x8x8xf16>, tensor<32x8x2x2xf16>)
      outs(%e : tensor<1x32x9x9xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x32x9x9xf16>
  %s = tensor.expand_shape %c [[0], [1, 2, 3], [4], [5]]
      output_shape [1, 2, 2, 8, 9, 9]
      : tensor<1x32x9x9xf16> into tensor<1x2x2x8x9x9xf16>
  %te = tensor.empty() : tensor<1x8x9x2x9x2xf16>
  %t = hip.transpose(%ctx) ins(%s : tensor<1x2x2x8x9x9xf16>)
      outs(%te : tensor<1x8x9x2x9x2xf16>) {perm = [0, 3, 4, 1, 5, 2]}
      : tensor<1x8x9x2x9x2xf16>
  %g = tensor.collapse_shape %t [[0], [1], [2, 3], [4, 5]]
      : tensor<1x8x9x2x9x2xf16> into tensor<1x8x18x18xf16>
  %r = tensor.extract_slice %g[0, 0, 1, 1] [1, 8, 8, 8] [1, 1, 1, 1]
      : tensor<1x8x18x18xf16> to tensor<1x8x8x8xf16>
  %le = tensor.empty() : tensor<1x8x8x8xf16>
  %l = hip.leaky_relu(%ctx) ins(%r : tensor<1x8x8x8xf16>)
      outs(%le : tensor<1x8x8x8xf16>) {alpha = 1.000000e-01 : f64}
      : tensor<1x8x8x8xf16>
  return %l : tensor<1x8x8x8xf16>
}

// -----

// CHECK-LABEL: func.func @no_anchor
// CHECK: hip.transpose
// CHECK: hip.leaky_relu({{.*}}) ins(%{{.*}} : tensor<1x8x16x16xf16>)
func.func @no_anchor(%ctx: !hip.context, %x: tensor<1x16x8x16xf16>)
    -> tensor<1x8x16x16xf16> {
  %te = tensor.empty() : tensor<1x8x16x16xf16>
  %t = hip.transpose(%ctx) ins(%x : tensor<1x16x8x16xf16>)
      outs(%te : tensor<1x8x16x16xf16>) {perm = [0, 2, 1, 3]}
      : tensor<1x8x16x16xf16>
  %le = tensor.empty() : tensor<1x8x16x16xf16>
  %l = hip.leaky_relu(%ctx) ins(%t : tensor<1x8x16x16xf16>)
      outs(%le : tensor<1x8x16x16xf16>) {alpha = 1.000000e-01 : f64}
      : tensor<1x8x16x16xf16>
  return %l : tensor<1x8x16x16xf16>
}
