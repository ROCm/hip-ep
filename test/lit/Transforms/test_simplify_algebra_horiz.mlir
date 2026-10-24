// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Matching sibling convolutions of one activation merge into one wide conv.
// A sibling with a different kernel stays, a use between two siblings still
// sees its slice, block-argument inits are kept, and siblings in different
// blocks do not merge.
// ============================================================================

// RUN: hip-mlir-opt %s --split-input-file --hip-simplify-algebra \
// RUN:   | FileCheck %s

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

// -----

// A use of the first sibling that sits before the second sibling still sees
// its slice: the merged conv is placed before the first sibling.
// CHECK-LABEL: func.func @interleaved_use
// CHECK: hip.conv
// CHECK-SAME: tensor<8x4x2x2xf32>
// CHECK: %[[S0:.*]] = tensor.extract_slice {{.*}}[0, 0, 0, 0] [1, 4, 4, 4]
// CHECK: %[[S1:.*]] = tensor.extract_slice {{.*}}[0, 4, 0, 0] [1, 4, 4, 4]
// CHECK: %[[I0:.*]] = tensor.insert_slice %[[S0]]
// CHECK: tensor.insert_slice %[[S1]] into %[[I0]]
// CHECK-NOT: hip.conv
func.func @interleaved_use(%ctx: !hip.context, %x: tensor<1x4x4x4xf32>,
                           %dest: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  %w0 = hip.constant {value = dense<1.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %w1 = hip.constant {value = dense<2.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %e0 = tensor.empty() : tensor<1x4x4x4xf32>
  %c0 = hip.conv(%ctx) ins(%x, %w0 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
      outs(%e0 : tensor<1x4x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
  %i0 = tensor.insert_slice %c0 into %dest[0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
      : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
  %e1 = tensor.empty() : tensor<1x4x4x4xf32>
  %c1 = hip.conv(%ctx) ins(%x, %w1 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
      outs(%e1 : tensor<1x4x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
  %i1 = tensor.insert_slice %c1 into %i0[0, 4, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
      : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
  return %i1 : tensor<1x8x4x4xf32>
}

// -----

// Inits that are block arguments are left alone when the siblings merge.
// CHECK-LABEL: func.func @block_arg_init
// CHECK: hip.conv
// CHECK-SAME: tensor<8x4x2x2xf32>
// CHECK-NOT: hip.conv
func.func @block_arg_init(%ctx: !hip.context, %x: tensor<1x4x4x4xf32>,
                          %o0: tensor<1x4x4x4xf32>, %o1: tensor<1x4x4x4xf32>)
    -> (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) {
  %w0 = hip.constant {value = dense<1.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %w1 = hip.constant {value = dense<2.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %c0 = hip.conv(%ctx) ins(%x, %w0 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
      outs(%o0 : tensor<1x4x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
  %c1 = hip.conv(%ctx) ins(%x, %w1 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
      outs(%o1 : tensor<1x4x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
  return %c0, %c1 : tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>
}

// -----

// Siblings in different blocks do not merge: neither branch dominates the
// other.
// CHECK-LABEL: func.func @different_blocks
// CHECK: scf.if
// CHECK: hip.conv
// CHECK-SAME: tensor<4x4x2x2xf32>
// CHECK: } else {
// CHECK: hip.conv
// CHECK-SAME: tensor<4x4x2x2xf32>
// CHECK-NOT: tensor.extract_slice
func.func @different_blocks(%ctx: !hip.context, %x: tensor<1x4x4x4xf32>,
                            %cond: i1) -> tensor<1x4x4x4xf32> {
  %w0 = hip.constant {value = dense<1.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %w1 = hip.constant {value = dense<2.000000e-02> : tensor<4x4x2x2xf32>}
      : tensor<4x4x2x2xf32>
  %r = scf.if %cond -> (tensor<1x4x4x4xf32>) {
    %e0 = tensor.empty() : tensor<1x4x4x4xf32>
    %c0 = hip.conv(%ctx) ins(%x, %w0 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
        outs(%e0 : tensor<1x4x4x4xf32>)
        {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
         pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
    scf.yield %c0 : tensor<1x4x4x4xf32>
  } else {
    %e1 = tensor.empty() : tensor<1x4x4x4xf32>
    %c1 = hip.conv(%ctx) ins(%x, %w1 : tensor<1x4x4x4xf32>, tensor<4x4x2x2xf32>)
        outs(%e1 : tensor<1x4x4x4xf32>)
        {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
         pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xf32>
    scf.yield %c1 : tensor<1x4x4x4xf32>
  }
  return %r : tensor<1x4x4x4xf32>
}

// -----

// Integer filters are not merged: the merge concatenates them as floats.
// CHECK-LABEL: func.func @integer_weights
// CHECK-COUNT-2: hip.conv
// CHECK-NOT: tensor.extract_slice
func.func @integer_weights(%ctx: !hip.context, %x: tensor<1x4x4x4xi32>)
    -> (tensor<1x4x4x4xi32>, tensor<1x4x4x4xi32>) {
  %w0 = hip.constant {value = dense<1> : tensor<4x4x2x2xi32>}
      : tensor<4x4x2x2xi32>
  %w1 = hip.constant {value = dense<2> : tensor<4x4x2x2xi32>}
      : tensor<4x4x2x2xi32>
  %e0 = tensor.empty() : tensor<1x4x4x4xi32>
  %e1 = tensor.empty() : tensor<1x4x4x4xi32>
  %c0 = hip.conv(%ctx) ins(%x, %w0 : tensor<1x4x4x4xi32>, tensor<4x4x2x2xi32>)
      outs(%e0 : tensor<1x4x4x4xi32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xi32>
  %c1 = hip.conv(%ctx) ins(%x, %w1 : tensor<1x4x4x4xi32>, tensor<4x4x2x2xi32>)
      outs(%e1 : tensor<1x4x4x4xi32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x4x4x4xi32>
  return %c0, %c1 : tensor<1x4x4x4xi32>, tensor<1x4x4x4xi32>
}

// -----

// A sibling whose filter constant is also its init still merges, and the
// constant is erased once, after the init is checked.
// CHECK-LABEL: func.func @weight_as_init
// CHECK: hip.conv
// CHECK-SAME: tensor<2x1x1x1xf32>
// CHECK-NOT: hip.conv
func.func @weight_as_init(%ctx: !hip.context, %x: tensor<1x1x1x1xf32>)
    -> (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) {
  %w0 = hip.constant {value = dense<1.000000e-02> : tensor<1x1x1x1xf32>}
      : tensor<1x1x1x1xf32>
  %w1 = hip.constant {value = dense<2.000000e-02> : tensor<1x1x1x1xf32>}
      : tensor<1x1x1x1xf32>
  %c0 = hip.conv(%ctx) ins(%x, %w0 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>)
      outs(%w0 : tensor<1x1x1x1xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x1x1x1xf32>
  %c1 = hip.conv(%ctx) ins(%x, %w1 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>)
      outs(%w1 : tensor<1x1x1x1xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x1x1x1xf32>
  return %c0, %c1 : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>
}
