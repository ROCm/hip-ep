// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A per-element op sandwiched between a transpose and its inverse computes the
// same values in either layout, so both transposes are removed and the op runs
// on the original tensor. Only hip.gelu qualifies for now.
//
// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Fires: [0, 2, 1] is its own inverse and the types round-trip.
// CHECK-LABEL: func.func @gelu
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x49x768xf32>) -> tensor<1x49x768xf32> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<1x49x768xf32>
// CHECK-NEXT:    %[[G:.*]] = hip.gelu(%[[CTX]]) ins(%[[X]] : tensor<1x49x768xf32>) outs(%[[INIT]] : tensor<1x49x768xf32>) {{.*}}: tensor<1x49x768xf32>
// CHECK-NEXT:    return %[[G]] : tensor<1x49x768xf32>
// CHECK-NEXT:  }
func.func @gelu(%ctx: !hip.context,
                %x: tensor<1x49x768xf32>) -> tensor<1x49x768xf32> {
  %e1 = tensor.empty() : tensor<1x768x49xf32>
  %t1 = hip.transpose(%ctx) ins(%x : tensor<1x49x768xf32>)
      outs(%e1 : tensor<1x768x49xf32>) {perm = [0, 2, 1]}
      : tensor<1x768x49xf32>
  %e2 = tensor.empty() : tensor<1x768x49xf32>
  %g = hip.gelu(%ctx) ins(%t1 : tensor<1x768x49xf32>)
      outs(%e2 : tensor<1x768x49xf32>) : tensor<1x768x49xf32>
  %e3 = tensor.empty() : tensor<1x49x768xf32>
  %t2 = hip.transpose(%ctx) ins(%g : tensor<1x768x49xf32>)
      outs(%e3 : tensor<1x49x768xf32>) {perm = [0, 2, 1]}
      : tensor<1x49x768xf32>
  return %t2 : tensor<1x49x768xf32>
}

// -----

// Fires: [1, 2, 0] is not its own inverse, so this only matches when the
// second perm is compared with the inverse of the first, here [2, 0, 1].
// CHECK-LABEL: func.func @three_cycle
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<2x3x4xf32>
// CHECK-NEXT:    %[[G:.*]] = hip.gelu(%[[CTX]]) ins(%[[X]] : tensor<2x3x4xf32>) outs(%[[INIT]] : tensor<2x3x4xf32>) {{.*}}: tensor<2x3x4xf32>
// CHECK-NEXT:    return %[[G]] : tensor<2x3x4xf32>
// CHECK-NEXT:  }
func.func @three_cycle(%ctx: !hip.context,
                       %x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %e1 = tensor.empty() : tensor<3x4x2xf32>
  %t1 = hip.transpose(%ctx) ins(%x : tensor<2x3x4xf32>)
      outs(%e1 : tensor<3x4x2xf32>) {perm = [1, 2, 0]}
      : tensor<3x4x2xf32>
  %e2 = tensor.empty() : tensor<3x4x2xf32>
  %g = hip.gelu(%ctx) ins(%t1 : tensor<3x4x2xf32>)
      outs(%e2 : tensor<3x4x2xf32>) : tensor<3x4x2xf32>
  %e3 = tensor.empty() : tensor<2x3x4xf32>
  %t2 = hip.transpose(%ctx) ins(%g : tensor<3x4x2xf32>)
      outs(%e3 : tensor<2x3x4xf32>) {perm = [2, 0, 1]}
      : tensor<2x3x4xf32>
  return %t2 : tensor<2x3x4xf32>
}

// -----

// Stays: the types round-trip on a cube, so only the perm comparison can
// reject [1, 2, 0] followed by [1, 2, 0], which is not its inverse.
// CHECK-LABEL: func.func @not_inverse
// CHECK:       hip.transpose
// CHECK:       hip.gelu
// CHECK:       hip.transpose
func.func @not_inverse(%ctx: !hip.context,
                       %x: tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
  %e1 = tensor.empty() : tensor<4x4x4xf32>
  %t1 = hip.transpose(%ctx) ins(%x : tensor<4x4x4xf32>)
      outs(%e1 : tensor<4x4x4xf32>) {perm = [1, 2, 0]}
      : tensor<4x4x4xf32>
  %e2 = tensor.empty() : tensor<4x4x4xf32>
  %g = hip.gelu(%ctx) ins(%t1 : tensor<4x4x4xf32>)
      outs(%e2 : tensor<4x4x4xf32>) : tensor<4x4x4xf32>
  %e3 = tensor.empty() : tensor<4x4x4xf32>
  %t2 = hip.transpose(%ctx) ins(%g : tensor<4x4x4xf32>)
      outs(%e3 : tensor<4x4x4xf32>) {perm = [1, 2, 0]}
      : tensor<4x4x4xf32>
  return %t2 : tensor<4x4x4xf32>
}

// -----

// Stays: the first transpose has a second user, so it would not go away.
// CHECK-LABEL: func.func @first_transpose_shared
// CHECK:       hip.transpose
// CHECK:       hip.gelu
// CHECK:       hip.transpose
func.func @first_transpose_shared(%ctx: !hip.context,
                                  %x: tensor<1x49x768xf32>)
    -> (tensor<1x49x768xf32>, tensor<1x768x49xf32>) {
  %e1 = tensor.empty() : tensor<1x768x49xf32>
  %t1 = hip.transpose(%ctx) ins(%x : tensor<1x49x768xf32>)
      outs(%e1 : tensor<1x768x49xf32>) {perm = [0, 2, 1]}
      : tensor<1x768x49xf32>
  %e2 = tensor.empty() : tensor<1x768x49xf32>
  %g = hip.gelu(%ctx) ins(%t1 : tensor<1x768x49xf32>)
      outs(%e2 : tensor<1x768x49xf32>) : tensor<1x768x49xf32>
  %e3 = tensor.empty() : tensor<1x49x768xf32>
  %t2 = hip.transpose(%ctx) ins(%g : tensor<1x768x49xf32>)
      outs(%e3 : tensor<1x49x768xf32>) {perm = [0, 2, 1]}
      : tensor<1x49x768xf32>
  return %t2, %t1 : tensor<1x49x768xf32>, tensor<1x768x49xf32>
}

// -----

// Stays: the gelu result has a second user that needs the transposed layout.
// CHECK-LABEL: func.func @gelu_shared
// CHECK:       hip.transpose
// CHECK:       hip.gelu
// CHECK:       hip.transpose
func.func @gelu_shared(%ctx: !hip.context,
                       %x: tensor<1x49x768xf32>)
    -> (tensor<1x49x768xf32>, tensor<1x768x49xf32>) {
  %e1 = tensor.empty() : tensor<1x768x49xf32>
  %t1 = hip.transpose(%ctx) ins(%x : tensor<1x49x768xf32>)
      outs(%e1 : tensor<1x768x49xf32>) {perm = [0, 2, 1]}
      : tensor<1x768x49xf32>
  %e2 = tensor.empty() : tensor<1x768x49xf32>
  %g = hip.gelu(%ctx) ins(%t1 : tensor<1x768x49xf32>)
      outs(%e2 : tensor<1x768x49xf32>) : tensor<1x768x49xf32>
  %e3 = tensor.empty() : tensor<1x49x768xf32>
  %t2 = hip.transpose(%ctx) ins(%g : tensor<1x768x49xf32>)
      outs(%e3 : tensor<1x49x768xf32>) {perm = [0, 2, 1]}
      : tensor<1x49x768xf32>
  return %t2, %g : tensor<1x49x768xf32>, tensor<1x768x49xf32>
}

// -----

// Stays: sigmoid is per-element and would be safe, but is not supported yet.
// CHECK-LABEL: func.func @sigmoid_not_supported
// CHECK:       hip.transpose
// CHECK:       hip.sigmoid
// CHECK:       hip.transpose
func.func @sigmoid_not_supported(%ctx: !hip.context,
                                 %x: tensor<1x49x768xf32>)
    -> tensor<1x49x768xf32> {
  %e1 = tensor.empty() : tensor<1x768x49xf32>
  %t1 = hip.transpose(%ctx) ins(%x : tensor<1x49x768xf32>)
      outs(%e1 : tensor<1x768x49xf32>) {perm = [0, 2, 1]}
      : tensor<1x768x49xf32>
  %e2 = tensor.empty() : tensor<1x768x49xf32>
  %g = hip.sigmoid(%ctx) ins(%t1 : tensor<1x768x49xf32>)
      outs(%e2 : tensor<1x768x49xf32>) : tensor<1x768x49xf32>
  %e3 = tensor.empty() : tensor<1x49x768xf32>
  %t2 = hip.transpose(%ctx) ins(%g : tensor<1x768x49xf32>)
      outs(%e3 : tensor<1x49x768xf32>) {perm = [0, 2, 1]}
      : tensor<1x49x768xf32>
  return %t2 : tensor<1x49x768xf32>
}
