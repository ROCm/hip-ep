// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Vision encoders scale the attention scores, mask them to one video frame with
// an outer comparison of frame ids, and softmax the result:
//
//   s = mul(scores, scale); m = equal(ids[R,1], ids[1,C]);
//   w = where(m, s, fill); p = softmax(w)
//
// hip-fusion-transform turns that into one hip.scaled_masked_softmax that reads
// the id vectors instead of materializing the mask and the two score copies.

// CHECK-LABEL: func.func @canonical
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[S:.*]]: tensor<2x4x8xf16>, %[[R:.*]]: tensor<4xi64>, %[[C:.*]]: tensor<8xi64>)
// CHECK-NOT: hip.mul
// CHECK-NOT: hip.where
// CHECK-NOT: hip.equal
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.scaled_masked_softmax(%[[CTX]]) ins(%[[S]], %[[R]], %[[C]] : tensor<2x4x8xf16>, tensor<4xi64>, tensor<8xi64>)
// CHECK-SAME: fill = -6.550400e+04 : f32
// CHECK-SAME: scale = 1.250000e-01 : f32
func.func @canonical(%ctx: !hip.context, %scores: tensor<2x4x8xf16>,
                     %rows: tensor<4xi64>, %cols: tensor<8xi64>) -> tensor<2x4x8xf16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f16>} : tensor<f16>
  %fill = hip.constant {value = dense<-6.550400e+04> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<2x4x8xf16>
  %s = hip.mul(%ctx) ins(%scores, %scale : tensor<2x4x8xf16>, tensor<f16>)
       outs(%e0 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  %r = tensor.expand_shape %rows [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
  %c = tensor.expand_shape %cols [[0, 1]] output_shape [1, 8] : tensor<8xi64> into tensor<1x8xi64>
  %e1 = tensor.empty() : tensor<4x8xui8>
  %m = hip.equal(%ctx) ins(%r, %c : tensor<4x1xi64>, tensor<1x8xi64>)
       outs(%e1 : tensor<4x8xui8>) : tensor<4x8xui8>
  %e2 = tensor.empty() : tensor<2x4x8xf16>
  %w = hip.where(%ctx) ins(%m, %s, %fill : tensor<4x8xui8>, tensor<2x4x8xf16>, tensor<f16>)
       outs(%e2 : tensor<2x4x8xf16>) : tensor<2x4x8xf16>
  %e3 = tensor.empty() : tensor<2x4x8xf16>
  %p = hip.miopen.softmax(%ctx) ins(%w : tensor<2x4x8xf16>)
       outs(%e3 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  return %p : tensor<2x4x8xf16>
}

// -----

// The vision export: dynamic patch count, one id vector for both sides, the
// softmax init sized from the masked scores. The size query moves to the raw
// scores so the masked tensor is not kept alive.
// CHECK-LABEL: func.func @vision_dynamic
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[S:.*]]: tensor<16x?x?xf16>, %[[IDS:.*]]: tensor<?xi64>)
// CHECK-NOT: hip.where
// CHECK: tensor.dim %[[S]]
// CHECK: hip.scaled_masked_softmax(%[[CTX]]) ins(%[[S]], %[[IDS]], %[[IDS]] : tensor<16x?x?xf16>, tensor<?xi64>, tensor<?xi64>)
func.func @vision_dynamic(%ctx: !hip.context, %scores: tensor<16x?x?xf16>,
                          %ids: tensor<?xi64>) -> tensor<16x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %scale = hip.constant {value = dense<1.178590e-01> : tensor<f16>} : tensor<f16>
  %fill = hip.constant {value = dense<-6.550400e+04> : tensor<f16>} : tensor<f16>
  %d1 = tensor.dim %scores, %c1 : tensor<16x?x?xf16>
  %d2 = tensor.dim %scores, %c2 : tensor<16x?x?xf16>
  %e0 = tensor.empty(%d1, %d2) : tensor<16x?x?xf16>
  %s = hip.mul(%ctx) ins(%scores, %scale : tensor<16x?x?xf16>, tensor<f16>)
       outs(%e0 : tensor<16x?x?xf16>) -> tensor<16x?x?xf16>
  %n = tensor.dim %ids, %c0 : tensor<?xi64>
  %r = tensor.expand_shape %ids [[0, 1]] output_shape [%n, 1] : tensor<?xi64> into tensor<?x1xi64>
  %c = tensor.expand_shape %ids [[0, 1]] output_shape [1, %n] : tensor<?xi64> into tensor<1x?xi64>
  %e1 = tensor.empty(%n, %n) : tensor<?x?xui8>
  %m = hip.equal(%ctx) ins(%r, %c : tensor<?x1xi64>, tensor<1x?xi64>)
       outs(%e1 : tensor<?x?xui8>) : tensor<?x?xui8>
  %e2 = tensor.empty(%d1, %d2) : tensor<16x?x?xf16>
  %w = hip.where(%ctx) ins(%m, %s, %fill : tensor<?x?xui8>, tensor<16x?x?xf16>, tensor<f16>)
       outs(%e2 : tensor<16x?x?xf16>) : tensor<16x?x?xf16>
  %w1 = tensor.dim %w, %c1 : tensor<16x?x?xf16>
  %w2 = tensor.dim %w, %c2 : tensor<16x?x?xf16>
  %e3 = tensor.empty(%w1, %w2) : tensor<16x?x?xf16>
  %p = hip.miopen.softmax(%ctx) ins(%w : tensor<16x?x?xf16>)
       outs(%e3 : tensor<16x?x?xf16>) -> tensor<16x?x?xf16>
  return %p : tensor<16x?x?xf16>
}

// -----

// Equal operands swapped and the scale on the left of the multiply.
// CHECK-LABEL: func.func @commuted
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[S:.*]]: tensor<2x4x8xf32>, %[[R:.*]]: tensor<4xi64>, %[[C:.*]]: tensor<8xi64>)
// CHECK: hip.scaled_masked_softmax(%[[CTX]]) ins(%[[S]], %[[R]], %[[C]] : tensor<2x4x8xf32>, tensor<4xi64>, tensor<8xi64>)
func.func @commuted(%ctx: !hip.context, %scores: tensor<2x4x8xf32>,
                    %rows: tensor<4xi64>, %cols: tensor<8xi64>) -> tensor<2x4x8xf32> {
  %scale = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %fill = hip.constant {value = dense<-1.000000e+09> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : tensor<2x4x8xf32>
  %s = hip.mul(%ctx) ins(%scale, %scores : tensor<f32>, tensor<2x4x8xf32>)
       outs(%e0 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  %r = tensor.expand_shape %rows [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
  %c = tensor.expand_shape %cols [[0, 1]] output_shape [1, 8] : tensor<8xi64> into tensor<1x8xi64>
  %e1 = tensor.empty() : tensor<4x8xui8>
  %m = hip.equal(%ctx) ins(%c, %r : tensor<1x8xi64>, tensor<4x1xi64>)
       outs(%e1 : tensor<4x8xui8>) : tensor<4x8xui8>
  %e2 = tensor.empty() : tensor<2x4x8xf32>
  %w = hip.where(%ctx) ins(%m, %s, %fill : tensor<4x8xui8>, tensor<2x4x8xf32>, tensor<f32>)
       outs(%e2 : tensor<2x4x8xf32>) : tensor<2x4x8xf32>
  %e3 = tensor.empty() : tensor<2x4x8xf32>
  %p = hip.miopen.softmax(%ctx) ins(%w : tensor<2x4x8xf32>)
       outs(%e3 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %p : tensor<2x4x8xf32>
}

// -----

// A scale that is not a compile-time constant cannot become an attribute.
// CHECK-LABEL: func.func @decline_runtime_scale
// CHECK-NOT: hip.scaled_masked_softmax
// CHECK: hip.miopen.softmax
func.func @decline_runtime_scale(%ctx: !hip.context, %scores: tensor<2x4x8xf16>,
                                 %scale: tensor<f16>, %rows: tensor<4xi64>,
                                 %cols: tensor<8xi64>) -> tensor<2x4x8xf16> {
  %fill = hip.constant {value = dense<-6.550400e+04> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<2x4x8xf16>
  %s = hip.mul(%ctx) ins(%scores, %scale : tensor<2x4x8xf16>, tensor<f16>)
       outs(%e0 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  %r = tensor.expand_shape %rows [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
  %c = tensor.expand_shape %cols [[0, 1]] output_shape [1, 8] : tensor<8xi64> into tensor<1x8xi64>
  %e1 = tensor.empty() : tensor<4x8xui8>
  %m = hip.equal(%ctx) ins(%r, %c : tensor<4x1xi64>, tensor<1x8xi64>)
       outs(%e1 : tensor<4x8xui8>) : tensor<4x8xui8>
  %e2 = tensor.empty() : tensor<2x4x8xf16>
  %w = hip.where(%ctx) ins(%m, %s, %fill : tensor<4x8xui8>, tensor<2x4x8xf16>, tensor<f16>)
       outs(%e2 : tensor<2x4x8xf16>) : tensor<2x4x8xf16>
  %e3 = tensor.empty() : tensor<2x4x8xf16>
  %p = hip.miopen.softmax(%ctx) ins(%w : tensor<2x4x8xf16>)
       outs(%e3 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  return %p : tensor<2x4x8xf16>
}

// -----

// The scaled scores have a second reader, so they must still be produced.
// CHECK-LABEL: func.func @decline_shared_scaled
// CHECK-NOT: hip.scaled_masked_softmax
// CHECK: hip.miopen.softmax
func.func @decline_shared_scaled(%ctx: !hip.context, %scores: tensor<2x4x8xf16>,
                                 %rows: tensor<4xi64>, %cols: tensor<8xi64>)
    -> (tensor<2x4x8xf16>, tensor<2x4x8xf16>) {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f16>} : tensor<f16>
  %fill = hip.constant {value = dense<-6.550400e+04> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<2x4x8xf16>
  %s = hip.mul(%ctx) ins(%scores, %scale : tensor<2x4x8xf16>, tensor<f16>)
       outs(%e0 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  %r = tensor.expand_shape %rows [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
  %c = tensor.expand_shape %cols [[0, 1]] output_shape [1, 8] : tensor<8xi64> into tensor<1x8xi64>
  %e1 = tensor.empty() : tensor<4x8xui8>
  %m = hip.equal(%ctx) ins(%r, %c : tensor<4x1xi64>, tensor<1x8xi64>)
       outs(%e1 : tensor<4x8xui8>) : tensor<4x8xui8>
  %e2 = tensor.empty() : tensor<2x4x8xf16>
  %w = hip.where(%ctx) ins(%m, %s, %fill : tensor<4x8xui8>, tensor<2x4x8xf16>, tensor<f16>)
       outs(%e2 : tensor<2x4x8xf16>) : tensor<2x4x8xf16>
  %e3 = tensor.empty() : tensor<2x4x8xf16>
  %p = hip.miopen.softmax(%ctx) ins(%w : tensor<2x4x8xf16>)
       outs(%e3 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  return %p, %s : tensor<2x4x8xf16>, tensor<2x4x8xf16>
}

// -----

// A tensor fill is per-element, not a single fill value.
// CHECK-LABEL: func.func @decline_tensor_fill
// CHECK-NOT: hip.scaled_masked_softmax
// CHECK: hip.miopen.softmax
func.func @decline_tensor_fill(%ctx: !hip.context, %scores: tensor<2x4x8xf16>,
                               %fill: tensor<2x4x8xf16>, %rows: tensor<4xi64>,
                               %cols: tensor<8xi64>) -> tensor<2x4x8xf16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<2x4x8xf16>
  %s = hip.mul(%ctx) ins(%scores, %scale : tensor<2x4x8xf16>, tensor<f16>)
       outs(%e0 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  %r = tensor.expand_shape %rows [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
  %c = tensor.expand_shape %cols [[0, 1]] output_shape [1, 8] : tensor<8xi64> into tensor<1x8xi64>
  %e1 = tensor.empty() : tensor<4x8xui8>
  %m = hip.equal(%ctx) ins(%r, %c : tensor<4x1xi64>, tensor<1x8xi64>)
       outs(%e1 : tensor<4x8xui8>) : tensor<4x8xui8>
  %e2 = tensor.empty() : tensor<2x4x8xf16>
  %w = hip.where(%ctx) ins(%m, %s, %fill : tensor<4x8xui8>, tensor<2x4x8xf16>, tensor<2x4x8xf16>)
       outs(%e2 : tensor<2x4x8xf16>) : tensor<2x4x8xf16>
  %e3 = tensor.empty() : tensor<2x4x8xf16>
  %p = hip.miopen.softmax(%ctx) ins(%w : tensor<2x4x8xf16>)
       outs(%e3 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  return %p : tensor<2x4x8xf16>
}

// -----

// An arbitrary mask has no id-vector form.
// CHECK-LABEL: func.func @decline_arbitrary_mask
// CHECK-NOT: hip.scaled_masked_softmax
// CHECK: hip.miopen.softmax
func.func @decline_arbitrary_mask(%ctx: !hip.context, %scores: tensor<2x4x8xf16>,
                                  %a: tensor<4x8xi64>, %b: tensor<4x8xi64>)
    -> tensor<2x4x8xf16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f16>} : tensor<f16>
  %fill = hip.constant {value = dense<-6.550400e+04> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<2x4x8xf16>
  %s = hip.mul(%ctx) ins(%scores, %scale : tensor<2x4x8xf16>, tensor<f16>)
       outs(%e0 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  %e1 = tensor.empty() : tensor<4x8xui8>
  %m = hip.equal(%ctx) ins(%a, %b : tensor<4x8xi64>, tensor<4x8xi64>)
       outs(%e1 : tensor<4x8xui8>) : tensor<4x8xui8>
  %e2 = tensor.empty() : tensor<2x4x8xf16>
  %w = hip.where(%ctx) ins(%m, %s, %fill : tensor<4x8xui8>, tensor<2x4x8xf16>, tensor<f16>)
       outs(%e2 : tensor<2x4x8xf16>) : tensor<2x4x8xf16>
  %e3 = tensor.empty() : tensor<2x4x8xf16>
  %p = hip.miopen.softmax(%ctx) ins(%w : tensor<2x4x8xf16>)
       outs(%e3 : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
  return %p : tensor<2x4x8xf16>
}
