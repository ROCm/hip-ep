// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A Qwen3.5 gated-delta-net layer gates its per-head normalized attention
// output with SiLU(z) in fp32 before the output projection.
// hip-fusion-transform folds the chain into hip.gated_rms_norm.
// Shapes: 2 heads of 8.

// CHECK-LABEL: func.func @per_head
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<1x?x16xf16>, %[[Z:[^:]*]]: tensor<1x?x16xf16>, %[[W:[^:]*]]: tensor<8xf16>)
// CHECK-NOT: hip.rms_norm
// CHECK-NOT: hip.cast
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.mul
// CHECK: %[[T:.*]] = tensor.dim %[[X]], %c1
// CHECK: %[[O:.*]] = tensor.empty(%[[T]]) : tensor<1x?x16xf16>
// CHECK: %[[Y:.*]] = hip.gated_rms_norm(%[[CTX]]) ins(%[[X]], %[[Z]], %[[W]] : tensor<1x?x16xf16>, tensor<1x?x16xf16>, tensor<8xf16>)
// CHECK-SAME: outs(%[[O]] : tensor<1x?x16xf16>) {epsilon = 9.99999997E-7 : f32}
// CHECK-NOT: hip.
// CHECK: return %[[Y]]
func.func @per_head(%ctx: !hip.context, %x: tensor<1x?x16xf16>,
    %z: tensor<1x?x16xf16>, %w: tensor<8xf16>) -> tensor<1x?x16xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %x, %c1 : tensor<1x?x16xf16>
  %xe = tensor.expand_shape %x [[0], [1], [2, 3]] output_shape [1, %t, 2, 8] : tensor<1x?x16xf16> into tensor<1x?x2x8xf16>
  %en = tensor.empty(%t) : tensor<1x?x2x8xf16>
  %n = hip.rms_norm(%ctx) ins(%xe, %w : tensor<1x?x2x8xf16>, tensor<8xf16>) outs(%en : tensor<1x?x2x8xf16>)
      {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64} : tensor<1x?x2x8xf16>
  %nc = tensor.collapse_shape %n [[0], [1], [2, 3]] : tensor<1x?x2x8xf16> into tensor<1x?x16xf16>
  %tn = tensor.dim %nc, %c1 : tensor<1x?x16xf16>
  %enf = tensor.empty(%tn) : tensor<1x?x16xf32>
  %nf = hip.cast(%ctx) ins(%nc : tensor<1x?x16xf16>) outs(%enf : tensor<1x?x16xf32>) {to = 1 : i64} : tensor<1x?x16xf32>

  %ezf = tensor.empty(%t) : tensor<1x?x16xf32>
  %zf = hip.cast(%ctx) ins(%z : tensor<1x?x16xf16>) outs(%ezf : tensor<1x?x16xf32>) {to = 1 : i64} : tensor<1x?x16xf32>
  %es = tensor.empty(%t) : tensor<1x?x16xf32>
  %s = hip.sigmoid(%ctx) ins(%zf : tensor<1x?x16xf32>) outs(%es : tensor<1x?x16xf32>) : tensor<1x?x16xf32>
  %eg = tensor.empty(%t) : tensor<1x?x16xf32>
  %g = hip.mul(%ctx) ins(%zf, %s : tensor<1x?x16xf32>, tensor<1x?x16xf32>) outs(%eg : tensor<1x?x16xf32>) -> tensor<1x?x16xf32>

  %tp = tensor.dim %nf, %c1 : tensor<1x?x16xf32>
  %ep = tensor.empty(%tp) : tensor<1x?x16xf32>
  %p = hip.mul(%ctx) ins(%nf, %g : tensor<1x?x16xf32>, tensor<1x?x16xf32>) outs(%ep : tensor<1x?x16xf32>) -> tensor<1x?x16xf32>
  %ty = tensor.dim %p, %c1 : tensor<1x?x16xf32>
  %ey = tensor.empty(%ty) : tensor<1x?x16xf16>
  %y = hip.cast(%ctx) ins(%p : tensor<1x?x16xf32>) outs(%ey : tensor<1x?x16xf16>) {to = 10 : i64} : tensor<1x?x16xf16>
  return %y : tensor<1x?x16xf16>
}

// -----

// A norm over the whole last axis, gated through hip.swiglu.
// CHECK-LABEL: func.func @whole_row_swiglu
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<2x16xf16>, %[[Z:[^:]*]]: tensor<2x16xf16>, %[[W:[^:]*]]: tensor<16xf16>)
// CHECK: %[[O:.*]] = tensor.empty() : tensor<2x16xf16>
// CHECK: %[[Y:.*]] = hip.gated_rms_norm(%[[CTX]]) ins(%[[X]], %[[Z]], %[[W]] : tensor<2x16xf16>, tensor<2x16xf16>, tensor<16xf16>)
// CHECK-SAME: outs(%[[O]] : tensor<2x16xf16>) {epsilon = 9.99999974E-6 : f32}
// CHECK-NOT: hip.
// CHECK: return %[[Y]]
func.func @whole_row_swiglu(%ctx: !hip.context, %x: tensor<2x16xf16>,
    %z: tensor<2x16xf16>, %w: tensor<16xf16>) -> tensor<2x16xf16> {
  %en = tensor.empty() : tensor<2x16xf16>
  %n = hip.rms_norm(%ctx) ins(%x, %w : tensor<2x16xf16>, tensor<16xf16>) outs(%en : tensor<2x16xf16>)
      {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64} : tensor<2x16xf16>
  %enf = tensor.empty() : tensor<2x16xf32>
  %nf = hip.cast(%ctx) ins(%n : tensor<2x16xf16>) outs(%enf : tensor<2x16xf32>) {to = 1 : i64} : tensor<2x16xf32>
  %ezf = tensor.empty() : tensor<2x16xf32>
  %zf = hip.cast(%ctx) ins(%z : tensor<2x16xf16>) outs(%ezf : tensor<2x16xf32>) {to = 1 : i64} : tensor<2x16xf32>
  %ep = tensor.empty() : tensor<2x16xf32>
  %p = hip.swiglu(%ctx) ins(%zf, %nf : tensor<2x16xf32>, tensor<2x16xf32>) outs(%ep : tensor<2x16xf32>) : tensor<2x16xf32>
  %ey = tensor.empty() : tensor<2x16xf16>
  %y = hip.cast(%ctx) ins(%p : tensor<2x16xf32>) outs(%ey : tensor<2x16xf16>) {to = 10 : i64} : tensor<2x16xf16>
  return %y : tensor<2x16xf16>
}

// -----

// A gate that broadcasts against the norm is not this op.
// CHECK-LABEL: func.func @gate_broadcasts
// CHECK-NOT: hip.gated_rms_norm
// CHECK: hip.rms_norm
func.func @gate_broadcasts(%ctx: !hip.context, %x: tensor<2x16xf16>,
    %z: tensor<1x16xf16>, %w: tensor<16xf16>) -> tensor<2x16xf16> {
  %en = tensor.empty() : tensor<2x16xf16>
  %n = hip.rms_norm(%ctx) ins(%x, %w : tensor<2x16xf16>, tensor<16xf16>) outs(%en : tensor<2x16xf16>)
      {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64} : tensor<2x16xf16>
  %enf = tensor.empty() : tensor<2x16xf32>
  %nf = hip.cast(%ctx) ins(%n : tensor<2x16xf16>) outs(%enf : tensor<2x16xf32>) {to = 1 : i64} : tensor<2x16xf32>
  %ezf = tensor.empty() : tensor<1x16xf32>
  %zf = hip.cast(%ctx) ins(%z : tensor<1x16xf16>) outs(%ezf : tensor<1x16xf32>) {to = 1 : i64} : tensor<1x16xf32>
  %es = tensor.empty() : tensor<1x16xf32>
  %s = hip.sigmoid(%ctx) ins(%zf : tensor<1x16xf32>) outs(%es : tensor<1x16xf32>) : tensor<1x16xf32>
  %eg = tensor.empty() : tensor<1x16xf32>
  %g = hip.mul(%ctx) ins(%zf, %s : tensor<1x16xf32>, tensor<1x16xf32>) outs(%eg : tensor<1x16xf32>) -> tensor<1x16xf32>
  %ep = tensor.empty() : tensor<2x16xf32>
  %p = hip.mul(%ctx) ins(%nf, %g : tensor<2x16xf32>, tensor<1x16xf32>) outs(%ep : tensor<2x16xf32>) -> tensor<2x16xf32>
  %ey = tensor.empty() : tensor<2x16xf16>
  %y = hip.cast(%ctx) ins(%p : tensor<2x16xf32>) outs(%ey : tensor<2x16xf16>) {to = 10 : i64} : tensor<2x16xf16>
  return %y : tensor<2x16xf16>
}

// -----

// The SiLU is also read elsewhere, so the chain has to stay.
// CHECK-LABEL: func.func @silu_shared
// CHECK-NOT: hip.gated_rms_norm
// CHECK: hip.rms_norm
func.func @silu_shared(%ctx: !hip.context, %x: tensor<2x16xf16>,
    %z: tensor<2x16xf16>, %w: tensor<16xf16>)
    -> (tensor<2x16xf16>, tensor<2x16xf32>) {
  %en = tensor.empty() : tensor<2x16xf16>
  %n = hip.rms_norm(%ctx) ins(%x, %w : tensor<2x16xf16>, tensor<16xf16>) outs(%en : tensor<2x16xf16>)
      {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64} : tensor<2x16xf16>
  %enf = tensor.empty() : tensor<2x16xf32>
  %nf = hip.cast(%ctx) ins(%n : tensor<2x16xf16>) outs(%enf : tensor<2x16xf32>) {to = 1 : i64} : tensor<2x16xf32>
  %ezf = tensor.empty() : tensor<2x16xf32>
  %zf = hip.cast(%ctx) ins(%z : tensor<2x16xf16>) outs(%ezf : tensor<2x16xf32>) {to = 1 : i64} : tensor<2x16xf32>
  %es = tensor.empty() : tensor<2x16xf32>
  %s = hip.sigmoid(%ctx) ins(%zf : tensor<2x16xf32>) outs(%es : tensor<2x16xf32>) : tensor<2x16xf32>
  %eg = tensor.empty() : tensor<2x16xf32>
  %g = hip.mul(%ctx) ins(%zf, %s : tensor<2x16xf32>, tensor<2x16xf32>) outs(%eg : tensor<2x16xf32>) -> tensor<2x16xf32>
  %ep = tensor.empty() : tensor<2x16xf32>
  %p = hip.mul(%ctx) ins(%nf, %g : tensor<2x16xf32>, tensor<2x16xf32>) outs(%ep : tensor<2x16xf32>) -> tensor<2x16xf32>
  %ey = tensor.empty() : tensor<2x16xf16>
  %y = hip.cast(%ctx) ins(%p : tensor<2x16xf32>) outs(%ey : tensor<2x16xf16>) {to = 10 : i64} : tensor<2x16xf16>
  return %y, %g : tensor<2x16xf16>, tensor<2x16xf32>
}
