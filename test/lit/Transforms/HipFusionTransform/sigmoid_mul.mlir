// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Gated attention scales its fp16 output by the sigmoid of a gate
// projection. hip-fusion-transform folds the pair into hip.sigmoid_mul.

// CHECK-LABEL: func.func @output_gate
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<1x?x64xf16>, %[[G:[^:]*]]: tensor<1x?x64xf16>)
// CHECK-NOT: hip.sigmoid(
// CHECK: %[[T:.*]] = tensor.dim %[[G]], %c1
// CHECK: %[[E:.*]] = tensor.empty(%[[T]]) : tensor<1x?x64xf16>
// CHECK: %[[Y:.*]] = hip.sigmoid_mul(%[[CTX]]) ins(%[[X]], %[[G]] : tensor<1x?x64xf16>, tensor<1x?x64xf16>) outs(%[[E]] : tensor<1x?x64xf16>)
// CHECK-NOT: hip.
// CHECK: return %[[Y]]
func.func @output_gate(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
    %g: tensor<1x?x64xf16>) -> tensor<1x?x64xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %g, %c1 : tensor<1x?x64xf16>
  %es = tensor.empty(%t) : tensor<1x?x64xf16>
  %s = hip.sigmoid(%ctx) ins(%g : tensor<1x?x64xf16>) outs(%es : tensor<1x?x64xf16>) : tensor<1x?x64xf16>
  %ts = tensor.dim %s, %c1 : tensor<1x?x64xf16>
  %e = tensor.empty(%ts) : tensor<1x?x64xf16>
  %y = hip.mul(%ctx) ins(%x, %s : tensor<1x?x64xf16>, tensor<1x?x64xf16>) outs(%e : tensor<1x?x64xf16>) -> tensor<1x?x64xf16>
  return %y : tensor<1x?x64xf16>
}

// -----

// SiLU is left to the SwiGLU and gated-norm patterns.

// CHECK-LABEL: func.func @silu
// CHECK: hip.sigmoid(
// CHECK: hip.mul(
// CHECK-NOT: hip.sigmoid_mul
func.func @silu(%ctx: !hip.context, %x: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
  %es = tensor.empty() : tensor<1x1x64xf16>
  %s = hip.sigmoid(%ctx) ins(%x : tensor<1x1x64xf16>) outs(%es : tensor<1x1x64xf16>) : tensor<1x1x64xf16>
  %e = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.mul(%ctx) ins(%s, %x : tensor<1x1x64xf16>, tensor<1x1x64xf16>) outs(%e : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y : tensor<1x1x64xf16>
}

// -----

// A gate that broadcasts (a per-token sigmoid weight) is not this op.

// CHECK-LABEL: func.func @gate_broadcasts
// CHECK: hip.sigmoid(
// CHECK: hip.mul(
// CHECK-NOT: hip.sigmoid_mul
func.func @gate_broadcasts(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %g: tensor<1x1x1xf16>) -> tensor<1x1x64xf16> {
  %es = tensor.empty() : tensor<1x1x1xf16>
  %s = hip.sigmoid(%ctx) ins(%g : tensor<1x1x1xf16>) outs(%es : tensor<1x1x1xf16>) : tensor<1x1x1xf16>
  %e = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.mul(%ctx) ins(%x, %s : tensor<1x1x64xf16>, tensor<1x1x1xf16>) outs(%e : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y : tensor<1x1x64xf16>
}

// -----

// The sigmoid has a second consumer: keep both ops.

// CHECK-LABEL: func.func @sigmoid_shared
// CHECK: hip.sigmoid(
// CHECK: hip.mul(
// CHECK-NOT: hip.sigmoid_mul
func.func @sigmoid_shared(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %g: tensor<1x1x64xf16>) -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %es = tensor.empty() : tensor<1x1x64xf16>
  %s = hip.sigmoid(%ctx) ins(%g : tensor<1x1x64xf16>) outs(%es : tensor<1x1x64xf16>) : tensor<1x1x64xf16>
  %e = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.mul(%ctx) ins(%x, %s : tensor<1x1x64xf16>, tensor<1x1x64xf16>) outs(%e : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y, %s : tensor<1x1x64xf16>, tensor<1x1x64xf16>
}

// -----

// The lhs sigmoid has a second consumer, so it is the input: the rhs sigmoid
// is the gate that folds.

// CHECK-LABEL: func.func @gate_on_rhs
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[A:[^:]*]]: tensor<1x1x64xf16>, %[[B:[^:]*]]: tensor<1x1x64xf16>)
// CHECK: %[[S:.*]] = hip.sigmoid(%[[CTX]]) ins(%[[A]]
// CHECK-NOT: hip.sigmoid(
// CHECK: %[[Y:.*]] = hip.sigmoid_mul(%[[CTX]]) ins(%[[S]], %[[B]] :
// CHECK-NOT: hip.mul(
// CHECK: return %[[Y]], %[[S]]
func.func @gate_on_rhs(%ctx: !hip.context, %a: tensor<1x1x64xf16>,
    %b: tensor<1x1x64xf16>) -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %ea = tensor.empty() : tensor<1x1x64xf16>
  %sa = hip.sigmoid(%ctx) ins(%a : tensor<1x1x64xf16>) outs(%ea : tensor<1x1x64xf16>) : tensor<1x1x64xf16>
  %eb = tensor.empty() : tensor<1x1x64xf16>
  %sb = hip.sigmoid(%ctx) ins(%b : tensor<1x1x64xf16>) outs(%eb : tensor<1x1x64xf16>) : tensor<1x1x64xf16>
  %e = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.mul(%ctx) ins(%sa, %sb : tensor<1x1x64xf16>, tensor<1x1x64xf16>) outs(%e : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y, %sa : tensor<1x1x64xf16>, tensor<1x1x64xf16>
}
