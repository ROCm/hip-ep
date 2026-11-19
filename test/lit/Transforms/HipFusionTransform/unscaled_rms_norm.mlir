// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// An RMS norm without a scale, exported as f32 primitives (Gemma's value
// norm): cast, x*x, last-axis mean, + eps, sqrt, reciprocal, multiply, cast
// back. hip-fusion-transform rewrites it to hip.rms_norm with a scale of
// ones.

// CHECK-LABEL: func.func @static_shape
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<1x1x2x256xf16>)
// CHECK-NOT: hip.reduce_mean
// CHECK-NOT: hip.reciprocal
// CHECK: %[[ONES:.*]] = hip.constant {value = dense<1.000000e+00> : tensor<256xf16>} : tensor<256xf16>
// CHECK: hip.rms_norm(%[[CTX]]) ins(%[[X]], %[[ONES]] : tensor<1x1x2x256xf16>, tensor<256xf16>)
// CHECK-SAME: {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
// CHECK-NOT: hip.cast
func.func @static_shape(%ctx: !hip.context, %x: tensor<1x1x2x256xf16>)
    -> tensor<1x1x2x256xf16> {
  %eps = hip.constant {value = dense<9.99999997E-7> : tensor<1xf32>} : tensor<1xf32>
  %axes = hip.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
  %e0 = tensor.empty() : tensor<1x1x2x256xf32>
  %x32 = hip.cast(%ctx) ins(%x : tensor<1x1x2x256xf16>) outs(%e0 : tensor<1x1x2x256xf32>) {to = 1 : i64} : tensor<1x1x2x256xf32>
  %e1 = tensor.empty() : tensor<1x1x2x256xf32>
  %sq = hip.mul(%ctx) ins(%x32, %x32 : tensor<1x1x2x256xf32>, tensor<1x1x2x256xf32>) outs(%e1 : tensor<1x1x2x256xf32>) -> tensor<1x1x2x256xf32>
  %e2 = tensor.empty() : tensor<1x1x2x1xf32>
  %m = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<1x1x2x256xf32>, tensor<1xi64>) outs(%e2 : tensor<1x1x2x1xf32>) {reduced_axes = array<i64: 3>} : tensor<1x1x2x1xf32>
  %e3 = tensor.empty() : tensor<1x1x2x1xf32>
  %a = hip.add(%ctx) ins(%m, %eps : tensor<1x1x2x1xf32>, tensor<1xf32>) outs(%e3 : tensor<1x1x2x1xf32>) -> tensor<1x1x2x1xf32>
  %e4 = tensor.empty() : tensor<1x1x2x1xf32>
  %s = hip.sqrt(%ctx) ins(%a : tensor<1x1x2x1xf32>) outs(%e4 : tensor<1x1x2x1xf32>) : tensor<1x1x2x1xf32>
  %e5 = tensor.empty() : tensor<1x1x2x1xf32>
  %r = hip.reciprocal(%ctx) ins(%s : tensor<1x1x2x1xf32>) outs(%e5 : tensor<1x1x2x1xf32>) : tensor<1x1x2x1xf32>
  %e6 = tensor.empty() : tensor<1x1x2x256xf32>
  %y32 = hip.mul(%ctx) ins(%x32, %r : tensor<1x1x2x256xf32>, tensor<1x1x2x1xf32>) outs(%e6 : tensor<1x1x2x256xf32>) -> tensor<1x1x2x256xf32>
  %e7 = tensor.empty() : tensor<1x1x2x256xf16>
  %y = hip.cast(%ctx) ins(%y32 : tensor<1x1x2x256xf32>) outs(%e7 : tensor<1x1x2x256xf16>) {to = 10 : i64} : tensor<1x1x2x256xf16>
  return %y : tensor<1x1x2x256xf16>
}

// -----

// The dynamic export sizes each init from the previous result. Those queries
// move to the f16 input, so nothing of the chain survives.
// CHECK-LABEL: func.func @dynamic_shape
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<?x?x2x512xf16>)
// CHECK-NOT: hip.mul
// CHECK-NOT: hip.reduce_mean
// CHECK-DAG: %[[D0:.*]] = tensor.dim %[[X]], %c0
// CHECK-DAG: %[[D1:.*]] = tensor.dim %[[X]], %c1
// CHECK: %[[E:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?x2x512xf16>
// CHECK: hip.rms_norm(%[[CTX]]) ins(%[[X]], %{{.*}} : tensor<?x?x2x512xf16>, tensor<512xf16>) outs(%[[E]] : tensor<?x?x2x512xf16>)
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x?x2x512xf16>)
    -> tensor<?x?x2x512xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %eps = hip.constant {value = dense<9.99999997E-7> : tensor<1xf32>} : tensor<1xf32>
  %axes = hip.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
  %d0 = tensor.dim %x, %c0 : tensor<?x?x2x512xf16>
  %d1 = tensor.dim %x, %c1 : tensor<?x?x2x512xf16>
  %e0 = tensor.empty(%d0, %d1) : tensor<?x?x2x512xf32>
  %x32 = hip.cast(%ctx) ins(%x : tensor<?x?x2x512xf16>) outs(%e0 : tensor<?x?x2x512xf32>) {to = 1 : i64} : tensor<?x?x2x512xf32>
  %q0 = tensor.dim %x32, %c0 : tensor<?x?x2x512xf32>
  %q1 = tensor.dim %x32, %c1 : tensor<?x?x2x512xf32>
  %e1 = tensor.empty(%q0, %q1) : tensor<?x?x2x512xf32>
  %sq = hip.mul(%ctx) ins(%x32, %x32 : tensor<?x?x2x512xf32>, tensor<?x?x2x512xf32>) outs(%e1 : tensor<?x?x2x512xf32>) -> tensor<?x?x2x512xf32>
  %w0 = tensor.dim %sq, %c0 : tensor<?x?x2x512xf32>
  %w1 = tensor.dim %sq, %c1 : tensor<?x?x2x512xf32>
  %e2 = tensor.empty(%w0, %w1) : tensor<?x?x2x1xf32>
  %m = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<?x?x2x512xf32>, tensor<1xi64>) outs(%e2 : tensor<?x?x2x1xf32>) {reduced_axes = array<i64: 3>} : tensor<?x?x2x1xf32>
  %m0 = tensor.dim %m, %c0 : tensor<?x?x2x1xf32>
  %m1 = tensor.dim %m, %c1 : tensor<?x?x2x1xf32>
  %e3 = tensor.empty(%m0, %m1) : tensor<?x?x2x1xf32>
  %a = hip.add(%ctx) ins(%m, %eps : tensor<?x?x2x1xf32>, tensor<1xf32>) outs(%e3 : tensor<?x?x2x1xf32>) -> tensor<?x?x2x1xf32>
  %a0 = tensor.dim %a, %c0 : tensor<?x?x2x1xf32>
  %a1 = tensor.dim %a, %c1 : tensor<?x?x2x1xf32>
  %e4 = tensor.empty(%a0, %a1) : tensor<?x?x2x1xf32>
  %s = hip.sqrt(%ctx) ins(%a : tensor<?x?x2x1xf32>) outs(%e4 : tensor<?x?x2x1xf32>) : tensor<?x?x2x1xf32>
  %s0 = tensor.dim %s, %c0 : tensor<?x?x2x1xf32>
  %s1 = tensor.dim %s, %c1 : tensor<?x?x2x1xf32>
  %e5 = tensor.empty(%s0, %s1) : tensor<?x?x2x1xf32>
  %r = hip.reciprocal(%ctx) ins(%s : tensor<?x?x2x1xf32>) outs(%e5 : tensor<?x?x2x1xf32>) : tensor<?x?x2x1xf32>
  %e6 = tensor.empty(%q0, %q1) : tensor<?x?x2x512xf32>
  %y32 = hip.mul(%ctx) ins(%x32, %r : tensor<?x?x2x512xf32>, tensor<?x?x2x1xf32>) outs(%e6 : tensor<?x?x2x512xf32>) -> tensor<?x?x2x512xf32>
  %y0 = tensor.dim %y32, %c0 : tensor<?x?x2x512xf32>
  %y1 = tensor.dim %y32, %c1 : tensor<?x?x2x512xf32>
  %e7 = tensor.empty(%y0, %y1) : tensor<?x?x2x512xf16>
  %y = hip.cast(%ctx) ins(%y32 : tensor<?x?x2x512xf32>) outs(%e7 : tensor<?x?x2x512xf16>) {to = 10 : i64} : tensor<?x?x2x512xf16>
  return %y : tensor<?x?x2x512xf16>
}

// -----

// The f32 copy of x is also returned, so the chain stays.
// CHECK-LABEL: func.func @cast_escapes
// CHECK: hip.reduce_mean
// CHECK-NOT: hip.rms_norm
func.func @cast_escapes(%ctx: !hip.context, %x: tensor<2x256xf16>)
    -> (tensor<2x256xf16>, tensor<2x256xf32>) {
  %eps = hip.constant {value = dense<9.99999997E-7> : tensor<1xf32>} : tensor<1xf32>
  %axes = hip.constant {value = dense<-1> : tensor<1xi64>} : tensor<1xi64>
  %e0 = tensor.empty() : tensor<2x256xf32>
  %x32 = hip.cast(%ctx) ins(%x : tensor<2x256xf16>) outs(%e0 : tensor<2x256xf32>) {to = 1 : i64} : tensor<2x256xf32>
  %e1 = tensor.empty() : tensor<2x256xf32>
  %sq = hip.mul(%ctx) ins(%x32, %x32 : tensor<2x256xf32>, tensor<2x256xf32>) outs(%e1 : tensor<2x256xf32>) -> tensor<2x256xf32>
  %e2 = tensor.empty() : tensor<2x1xf32>
  %m = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<2x256xf32>, tensor<1xi64>) outs(%e2 : tensor<2x1xf32>) {reduced_axes = array<i64: 1>} : tensor<2x1xf32>
  %e3 = tensor.empty() : tensor<2x1xf32>
  %a = hip.add(%ctx) ins(%m, %eps : tensor<2x1xf32>, tensor<1xf32>) outs(%e3 : tensor<2x1xf32>) -> tensor<2x1xf32>
  %e4 = tensor.empty() : tensor<2x1xf32>
  %s = hip.sqrt(%ctx) ins(%a : tensor<2x1xf32>) outs(%e4 : tensor<2x1xf32>) : tensor<2x1xf32>
  %e5 = tensor.empty() : tensor<2x1xf32>
  %r = hip.reciprocal(%ctx) ins(%s : tensor<2x1xf32>) outs(%e5 : tensor<2x1xf32>) : tensor<2x1xf32>
  %e6 = tensor.empty() : tensor<2x256xf32>
  %y32 = hip.mul(%ctx) ins(%x32, %r : tensor<2x256xf32>, tensor<2x1xf32>) outs(%e6 : tensor<2x256xf32>) -> tensor<2x256xf32>
  %e7 = tensor.empty() : tensor<2x256xf16>
  %y = hip.cast(%ctx) ins(%y32 : tensor<2x256xf32>) outs(%e7 : tensor<2x256xf16>) {to = 10 : i64} : tensor<2x256xf16>
  return %y, %x32 : tensor<2x256xf16>, tensor<2x256xf32>
}
