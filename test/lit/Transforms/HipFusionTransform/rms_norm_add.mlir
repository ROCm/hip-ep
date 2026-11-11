// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A post-norm residual connection lowers to two HIP ops:
//
//   n = hip.rms_norm(x, scale); y = hip.add(residual, n)
//
// hip-fusion-transform collapses the pair into one hip.rms_norm_add. The add
// is commutative. The norm result must be single-use, and every tensor must
// have one runtime shape, since hip.add broadcasts and the fused op does not.

// CHECK-LABEL: func.func @residual
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<1x4x64xf16>, %[[G:[^:]*]]: tensor<64xf16>, %[[R:[^:]*]]: tensor<1x4x64xf16>)
// CHECK-NOT: hip.rms_norm(
// CHECK-NOT: {{hip\.add\(}}
// CHECK: hip.rms_norm_add(%[[CTX]]) ins(%[[X]], %[[G]], %[[R]] : tensor<1x4x64xf16>, tensor<64xf16>, tensor<1x4x64xf16>)
// CHECK-SAME: {axis = -1 : i64, epsilon = 9.99999997E-7 : f32}
func.func @residual(%ctx: !hip.context, %x: tensor<1x4x64xf16>,
                    %g: tensor<64xf16>, %r: tensor<1x4x64xf16>)
    -> tensor<1x4x64xf16> {
  %e0 = tensor.empty() : tensor<1x4x64xf16>
  %n = hip.rms_norm(%ctx) ins(%x, %g : tensor<1x4x64xf16>, tensor<64xf16>)
       outs(%e0 : tensor<1x4x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<1x4x64xf16>
  %e1 = tensor.empty() : tensor<1x4x64xf16>
  %y = hip.add(%ctx) ins(%r, %n : tensor<1x4x64xf16>, tensor<1x4x64xf16>)
       outs(%e1 : tensor<1x4x64xf16>) -> tensor<1x4x64xf16>
  return %y : tensor<1x4x64xf16>
}

// -----

// The norm on the left of the add.
// CHECK-LABEL: func.func @residual_commuted
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<2x64xf16>, %[[G:[^:]*]]: tensor<64xf16>, %[[R:[^:]*]]: tensor<2x64xf16>)
// CHECK: hip.rms_norm_add(%[[CTX]]) ins(%[[X]], %[[G]], %[[R]] :
func.func @residual_commuted(%ctx: !hip.context, %x: tensor<2x64xf16>,
                             %g: tensor<64xf16>, %r: tensor<2x64xf16>)
    -> tensor<2x64xf16> {
  %e0 = tensor.empty() : tensor<2x64xf16>
  %n = hip.rms_norm(%ctx) ins(%x, %g : tensor<2x64xf16>, tensor<64xf16>)
       outs(%e0 : tensor<2x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<2x64xf16>
  %e1 = tensor.empty() : tensor<2x64xf16>
  %y = hip.add(%ctx) ins(%n, %r : tensor<2x64xf16>, tensor<2x64xf16>)
       outs(%e1 : tensor<2x64xf16>) -> tensor<2x64xf16>
  return %y : tensor<2x64xf16>
}

// -----

// The norm input comes from an elementwise op that sized its destination for
// a possible broadcast, select(lhs == 1, rhs, lhs) per axis. Both arms trace
// to the residual's extents, so the select is that extent.
// CHECK-LABEL: func.func @residual_broadcast_sized_input
// CHECK-NOT: hip.rms_norm(
// CHECK-NOT: {{hip\.add\(}}
// CHECK: hip.rms_norm_add
func.func @residual_broadcast_sized_input(%ctx: !hip.context,
                                          %r: tensor<?x?x64xf16>,
                                          %g: tensor<64xf16>)
    -> tensor<?x?x64xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?x64xf16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?x64xf16>
  %ea = tensor.empty(%d0, %d1) : tensor<?x?x64xf16>
  %a = hip.gelu(%ctx) ins(%r : tensor<?x?x64xf16>)
       outs(%ea : tensor<?x?x64xf16>) {approximate = "tanh"} : tensor<?x?x64xf16>
  %a0 = tensor.dim %a, %c0 : tensor<?x?x64xf16>
  %a1 = tensor.dim %a, %c1 : tensor<?x?x64xf16>
  %b0 = tensor.dim %r, %c0 : tensor<?x?x64xf16>
  %b1 = tensor.dim %r, %c1 : tensor<?x?x64xf16>
  %p0 = arith.cmpi eq, %a0, %c1 : index
  %s0 = arith.select %p0, %b0, %a0 : index
  %p1 = arith.cmpi eq, %a1, %c1 : index
  %s1 = arith.select %p1, %b1, %a1 : index
  %em = tensor.empty(%s0, %s1) : tensor<?x?x64xf16>
  %m = hip.mul(%ctx) ins(%a, %r : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
       outs(%em : tensor<?x?x64xf16>) -> tensor<?x?x64xf16>
  %m0 = tensor.dim %m, %c0 : tensor<?x?x64xf16>
  %m1 = tensor.dim %m, %c1 : tensor<?x?x64xf16>
  %en = tensor.empty(%m0, %m1) : tensor<?x?x64xf16>
  %n = hip.rms_norm(%ctx) ins(%m, %g : tensor<?x?x64xf16>, tensor<64xf16>)
       outs(%en : tensor<?x?x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<?x?x64xf16>
  %ey = tensor.empty(%d0, %d1) : tensor<?x?x64xf16>
  %y = hip.add(%ctx) ins(%r, %n : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
       outs(%ey : tensor<?x?x64xf16>) -> tensor<?x?x64xf16>
  return %y : tensor<?x?x64xf16>
}

// -----

// A select whose arms come from independent arguments proves nothing.
// CHECK-LABEL: func.func @blocked_select_of_unrelated_extents
// CHECK-NOT: hip.rms_norm_add
// CHECK: hip.rms_norm(
// CHECK: hip.add(
func.func @blocked_select_of_unrelated_extents(%ctx: !hip.context,
                                               %r: tensor<?x64xf16>,
                                               %q: tensor<?x64xf16>,
                                               %g: tensor<64xf16>)
    -> tensor<?x64xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r0 = tensor.dim %r, %c0 : tensor<?x64xf16>
  %q0 = tensor.dim %q, %c0 : tensor<?x64xf16>
  %p = arith.cmpi eq, %q0, %c1 : index
  %s = arith.select %p, %r0, %q0 : index
  %em = tensor.empty(%s) : tensor<?x64xf16>
  %m = hip.mul(%ctx) ins(%q, %r : tensor<?x64xf16>, tensor<?x64xf16>)
       outs(%em : tensor<?x64xf16>) -> tensor<?x64xf16>
  %en = tensor.empty(%s) : tensor<?x64xf16>
  %n = hip.rms_norm(%ctx) ins(%m, %g : tensor<?x64xf16>, tensor<64xf16>)
       outs(%en : tensor<?x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<?x64xf16>
  %ey = tensor.empty(%r0) : tensor<?x64xf16>
  %y = hip.add(%ctx) ins(%r, %n : tensor<?x64xf16>, tensor<?x64xf16>)
       outs(%ey : tensor<?x64xf16>) -> tensor<?x64xf16>
  return %y : tensor<?x64xf16>
}

// -----

// Gemma's per-layer input: the norm input is gelu(h) times a second argument,
// sized for a possible broadcast. The arguments name their axes alike in
// onnx.dim_params, so the select arms share a root and the norm input has
// the residual's shape.
// CHECK-LABEL: func.func @residual_named_dims_across_arguments
// CHECK-NOT: {{hip\.add\(}}
// CHECK: hip.geglu
// CHECK: hip.rms_norm_add
func.func @residual_named_dims_across_arguments(
    %ctx: !hip.context,
    %h: tensor<?x?x64xf16> {onnx.dim_params = ["batch", "seq", ""]},
    %p: tensor<?x?x64xf16> {onnx.dim_params = ["batch", "seq", ""]},
    %g: tensor<64xf16>) -> tensor<?x?x64xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %h0 = tensor.dim %h, %c0 : tensor<?x?x64xf16>
  %h1 = tensor.dim %h, %c1 : tensor<?x?x64xf16>
  %ea = tensor.empty(%h0, %h1) : tensor<?x?x64xf16>
  %a = hip.gelu(%ctx) ins(%h : tensor<?x?x64xf16>)
       outs(%ea : tensor<?x?x64xf16>) {approximate = "tanh"} : tensor<?x?x64xf16>
  %a0 = tensor.dim %a, %c0 : tensor<?x?x64xf16>
  %a1 = tensor.dim %a, %c1 : tensor<?x?x64xf16>
  %p0 = tensor.dim %p, %c0 : tensor<?x?x64xf16>
  %p1 = tensor.dim %p, %c1 : tensor<?x?x64xf16>
  %q0 = arith.cmpi eq, %a0, %c1 : index
  %s0 = arith.select %q0, %p0, %a0 : index
  %q1 = arith.cmpi eq, %a1, %c1 : index
  %s1 = arith.select %q1, %p1, %a1 : index
  %em = tensor.empty(%s0, %s1) : tensor<?x?x64xf16>
  %m = hip.mul(%ctx) ins(%a, %p : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
       outs(%em : tensor<?x?x64xf16>) -> tensor<?x?x64xf16>
  %m0 = tensor.dim %m, %c0 : tensor<?x?x64xf16>
  %m1 = tensor.dim %m, %c1 : tensor<?x?x64xf16>
  %en = tensor.empty(%m0, %m1) : tensor<?x?x64xf16>
  %n = hip.rms_norm(%ctx) ins(%m, %g : tensor<?x?x64xf16>, tensor<64xf16>)
       outs(%en : tensor<?x?x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<?x?x64xf16>
  %ey = tensor.empty(%h0, %h1) : tensor<?x?x64xf16>
  %y = hip.add(%ctx) ins(%h, %n : tensor<?x?x64xf16>, tensor<?x?x64xf16>)
       outs(%ey : tensor<?x?x64xf16>) -> tensor<?x?x64xf16>
  return %y : tensor<?x?x64xf16>
}

// -----

// Six multiplies each size their destination for a broadcast against the
// residual, so every select arm leads back through the select before it.
// The roots are followed however deep the chain is.
// CHECK-LABEL: func.func @residual_deep_broadcast_sized_chain
// CHECK-NOT: {{hip\.add\(}}
// CHECK: hip.rms_norm_add
func.func @residual_deep_broadcast_sized_chain(%ctx: !hip.context,
                                               %r: tensor<?x64xf16>,
                                               %g: tensor<64xf16>)
    -> tensor<?x64xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r0 = tensor.dim %r, %c0 : tensor<?x64xf16>
  %d1 = tensor.dim %r, %c0 : tensor<?x64xf16>
  %p1 = arith.cmpi eq, %d1, %c1 : index
  %s1 = arith.select %p1, %r0, %d1 : index
  %e1 = tensor.empty(%s1) : tensor<?x64xf16>
  %x1 = hip.mul(%ctx) ins(%r, %r : tensor<?x64xf16>, tensor<?x64xf16>)
        outs(%e1 : tensor<?x64xf16>) -> tensor<?x64xf16>
  %d2 = tensor.dim %x1, %c0 : tensor<?x64xf16>
  %p2 = arith.cmpi eq, %d2, %c1 : index
  %s2 = arith.select %p2, %r0, %d2 : index
  %e2 = tensor.empty(%s2) : tensor<?x64xf16>
  %x2 = hip.mul(%ctx) ins(%x1, %r : tensor<?x64xf16>, tensor<?x64xf16>)
        outs(%e2 : tensor<?x64xf16>) -> tensor<?x64xf16>
  %d3 = tensor.dim %x2, %c0 : tensor<?x64xf16>
  %p3 = arith.cmpi eq, %d3, %c1 : index
  %s3 = arith.select %p3, %r0, %d3 : index
  %e3 = tensor.empty(%s3) : tensor<?x64xf16>
  %x3 = hip.mul(%ctx) ins(%x2, %r : tensor<?x64xf16>, tensor<?x64xf16>)
        outs(%e3 : tensor<?x64xf16>) -> tensor<?x64xf16>
  %d4 = tensor.dim %x3, %c0 : tensor<?x64xf16>
  %p4 = arith.cmpi eq, %d4, %c1 : index
  %s4 = arith.select %p4, %r0, %d4 : index
  %e4 = tensor.empty(%s4) : tensor<?x64xf16>
  %x4 = hip.mul(%ctx) ins(%x3, %r : tensor<?x64xf16>, tensor<?x64xf16>)
        outs(%e4 : tensor<?x64xf16>) -> tensor<?x64xf16>
  %d5 = tensor.dim %x4, %c0 : tensor<?x64xf16>
  %p5 = arith.cmpi eq, %d5, %c1 : index
  %s5 = arith.select %p5, %r0, %d5 : index
  %e5 = tensor.empty(%s5) : tensor<?x64xf16>
  %x5 = hip.mul(%ctx) ins(%x4, %r : tensor<?x64xf16>, tensor<?x64xf16>)
        outs(%e5 : tensor<?x64xf16>) -> tensor<?x64xf16>
  %d6 = tensor.dim %x5, %c0 : tensor<?x64xf16>
  %p6 = arith.cmpi eq, %d6, %c1 : index
  %s6 = arith.select %p6, %r0, %d6 : index
  %e6 = tensor.empty(%s6) : tensor<?x64xf16>
  %x6 = hip.mul(%ctx) ins(%x5, %r : tensor<?x64xf16>, tensor<?x64xf16>)
        outs(%e6 : tensor<?x64xf16>) -> tensor<?x64xf16>
  %en = tensor.empty(%s6) : tensor<?x64xf16>
  %n = hip.rms_norm(%ctx) ins(%x6, %g : tensor<?x64xf16>, tensor<64xf16>)
       outs(%en : tensor<?x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<?x64xf16>
  %ey = tensor.empty(%r0) : tensor<?x64xf16>
  %y = hip.add(%ctx) ins(%r, %n : tensor<?x64xf16>, tensor<?x64xf16>)
       outs(%ey : tensor<?x64xf16>) -> tensor<?x64xf16>
  return %y : tensor<?x64xf16>
}

// -----

// The norm result is also returned, so it must stay materialized.
// CHECK-LABEL: func.func @blocked_norm_multi_use
// CHECK-NOT: hip.rms_norm_add
// CHECK: hip.rms_norm(
// CHECK: hip.add(
func.func @blocked_norm_multi_use(%ctx: !hip.context, %x: tensor<2x64xf16>,
                                  %g: tensor<64xf16>, %r: tensor<2x64xf16>)
    -> (tensor<2x64xf16>, tensor<2x64xf16>) {
  %e0 = tensor.empty() : tensor<2x64xf16>
  %n = hip.rms_norm(%ctx) ins(%x, %g : tensor<2x64xf16>, tensor<64xf16>)
       outs(%e0 : tensor<2x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<2x64xf16>
  %e1 = tensor.empty() : tensor<2x64xf16>
  %y = hip.add(%ctx) ins(%r, %n : tensor<2x64xf16>, tensor<2x64xf16>)
       outs(%e1 : tensor<2x64xf16>) -> tensor<2x64xf16>
  return %y, %n : tensor<2x64xf16>, tensor<2x64xf16>
}

// -----

// A broadcasting residual has no fused form.
// CHECK-LABEL: func.func @blocked_broadcast_residual
// CHECK-NOT: hip.rms_norm_add
// CHECK: hip.rms_norm(
// CHECK: hip.add(
func.func @blocked_broadcast_residual(%ctx: !hip.context, %x: tensor<2x64xf16>,
                                      %g: tensor<64xf16>, %r: tensor<1x64xf16>)
    -> tensor<2x64xf16> {
  %e0 = tensor.empty() : tensor<2x64xf16>
  %n = hip.rms_norm(%ctx) ins(%x, %g : tensor<2x64xf16>, tensor<64xf16>)
       outs(%e0 : tensor<2x64xf16>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<2x64xf16>
  %e1 = tensor.empty() : tensor<2x64xf16>
  %y = hip.add(%ctx) ins(%r, %n : tensor<1x64xf16>, tensor<2x64xf16>)
       outs(%e1 : tensor<2x64xf16>) -> tensor<2x64xf16>
  return %y : tensor<2x64xf16>
}

// -----

// The fused kernel's residual add is f16 only.
// CHECK-LABEL: func.func @blocked_f32
// CHECK-NOT: hip.rms_norm_add
// CHECK: hip.rms_norm(
// CHECK: hip.add(
func.func @blocked_f32(%ctx: !hip.context, %x: tensor<2x64xf32>,
                       %g: tensor<64xf32>, %r: tensor<2x64xf32>)
    -> tensor<2x64xf32> {
  %e0 = tensor.empty() : tensor<2x64xf32>
  %n = hip.rms_norm(%ctx) ins(%x, %g : tensor<2x64xf32>, tensor<64xf32>)
       outs(%e0 : tensor<2x64xf32>)
       {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
       : tensor<2x64xf32>
  %e1 = tensor.empty() : tensor<2x64xf32>
  %y = hip.add(%ctx) ins(%r, %n : tensor<2x64xf32>, tensor<2x64xf32>)
       outs(%e1 : tensor<2x64xf32>) -> tensor<2x64xf32>
  return %y : tensor<2x64xf32>
}
