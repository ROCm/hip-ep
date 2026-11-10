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
