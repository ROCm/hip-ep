// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// The residual add's destination may be produced by a side-effecting op.
// The add folds into hip.skip_layer_norm, and that producer stays.
// CHECK-LABEL: func.func @residual_keeps_side_effecting_init
// CHECK: call @make_init_and_record
// CHECK-NOT: hip.add
// CHECK: hip.skip_layer_norm
func.func private @make_init_and_record() -> tensor<2x8xf32>
func.func @residual_keeps_side_effecting_init(%ctx: !hip.context,
                                              %x: tensor<2x8xf32>,
                                              %r: tensor<2x8xf32>,
                                              %g: tensor<8xf32>)
    -> tensor<2x8xf32> {
  %init = func.call @make_init_and_record() : () -> tensor<2x8xf32>
  %s = hip.add(%ctx) ins(%x, %r : tensor<2x8xf32>, tensor<2x8xf32>)
       outs(%init : tensor<2x8xf32>) -> tensor<2x8xf32>
  %e = tensor.empty() : tensor<2x8xf32>
  %y = hip.layer_norm(%ctx) ins(%s, %g : tensor<2x8xf32>, tensor<8xf32>)
       outs(%e : tensor<2x8xf32>)
       {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
       : tensor<2x8xf32>
  return %y : tensor<2x8xf32>
}

// -----

// The same for a destination in the primitive chain.
// CHECK-LABEL: func.func @chain_keeps_side_effecting_init
// CHECK: call @make_sub_init
// CHECK-NOT: hip.sub
// CHECK: hip.layer_norm
func.func private @make_sub_init() -> tensor<2x8xf32>
func.func @chain_keeps_side_effecting_init(%ctx: !hip.context,
                                           %x: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  %axes = arith.constant dense<-1> : tensor<1xi64>
  %eps = hip.constant {value = dense<9.99999974E-6> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : tensor<2x1xf32>
  %mean = hip.reduce_mean(%ctx) ins(%x, %axes : tensor<2x8xf32>, tensor<1xi64>)
          outs(%e0 : tensor<2x1xf32>) {reduced_axes = array<i64: 1>}
          : tensor<2x1xf32>
  %init = func.call @make_sub_init() : () -> tensor<2x8xf32>
  %d = hip.sub(%ctx) ins(%x, %mean : tensor<2x8xf32>, tensor<2x1xf32>)
       outs(%init : tensor<2x8xf32>) : tensor<2x8xf32>
  %e1 = tensor.empty() : tensor<2x8xf32>
  %sq = hip.mul(%ctx) ins(%d, %d : tensor<2x8xf32>, tensor<2x8xf32>)
        outs(%e1 : tensor<2x8xf32>) -> tensor<2x8xf32>
  %e2 = tensor.empty() : tensor<2x1xf32>
  %var = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<2x8xf32>, tensor<1xi64>)
         outs(%e2 : tensor<2x1xf32>) {reduced_axes = array<i64: 1>}
         : tensor<2x1xf32>
  %e3 = tensor.empty() : tensor<2x1xf32>
  %a = hip.add(%ctx) ins(%var, %eps : tensor<2x1xf32>, tensor<f32>)
       outs(%e3 : tensor<2x1xf32>) -> tensor<2x1xf32>
  %e4 = tensor.empty() : tensor<2x1xf32>
  %std = hip.sqrt(%ctx) ins(%a : tensor<2x1xf32>)
         outs(%e4 : tensor<2x1xf32>) : tensor<2x1xf32>
  %e5 = tensor.empty() : tensor<2x1xf32>
  %r = hip.reciprocal(%ctx) ins(%std : tensor<2x1xf32>)
       outs(%e5 : tensor<2x1xf32>) : tensor<2x1xf32>
  %e6 = tensor.empty() : tensor<2x8xf32>
  %n = hip.mul(%ctx) ins(%d, %r : tensor<2x8xf32>, tensor<2x1xf32>)
       outs(%e6 : tensor<2x8xf32>) -> tensor<2x8xf32>
  return %n : tensor<2x8xf32>
}

// -----

// The norm writes into the sum, which is also read later: the fused op gets
// a fresh output buffer, not its own sum result.
// CHECK-LABEL: func.func @norm_writes_into_sum
// CHECK: %[[R:.*]]:2 = hip.skip_layer_norm
// CHECK-SAME: round_sum
// CHECK: hip.mul(%{{.*}}) ins(%[[R]]#1,
// CHECK: return %[[R]]#0,
func.func @norm_writes_into_sum(%ctx: !hip.context, %x: tensor<2x8xf32>,
                                %r: tensor<2x8xf32>, %g: tensor<8xf32>)
    -> (tensor<2x8xf32>, tensor<2x8xf32>) {
  %e = tensor.empty() : tensor<2x8xf32>
  %s = hip.add(%ctx) ins(%x, %r : tensor<2x8xf32>, tensor<2x8xf32>)
       outs(%e : tensor<2x8xf32>) -> tensor<2x8xf32>
  %y = hip.layer_norm(%ctx) ins(%s, %g : tensor<2x8xf32>, tensor<8xf32>)
       outs(%s : tensor<2x8xf32>)
       {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
       : tensor<2x8xf32>
  %e2 = tensor.empty() : tensor<2x8xf32>
  %z = hip.mul(%ctx) ins(%s, %x : tensor<2x8xf32>, tensor<2x8xf32>)
       outs(%e2 : tensor<2x8xf32>) -> tensor<2x8xf32>
  return %y, %z : tensor<2x8xf32>, tensor<2x8xf32>
}
