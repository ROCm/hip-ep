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

// -----

// The scale is viewed as [2, 1], so it scales rows of the [2, 2] norm, not
// channels: the norm folds and the multiply stays after it.
// CHECK-LABEL: func.func @scale_view_on_rows
// CHECK: hip.layer_norm
// CHECK: hip.mul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<2x2xf32>, tensor<2x1xf32>)
func.func @scale_view_on_rows(%ctx: !hip.context, %x: tensor<2x2xf32>,
                              %g: tensor<2xf32>) -> tensor<2x2xf32> {
  %axes = arith.constant dense<-1> : tensor<1xi64>
  %eps = hip.constant {value = dense<9.99999974E-6> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : tensor<2x1xf32>
  %mean = hip.reduce_mean(%ctx) ins(%x, %axes : tensor<2x2xf32>, tensor<1xi64>)
          outs(%e0 : tensor<2x1xf32>) {reduced_axes = array<i64: 1>}
          : tensor<2x1xf32>
  %e1 = tensor.empty() : tensor<2x2xf32>
  %d = hip.sub(%ctx) ins(%x, %mean : tensor<2x2xf32>, tensor<2x1xf32>)
       outs(%e1 : tensor<2x2xf32>) : tensor<2x2xf32>
  %e2 = tensor.empty() : tensor<2x2xf32>
  %sq = hip.mul(%ctx) ins(%d, %d : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%e2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %e3 = tensor.empty() : tensor<2x1xf32>
  %var = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<2x2xf32>, tensor<1xi64>)
         outs(%e3 : tensor<2x1xf32>) {reduced_axes = array<i64: 1>}
         : tensor<2x1xf32>
  %e4 = tensor.empty() : tensor<2x1xf32>
  %a = hip.add(%ctx) ins(%var, %eps : tensor<2x1xf32>, tensor<f32>)
       outs(%e4 : tensor<2x1xf32>) -> tensor<2x1xf32>
  %e5 = tensor.empty() : tensor<2x1xf32>
  %std = hip.sqrt(%ctx) ins(%a : tensor<2x1xf32>)
         outs(%e5 : tensor<2x1xf32>) : tensor<2x1xf32>
  %e6 = tensor.empty() : tensor<2x2xf32>
  %n = hip.div(%ctx) ins(%d, %std : tensor<2x2xf32>, tensor<2x1xf32>)
       outs(%e6 : tensor<2x2xf32>) : tensor<2x2xf32>
  %gv = tensor.expand_shape %g [[0, 1]] output_shape [2, 1]
        : tensor<2xf32> into tensor<2x1xf32>
  %e7 = tensor.empty() : tensor<2x2xf32>
  %y = hip.mul(%ctx) ins(%n, %gv : tensor<2x2xf32>, tensor<2x1xf32>)
       outs(%e7 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// -----

// x [1, 2, 2] is packed to [2, 2] but its [1, 2, 1] mean to [1, 2], so the
// packed Sub subtracts the row means by column. Not a LayerNorm.
// CHECK-LABEL: func.func @pack_groups_disagree
// CHECK-NOT: hip.layer_norm
// CHECK: return
func.func @pack_groups_disagree(%ctx: !hip.context, %x: tensor<1x2x2xf32>)
    -> tensor<1x2x2xf32> {
  %axes = arith.constant dense<-1> : tensor<1xi64>
  %eps = hip.constant {value = dense<9.99999974E-6> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : tensor<1x2x1xf32>
  %mean = hip.reduce_mean(%ctx) ins(%x, %axes : tensor<1x2x2xf32>, tensor<1xi64>)
          outs(%e0 : tensor<1x2x1xf32>) {reduced_axes = array<i64: 2>}
          : tensor<1x2x1xf32>
  %xp = tensor.collapse_shape %x [[0, 1], [2]]
        : tensor<1x2x2xf32> into tensor<2x2xf32>
  %mp = tensor.collapse_shape %mean [[0], [1, 2]]
        : tensor<1x2x1xf32> into tensor<1x2xf32>
  %e1 = tensor.empty() : tensor<2x2xf32>
  %dp = hip.sub(%ctx) ins(%xp, %mp : tensor<2x2xf32>, tensor<1x2xf32>)
        outs(%e1 : tensor<2x2xf32>) : tensor<2x2xf32>
  %d = tensor.expand_shape %dp [[0, 1], [2]] output_shape [1, 2, 2]
       : tensor<2x2xf32> into tensor<1x2x2xf32>
  %e2 = tensor.empty() : tensor<1x2x2xf32>
  %sq = hip.mul(%ctx) ins(%d, %d : tensor<1x2x2xf32>, tensor<1x2x2xf32>)
        outs(%e2 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  %e3 = tensor.empty() : tensor<1x2x1xf32>
  %var = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<1x2x2xf32>, tensor<1xi64>)
         outs(%e3 : tensor<1x2x1xf32>) {reduced_axes = array<i64: 2>}
         : tensor<1x2x1xf32>
  %e4 = tensor.empty() : tensor<1x2x1xf32>
  %a = hip.add(%ctx) ins(%var, %eps : tensor<1x2x1xf32>, tensor<f32>)
       outs(%e4 : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  %e5 = tensor.empty() : tensor<1x2x1xf32>
  %std = hip.sqrt(%ctx) ins(%a : tensor<1x2x1xf32>)
         outs(%e5 : tensor<1x2x1xf32>) : tensor<1x2x1xf32>
  %e6 = tensor.empty() : tensor<1x2x2xf32>
  %n = hip.div(%ctx) ins(%d, %std : tensor<1x2x2xf32>, tensor<1x2x1xf32>)
       outs(%e6 : tensor<1x2x2xf32>) : tensor<1x2x2xf32>
  return %n : tensor<1x2x2xf32>
}

// -----

// The pack restores [B, S, 4, 5, 8] as [S, B, 4, 5, 8]. Both are
// tensor<?x?x4x5x8xf32>, but a norm sized from x would change the result
// shape when B and S differ.
// CHECK-LABEL: func.func @pack_restore_swaps_dynamic_extents
// CHECK-NOT: hip.layer_norm
// CHECK: return
func.func @pack_restore_swaps_dynamic_extents(%ctx: !hip.context,
                                              %x: tensor<?x?x4x5x8xf32>)
    -> tensor<?x?x4x5x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b = tensor.dim %x, %c0 : tensor<?x?x4x5x8xf32>
  %s = tensor.dim %x, %c1 : tensor<?x?x4x5x8xf32>
  %axes = arith.constant dense<-1> : tensor<1xi64>
  %eps = hip.constant {value = dense<9.99999974E-6> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty(%b, %s) : tensor<?x?x4x5x1xf32>
  %mean = hip.reduce_mean(%ctx) ins(%x, %axes : tensor<?x?x4x5x8xf32>, tensor<1xi64>)
          outs(%e0 : tensor<?x?x4x5x1xf32>) {reduced_axes = array<i64: 4>}
          : tensor<?x?x4x5x1xf32>
  %e1 = tensor.empty(%b, %s) : tensor<?x?x4x5x8xf32>
  %d = hip.sub(%ctx) ins(%x, %mean : tensor<?x?x4x5x8xf32>, tensor<?x?x4x5x1xf32>)
       outs(%e1 : tensor<?x?x4x5x8xf32>) : tensor<?x?x4x5x8xf32>
  %e2 = tensor.empty(%b, %s) : tensor<?x?x4x5x8xf32>
  %sq = hip.mul(%ctx) ins(%d, %d : tensor<?x?x4x5x8xf32>, tensor<?x?x4x5x8xf32>)
        outs(%e2 : tensor<?x?x4x5x8xf32>) -> tensor<?x?x4x5x8xf32>
  %e3 = tensor.empty(%b, %s) : tensor<?x?x4x5x1xf32>
  %var = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<?x?x4x5x8xf32>, tensor<1xi64>)
         outs(%e3 : tensor<?x?x4x5x1xf32>) {reduced_axes = array<i64: 4>}
         : tensor<?x?x4x5x1xf32>
  %e4 = tensor.empty(%b, %s) : tensor<?x?x4x5x1xf32>
  %a = hip.add(%ctx) ins(%var, %eps : tensor<?x?x4x5x1xf32>, tensor<f32>)
       outs(%e4 : tensor<?x?x4x5x1xf32>) -> tensor<?x?x4x5x1xf32>
  %e5 = tensor.empty(%b, %s) : tensor<?x?x4x5x1xf32>
  %std = hip.sqrt(%ctx) ins(%a : tensor<?x?x4x5x1xf32>)
         outs(%e5 : tensor<?x?x4x5x1xf32>) : tensor<?x?x4x5x1xf32>
  %dp = tensor.collapse_shape %d [[0, 1], [2], [3], [4]]
        : tensor<?x?x4x5x8xf32> into tensor<?x4x5x8xf32>
  %sp = tensor.collapse_shape %std [[0, 1], [2], [3], [4]]
        : tensor<?x?x4x5x1xf32> into tensor<?x4x5x1xf32>
  %bs = arith.muli %b, %s : index
  %e6 = tensor.empty(%bs) : tensor<?x4x5x8xf32>
  %np = hip.div(%ctx) ins(%dp, %sp : tensor<?x4x5x8xf32>, tensor<?x4x5x1xf32>)
        outs(%e6 : tensor<?x4x5x8xf32>) : tensor<?x4x5x8xf32>
  %n = tensor.expand_shape %np [[0, 1], [2], [3], [4]] output_shape [%s, %b, 4, 5, 8]
       : tensor<?x4x5x8xf32> into tensor<?x?x4x5x8xf32>
  return %n : tensor<?x?x4x5x8xf32>
}
