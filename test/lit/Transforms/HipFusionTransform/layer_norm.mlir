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
  %eps = hip.constant {value = dense<1.0e-05> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : tensor<2x1xf32>
  %mean = hip.reduce_mean(%ctx) ins(%x, %axes : tensor<2x8xf32>, tensor<1xi64>)
          outs(%e0 : tensor<2x1xf32>)
          {keepdims = 1 : i64, reduced_axes = [-1]} : tensor<2x1xf32>
  %init = func.call @make_sub_init() : () -> tensor<2x8xf32>
  %d = hip.sub(%ctx) ins(%x, %mean : tensor<2x8xf32>, tensor<2x1xf32>)
       outs(%init : tensor<2x8xf32>) -> tensor<2x8xf32>
  %e1 = tensor.empty() : tensor<2x8xf32>
  %sq = hip.mul(%ctx) ins(%d, %d : tensor<2x8xf32>, tensor<2x8xf32>)
        outs(%e1 : tensor<2x8xf32>) -> tensor<2x8xf32>
  %e2 = tensor.empty() : tensor<2x1xf32>
  %var = hip.reduce_mean(%ctx) ins(%sq, %axes : tensor<2x8xf32>, tensor<1xi64>)
         outs(%e2 : tensor<2x1xf32>)
         {keepdims = 1 : i64, reduced_axes = [-1]} : tensor<2x1xf32>
  %e3 = tensor.empty() : tensor<2x1xf32>
  %a = hip.add(%ctx) ins(%var, %eps : tensor<2x1xf32>, tensor<f32>)
       outs(%e3 : tensor<2x1xf32>) -> tensor<2x1xf32>
  %e4 = tensor.empty() : tensor<2x1xf32>
  %std = hip.sqrt(%ctx) ins(%a : tensor<2x1xf32>)
         outs(%e4 : tensor<2x1xf32>) : tensor<2x1xf32>
  %e5 = tensor.empty() : tensor<2x8xf32>
  %n = hip.div(%ctx) ins(%d, %std : tensor<2x8xf32>, tensor<2x1xf32>)
       outs(%e5 : tensor<2x8xf32>) -> tensor<2x8xf32>
  return %n : tensor<2x8xf32>
}
