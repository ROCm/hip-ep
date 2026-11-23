// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// The batch/sequence sizes of the init are read through tensor.dim of the
// folded copy; they are retargeted to the grouped source so the copy dies.
// CHECK-LABEL: func.func @fold_grouped_query_and_key
// CHECK-SAME:    %[[Q:[a-z0-9]+]]: tensor<?x?x1024xf32>, %[[K:[a-z0-9]+]]: tensor<?x?x1024xf32>
// CHECK-NOT:     hip.expand
// CHECK:         hip.linear_attention(%{{.*}}) ins(%[[Q]], %[[K]], %{{.*}} : tensor<?x?x1024xf32>, tensor<?x?x1024xf32>, tensor<?x?x8192xf32>)
// CHECK-SAME:    kv_num_heads = 128 : i64, q_num_heads = 8 : i64
func.func @fold_grouped_query_and_key(%ctx: !hip.context,
    %q: tensor<?x?x1024xf32>, %k: tensor<?x?x1024xf32>,
    %v: tensor<?x?x8192xf32>, %s: tensor<?x128x128x64xf32>,
    %d: tensor<?x?x128xf32>) -> (tensor<?x?x8192xf32>, tensor<?x128x128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rep = hip.constant {value = dense<[1, 1, 1, 16, 1]> : tensor<5xi64>} : tensor<5xi64>
  %b = tensor.dim %q, %c0 : tensor<?x?x1024xf32>
  %t = tensor.dim %q, %c1 : tensor<?x?x1024xf32>
  %qs = tensor.expand_shape %q [[0], [1], [2, 3, 4]] output_shape [%b, %t, 8, 1, 128] : tensor<?x?x1024xf32> into tensor<?x?x8x1x128xf32>
  %qi = tensor.empty(%b, %t) : tensor<?x?x8x16x128xf32>
  %qr = hip.expand(%ctx) ins(%qs, %rep : tensor<?x?x8x1x128xf32>, tensor<5xi64>) outs(%qi : tensor<?x?x8x16x128xf32>) : tensor<?x?x8x16x128xf32>
  %qc = tensor.collapse_shape %qr [[0], [1], [2, 3, 4]] : tensor<?x?x8x16x128xf32> into tensor<?x?x16384xf32>
  %ks = tensor.expand_shape %k [[0], [1], [2, 3, 4]] output_shape [%b, %t, 8, 1, 128] : tensor<?x?x1024xf32> into tensor<?x?x8x1x128xf32>
  %ki = tensor.empty(%b, %t) : tensor<?x?x8x16x128xf32>
  %kr = hip.expand(%ctx) ins(%ks, %rep : tensor<?x?x8x1x128xf32>, tensor<5xi64>) outs(%ki : tensor<?x?x8x16x128xf32>) : tensor<?x?x8x16x128xf32>
  %kc = tensor.collapse_shape %kr [[0], [1], [2, 3, 4]] : tensor<?x?x8x16x128xf32> into tensor<?x?x16384xf32>
  %ob = tensor.dim %qc, %c0 : tensor<?x?x16384xf32>
  %ot = tensor.dim %qc, %c1 : tensor<?x?x16384xf32>
  %oi = tensor.empty(%ob, %ot) : tensor<?x?x8192xf32>
  %si = tensor.empty(%ob) : tensor<?x128x128x64xf32>
  %y:2 = hip.linear_attention(%ctx) ins(%qc, %kc, %v : tensor<?x?x16384xf32>, tensor<?x?x16384xf32>, tensor<?x?x8192xf32>) past_state(%s : tensor<?x128x128x64xf32>) decay(%d : tensor<?x?x128xf32>) outs(%oi, %si : tensor<?x?x8192xf32>, tensor<?x128x128x64xf32>) {kv_num_heads = 128 : i64, q_num_heads = 128 : i64, scale = 1.000000e+00 : f32, update_rule = "gated"} : tensor<?x?x8192xf32>, tensor<?x128x128x64xf32>
  return %y#0, %y#1 : tensor<?x?x8192xf32>, tensor<?x128x128x64xf32>
}

// -----

// Tiling the groups ([1, G, D] -> [R, G, D]) puts group h % G on state head
// h, not h / R, so the copy stays.
// CHECK-LABEL: func.func @keep_tiled_key
// CHECK:         hip.expand
// CHECK:         hip.linear_attention
// CHECK-SAME:    q_num_heads = 128 : i64
func.func @keep_tiled_key(%ctx: !hip.context,
    %q: tensor<1x4x16384xf32>, %k: tensor<1x4x1024xf32>,
    %v: tensor<1x4x8192xf32>, %s: tensor<1x128x128x64xf32>,
    %d: tensor<1x4x128xf32>) -> (tensor<1x4x8192xf32>, tensor<1x128x128x64xf32>) {
  %rep = hip.constant {value = dense<[1, 1, 16, 1, 1]> : tensor<5xi64>} : tensor<5xi64>
  %ks = tensor.expand_shape %k [[0], [1], [2, 3, 4]] output_shape [1, 4, 1, 8, 128] : tensor<1x4x1024xf32> into tensor<1x4x1x8x128xf32>
  %ki = tensor.empty() : tensor<1x4x16x8x128xf32>
  %kr = hip.expand(%ctx) ins(%ks, %rep : tensor<1x4x1x8x128xf32>, tensor<5xi64>) outs(%ki : tensor<1x4x16x8x128xf32>) : tensor<1x4x16x8x128xf32>
  %kc = tensor.collapse_shape %kr [[0], [1], [2, 3, 4]] : tensor<1x4x16x8x128xf32> into tensor<1x4x16384xf32>
  %oi = tensor.empty() : tensor<1x4x8192xf32>
  %si = tensor.empty() : tensor<1x128x128x64xf32>
  %y:2 = hip.linear_attention(%ctx) ins(%q, %kc, %v : tensor<1x4x16384xf32>, tensor<1x4x16384xf32>, tensor<1x4x8192xf32>) past_state(%s : tensor<1x128x128x64xf32>) decay(%d : tensor<1x4x128xf32>) outs(%oi, %si : tensor<1x4x8192xf32>, tensor<1x128x128x64xf32>) {kv_num_heads = 128 : i64, q_num_heads = 128 : i64, scale = 1.000000e+00 : f32, update_rule = "gated"} : tensor<1x4x8192xf32>, tensor<1x128x128x64xf32>
  return %y#0, %y#1 : tensor<1x4x8192xf32>, tensor<1x128x128x64xf32>
}
