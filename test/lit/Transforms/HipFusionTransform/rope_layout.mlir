// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A Qwen3.5 attention layer rotates fp16 BSNH q / k as fp32 BNSH (partial
// rotary, 64 of 256 dims). hip-fusion-transform rotates the fp16 BSNH tensor
// directly against the fp32 caches.

// CHECK-LABEL: func.func @bsnh_round_trip
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<?x?x2x256xf16>, %[[POS:[^:]*]]: tensor<?x?xi64>, %[[COS:[^:]*]]: tensor<?x32xf32>, %[[SIN:[^:]*]]: tensor<?x32xf32>)
// CHECK-NOT: hip.transpose
// CHECK-NOT: hip.cast
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[X]] {{\[\[}}0], [1], [2, 3]] : tensor<?x?x2x256xf16> into tensor<?x?x512xf16>
// CHECK: %[[O:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x512xf16>
// CHECK: %[[R:.*]] = hip.rope(%[[CTX]]) ins(%[[FLAT]], %[[POS]], %[[COS]], %[[SIN]] : tensor<?x?x512xf16>, tensor<?x?xi64>, tensor<?x32xf32>, tensor<?x32xf32>) outs(%[[O]] : tensor<?x?x512xf16>)
// CHECK-SAME: {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 64 : i64}
// CHECK: %[[Y:.*]] = tensor.expand_shape %[[R]] {{\[\[}}0], [1], [2, 3]]
// CHECK-SAME: into tensor<?x?x2x256xf16>
// CHECK-NOT: hip.
// CHECK: return %[[Y]]
func.func @bsnh_round_trip(%ctx: !hip.context, %x: tensor<?x?x2x256xf16>,
    %pos: tensor<?x?xi64>, %cos: tensor<?x32xf32>, %sin: tensor<?x32xf32>)
    -> tensor<?x?x2x256xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %b = tensor.dim %x, %c0 : tensor<?x?x2x256xf16>
  %s = tensor.dim %x, %c1 : tensor<?x?x2x256xf16>
  %e0 = tensor.empty(%b, %s) : tensor<?x2x?x256xf16>
  %t = hip.transpose(%ctx) ins(%x : tensor<?x?x2x256xf16>) outs(%e0 : tensor<?x2x?x256xf16>) {perm = [0, 2, 1, 3]} : tensor<?x2x?x256xf16>
  %tb = tensor.dim %t, %c0 : tensor<?x2x?x256xf16>
  %ts = tensor.dim %t, %c2 : tensor<?x2x?x256xf16>
  %e1 = tensor.empty(%tb, %ts) : tensor<?x2x?x256xf32>
  %f = hip.cast(%ctx) ins(%t : tensor<?x2x?x256xf16>) outs(%e1 : tensor<?x2x?x256xf32>) {to = 1 : i64} : tensor<?x2x?x256xf32>
  %fb = tensor.dim %f, %c0 : tensor<?x2x?x256xf32>
  %fs = tensor.dim %f, %c2 : tensor<?x2x?x256xf32>
  %e2 = tensor.empty(%fb, %fs) : tensor<?x2x?x256xf32>
  %r = hip.rope(%ctx) ins(%f, %pos, %cos, %sin : tensor<?x2x?x256xf32>, tensor<?x?xi64>, tensor<?x32xf32>, tensor<?x32xf32>) outs(%e2 : tensor<?x2x?x256xf32>) {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 64 : i64} : tensor<?x2x?x256xf32>
  %rb = tensor.dim %r, %c0 : tensor<?x2x?x256xf32>
  %rs = tensor.dim %r, %c2 : tensor<?x2x?x256xf32>
  %e3 = tensor.empty(%rb, %rs) : tensor<?x2x?x256xf16>
  %h = hip.cast(%ctx) ins(%r : tensor<?x2x?x256xf32>) outs(%e3 : tensor<?x2x?x256xf16>) {to = 10 : i64} : tensor<?x2x?x256xf16>
  %hb = tensor.dim %h, %c0 : tensor<?x2x?x256xf16>
  %hs = tensor.dim %h, %c2 : tensor<?x2x?x256xf16>
  %e4 = tensor.empty(%hb, %hs) : tensor<?x?x2x256xf16>
  %y = hip.transpose(%ctx) ins(%h : tensor<?x2x?x256xf16>) outs(%e4 : tensor<?x?x2x256xf16>) {perm = [0, 2, 1, 3]} : tensor<?x?x2x256xf16>
  return %y : tensor<?x?x2x256xf16>
}

// -----

// The BNSH fp16 result has a second consumer: keep the chain.

// CHECK-LABEL: func.func @bnsh_result_used
// CHECK: hip.transpose
// CHECK: hip.cast
// CHECK: hip.rope
// CHECK-SAME: tensor<1x2x1x256xf32>
// CHECK: hip.cast
// CHECK: hip.transpose
func.func @bnsh_result_used(%ctx: !hip.context, %x: tensor<1x1x2x256xf16>,
    %pos: tensor<1x1xi64>, %cos: tensor<8x32xf32>, %sin: tensor<8x32xf32>)
    -> (tensor<1x1x2x256xf16>, tensor<1x2x1x256xf16>) {
  %e0 = tensor.empty() : tensor<1x2x1x256xf16>
  %t = hip.transpose(%ctx) ins(%x : tensor<1x1x2x256xf16>) outs(%e0 : tensor<1x2x1x256xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x1x256xf16>
  %e1 = tensor.empty() : tensor<1x2x1x256xf32>
  %f = hip.cast(%ctx) ins(%t : tensor<1x2x1x256xf16>) outs(%e1 : tensor<1x2x1x256xf32>) {to = 1 : i64} : tensor<1x2x1x256xf32>
  %e2 = tensor.empty() : tensor<1x2x1x256xf32>
  %r = hip.rope(%ctx) ins(%f, %pos, %cos, %sin : tensor<1x2x1x256xf32>, tensor<1x1xi64>, tensor<8x32xf32>, tensor<8x32xf32>) outs(%e2 : tensor<1x2x1x256xf32>) {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 64 : i64} : tensor<1x2x1x256xf32>
  %e3 = tensor.empty() : tensor<1x2x1x256xf16>
  %h = hip.cast(%ctx) ins(%r : tensor<1x2x1x256xf32>) outs(%e3 : tensor<1x2x1x256xf16>) {to = 10 : i64} : tensor<1x2x1x256xf16>
  %e4 = tensor.empty() : tensor<1x1x2x256xf16>
  %y = hip.transpose(%ctx) ins(%h : tensor<1x2x1x256xf16>) outs(%e4 : tensor<1x1x2x256xf16>) {perm = [0, 2, 1, 3]} : tensor<1x1x2x256xf16>
  return %y, %h : tensor<1x1x2x256xf16>, tensor<1x2x1x256xf16>
}

// -----

// fp16 caches would rotate in fp16, not fp32: keep the chain.

// CHECK-LABEL: func.func @f16_caches
// CHECK: hip.transpose
// CHECK: hip.cast
// CHECK: hip.rope
// CHECK: hip.cast
// CHECK: hip.transpose
func.func @f16_caches(%ctx: !hip.context, %x: tensor<1x1x2x256xf16>,
    %pos: tensor<1x1xi64>, %cos: tensor<8x32xf16>, %sin: tensor<8x32xf16>)
    -> tensor<1x1x2x256xf16> {
  %e0 = tensor.empty() : tensor<1x2x1x256xf16>
  %t = hip.transpose(%ctx) ins(%x : tensor<1x1x2x256xf16>) outs(%e0 : tensor<1x2x1x256xf16>) {perm = [0, 2, 1, 3]} : tensor<1x2x1x256xf16>
  %e1 = tensor.empty() : tensor<1x2x1x256xf32>
  %f = hip.cast(%ctx) ins(%t : tensor<1x2x1x256xf16>) outs(%e1 : tensor<1x2x1x256xf32>) {to = 1 : i64} : tensor<1x2x1x256xf32>
  %e2 = tensor.empty() : tensor<1x2x1x256xf32>
  %r = hip.rope(%ctx) ins(%f, %pos, %cos, %sin : tensor<1x2x1x256xf32>, tensor<1x1xi64>, tensor<8x32xf16>, tensor<8x32xf16>) outs(%e2 : tensor<1x2x1x256xf32>) {interleaved = 0 : i64, num_heads = 2 : i64, rotary_embedding_dim = 64 : i64} : tensor<1x2x1x256xf32>
  %e3 = tensor.empty() : tensor<1x2x1x256xf16>
  %h = hip.cast(%ctx) ins(%r : tensor<1x2x1x256xf32>) outs(%e3 : tensor<1x2x1x256xf16>) {to = 10 : i64} : tensor<1x2x1x256xf16>
  %e4 = tensor.empty() : tensor<1x1x2x256xf16>
  %y = hip.transpose(%ctx) ins(%h : tensor<1x2x1x256xf16>) outs(%e4 : tensor<1x1x2x256xf16>) {perm = [0, 2, 1, 3]} : tensor<1x1x2x256xf16>
  return %y : tensor<1x1x2x256xf16>
}
