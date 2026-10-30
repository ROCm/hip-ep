// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --canonicalize --verify-each -o %t
// RUN: FileCheck %s < %t
// RUN: hip-mlir-opt %t --canonicalize --verify-each -o %t.again
// RUN: diff %t %t.again

// The cast pair around a nearest resize folds away, and the resize runs on f16.
// CHECK-LABEL: func.func @nearest_resize
// CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<1x2x4x4xf16>)
// CHECK-NOT: hip.cast
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x2x8x8xf16>
// CHECK: %[[R:.*]] = hip.resize(%{{.*}}) ins(%[[X]] : tensor<1x2x4x4xf16>) outs(%[[E]] : tensor<1x2x8x8xf16>) {{.*}}mode = 0
// CHECK-NOT: hip.cast
// CHECK: return %[[R]] : tensor<1x2x8x8xf16>
func.func @nearest_resize(%ctx: !hip.context, %x: tensor<1x2x4x4xf16>)
    -> tensor<1x2x8x8xf16> {
  %we = tensor.empty() : tensor<1x2x4x4xf32>
  %w = hip.cast(%ctx) ins(%x : tensor<1x2x4x4xf16>)
      outs(%we : tensor<1x2x4x4xf32>) {to = 1 : i64} : tensor<1x2x4x4xf32>
  %re = tensor.empty() : tensor<1x2x8x8xf32>
  %r = hip.resize(%ctx) ins(%w : tensor<1x2x4x4xf32>)
                        outs(%re : tensor<1x2x8x8xf32>)
       {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 2 : i64}
       : tensor<1x2x8x8xf32>
  %ye = tensor.empty() : tensor<1x2x8x8xf16>
  %y = hip.cast(%ctx) ins(%r : tensor<1x2x8x8xf32>)
      outs(%ye : tensor<1x2x8x8xf16>) {to = 10 : i64} : tensor<1x2x8x8xf16>
  return %y : tensor<1x2x8x8xf16>
}

// A chain of element moves folds too: a bf16 to f64 widening, then a
// transpose and an expand_shape.
// CHECK-LABEL: func.func @transpose_expand
// CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<1x4x4x8xbf16>)
// CHECK-NOT: hip.cast
// CHECK: %[[T:.*]] = hip.transpose(%{{.*}}) ins(%[[X]] : tensor<1x4x4x8xbf16>) outs(%{{.*}} : tensor<1x8x4x4xbf16>)
// CHECK: %[[S:.*]] = tensor.expand_shape %[[T]] {{.*}} : tensor<1x8x4x4xbf16> into tensor<1x2x4x4x4xbf16>
// CHECK-NOT: hip.cast
// CHECK: return %[[S]] : tensor<1x2x4x4x4xbf16>
func.func @transpose_expand(%ctx: !hip.context, %x: tensor<1x4x4x8xbf16>)
    -> tensor<1x2x4x4x4xbf16> {
  %we = tensor.empty() : tensor<1x4x4x8xf64>
  %w = hip.cast(%ctx) ins(%x : tensor<1x4x4x8xbf16>)
      outs(%we : tensor<1x4x4x8xf64>) {to = 11 : i64} : tensor<1x4x4x8xf64>
  %te = tensor.empty() : tensor<1x8x4x4xf64>
  %t = hip.transpose(%ctx) ins(%w : tensor<1x4x4x8xf64>)
      outs(%te : tensor<1x8x4x4xf64>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf64>
  %s = tensor.expand_shape %t [[0], [1, 2], [3], [4]]
      output_shape [1, 2, 4, 4, 4] : tensor<1x8x4x4xf64> into tensor<1x2x4x4x4xf64>
  %ye = tensor.empty() : tensor<1x2x4x4x4xbf16>
  %y = hip.cast(%ctx) ins(%s : tensor<1x2x4x4x4xf64>)
      outs(%ye : tensor<1x2x4x4x4xbf16>) {to = 16 : i64} : tensor<1x2x4x4x4xbf16>
  return %y : tensor<1x2x4x4x4xbf16>
}

// A linear resize computes, so its casts stay.
// CHECK-LABEL: func.func @linear_resize
// CHECK: hip.cast
// CHECK: hip.resize{{.*}}tensor<1x2x4x4xf32>
// CHECK: hip.cast
func.func @linear_resize(%ctx: !hip.context, %x: tensor<1x2x4x4xf16>)
    -> tensor<1x2x8x8xf16> {
  %we = tensor.empty() : tensor<1x2x4x4xf32>
  %w = hip.cast(%ctx) ins(%x : tensor<1x2x4x4xf16>)
      outs(%we : tensor<1x2x4x4xf32>) {to = 1 : i64} : tensor<1x2x4x4xf32>
  %re = tensor.empty() : tensor<1x2x8x8xf32>
  %r = hip.resize(%ctx) ins(%w : tensor<1x2x4x4xf32>)
                        outs(%re : tensor<1x2x8x8xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x2x8x8xf32>
  %ye = tensor.empty() : tensor<1x2x8x8xf16>
  %y = hip.cast(%ctx) ins(%r : tensor<1x2x8x8xf32>)
      outs(%ye : tensor<1x2x8x8xf16>) {to = 10 : i64} : tensor<1x2x8x8xf16>
  return %y : tensor<1x2x8x8xf16>
}

// A narrowing inner cast rounds, so the round trip is not exact.
// CHECK-LABEL: func.func @narrowing_inner
// CHECK: hip.cast{{.*}}tensor<1x8xf32>) outs({{.*}}tensor<1x8xf16>)
// CHECK: hip.transpose{{.*}}tensor<1x8xf16>
// CHECK: hip.cast{{.*}}tensor<8x1xf16>) outs({{.*}}tensor<8x1xf32>)
func.func @narrowing_inner(%ctx: !hip.context, %x: tensor<1x8xf32>)
    -> tensor<8x1xf32> {
  %we = tensor.empty() : tensor<1x8xf16>
  %w = hip.cast(%ctx) ins(%x : tensor<1x8xf32>)
      outs(%we : tensor<1x8xf16>) {to = 10 : i64} : tensor<1x8xf16>
  %te = tensor.empty() : tensor<8x1xf16>
  %t = hip.transpose(%ctx) ins(%w : tensor<1x8xf16>)
      outs(%te : tensor<8x1xf16>) {perm = [1, 0]} : tensor<8x1xf16>
  %ye = tensor.empty() : tensor<8x1xf32>
  %y = hip.cast(%ctx) ins(%t : tensor<8x1xf16>)
      outs(%ye : tensor<8x1xf32>) {to = 1 : i64} : tensor<8x1xf32>
  return %y : tensor<8x1xf32>
}

// The f32 resize result has another reader, so the f32 resize must stay.
// CHECK-LABEL: func.func @multi_use
// CHECK: hip.cast
// CHECK: hip.resize{{.*}}tensor<1x2x4x4xf32>
// CHECK: hip.cast
func.func @multi_use(%ctx: !hip.context, %x: tensor<1x2x4x4xf16>)
    -> (tensor<1x2x8x8xf16>, tensor<1x2x8x8xf32>) {
  %we = tensor.empty() : tensor<1x2x4x4xf32>
  %w = hip.cast(%ctx) ins(%x : tensor<1x2x4x4xf16>)
      outs(%we : tensor<1x2x4x4xf32>) {to = 1 : i64} : tensor<1x2x4x4xf32>
  %re = tensor.empty() : tensor<1x2x8x8xf32>
  %r = hip.resize(%ctx) ins(%w : tensor<1x2x4x4xf32>)
                        outs(%re : tensor<1x2x8x8xf32>)
       {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 2 : i64}
       : tensor<1x2x8x8xf32>
  %ye = tensor.empty() : tensor<1x2x8x8xf16>
  %y = hip.cast(%ctx) ins(%r : tensor<1x2x8x8xf32>)
      outs(%ye : tensor<1x2x8x8xf16>) {to = 10 : i64} : tensor<1x2x8x8xf16>
  return %y, %r : tensor<1x2x8x8xf16>, tensor<1x2x8x8xf32>
}

// The outer cast goes to bf16, not back to the f16 source.
// CHECK-LABEL: func.func @other_outer_type
// CHECK: hip.cast
// CHECK: hip.transpose{{.*}}tensor<1x8xf32>
// CHECK: hip.cast
func.func @other_outer_type(%ctx: !hip.context, %x: tensor<1x8xf16>)
    -> tensor<8x1xbf16> {
  %we = tensor.empty() : tensor<1x8xf32>
  %w = hip.cast(%ctx) ins(%x : tensor<1x8xf16>)
      outs(%we : tensor<1x8xf32>) {to = 1 : i64} : tensor<1x8xf32>
  %te = tensor.empty() : tensor<8x1xf32>
  %t = hip.transpose(%ctx) ins(%w : tensor<1x8xf32>)
      outs(%te : tensor<8x1xf32>) {perm = [1, 0]} : tensor<8x1xf32>
  %ye = tensor.empty() : tensor<8x1xbf16>
  %y = hip.cast(%ctx) ins(%t : tensor<8x1xf32>)
      outs(%ye : tensor<8x1xbf16>) {to = 16 : i64} : tensor<8x1xbf16>
  return %y : tensor<8x1xbf16>
}

// An int to float widening is not a float widening.
// CHECK-LABEL: func.func @int_source
// CHECK: hip.cast
// CHECK: hip.transpose{{.*}}tensor<1x8xf32>
// CHECK: hip.cast
func.func @int_source(%ctx: !hip.context, %x: tensor<1x8xi16>)
    -> tensor<8x1xi16> {
  %we = tensor.empty() : tensor<1x8xf32>
  %w = hip.cast(%ctx) ins(%x : tensor<1x8xi16>)
      outs(%we : tensor<1x8xf32>) {to = 1 : i64} : tensor<1x8xf32>
  %te = tensor.empty() : tensor<8x1xf32>
  %t = hip.transpose(%ctx) ins(%w : tensor<1x8xf32>)
      outs(%te : tensor<8x1xf32>) {perm = [1, 0]} : tensor<8x1xf32>
  %ye = tensor.empty() : tensor<8x1xi16>
  %y = hip.cast(%ctx) ins(%t : tensor<8x1xf32>)
      outs(%ye : tensor<8x1xi16>) {to = 5 : i64} : tensor<8x1xi16>
  return %y : tensor<8x1xi16>
}

// A dynamic dim stays: the rebuilt init would need its extent.
// CHECK-LABEL: func.func @dynamic_dims
// CHECK: hip.cast
// CHECK: hip.transpose{{.*}}tensor<?x8xf32>
// CHECK: hip.cast
func.func @dynamic_dims(%ctx: !hip.context, %x: tensor<?x8xf16>)
    -> tensor<8x?xf16> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x8xf16>
  %we = tensor.empty(%n) : tensor<?x8xf32>
  %w = hip.cast(%ctx) ins(%x : tensor<?x8xf16>)
      outs(%we : tensor<?x8xf32>) {to = 1 : i64} : tensor<?x8xf32>
  %te = tensor.empty(%n) : tensor<8x?xf32>
  %t = hip.transpose(%ctx) ins(%w : tensor<?x8xf32>)
      outs(%te : tensor<8x?xf32>) {perm = [1, 0]} : tensor<8x?xf32>
  %ye = tensor.empty(%n) : tensor<8x?xf16>
  %y = hip.cast(%ctx) ins(%t : tensor<8x?xf32>)
      outs(%ye : tensor<8x?xf16>) {to = 10 : i64} : tensor<8x?xf16>
  return %y : tensor<8x?xf16>
}
