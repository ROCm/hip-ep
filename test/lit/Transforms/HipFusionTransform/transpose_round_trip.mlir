// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// The permutations are inverse but not self-inverse.
// CHECK-LABEL: func.func @inverse
// CHECK-SAME: %[[X:[^, :]+]]: tensor<2x3x4xf32>
// CHECK-NOT: hip.transpose
// CHECK: return %[[X]] : tensor<2x3x4xf32>
func.func @inverse(%ctx: !hip.context, %x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %e0 = tensor.empty() : tensor<3x4x2xf32>
  %a = hip.transpose(%ctx) ins(%x : tensor<2x3x4xf32>) outs(%e0 : tensor<3x4x2xf32>) {perm = [1, 2, 0]} : tensor<3x4x2xf32>
  %e1 = tensor.empty() : tensor<2x3x4xf32>
  %b = hip.transpose(%ctx) ins(%a : tensor<3x4x2xf32>) outs(%e1 : tensor<2x3x4xf32>) {perm = [2, 0, 1]} : tensor<2x3x4xf32>
  return %b : tensor<2x3x4xf32>
}

// -----

// The first transpose still serves another consumer.
// CHECK-LABEL: func.func @shared
// CHECK-SAME: %[[X:[^, :]+]]: tensor<2x3xf32>
// CHECK: %[[FIRST:.*]] = hip.transpose
// CHECK-NOT: hip.transpose
// CHECK: return %[[FIRST]], %[[X]]
func.func @shared(%ctx: !hip.context, %x: tensor<2x3xf32>) -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  %e0 = tensor.empty() : tensor<3x2xf32>
  %a = hip.transpose(%ctx) ins(%x : tensor<2x3xf32>) outs(%e0 : tensor<3x2xf32>) {perm = [1, 0]} : tensor<3x2xf32>
  %e1 = tensor.empty() : tensor<2x3xf32>
  %b = hip.transpose(%ctx) ins(%a : tensor<3x2xf32>) outs(%e1 : tensor<2x3xf32>) {perm = [1, 0]} : tensor<2x3xf32>
  return %a, %b : tensor<3x2xf32>, tensor<2x3xf32>
}

// -----

// Equal shapes do not imply inverse permutations.
// CHECK-LABEL: func.func @not_inverse
// CHECK: hip.transpose
// CHECK: hip.transpose
// CHECK: return
func.func @not_inverse(%ctx: !hip.context, %x: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %e0 = tensor.empty() : tensor<2x2x2xf32>
  %a = hip.transpose(%ctx) ins(%x : tensor<2x2x2xf32>) outs(%e0 : tensor<2x2x2xf32>) {perm = [1, 2, 0]} : tensor<2x2x2xf32>
  %e1 = tensor.empty() : tensor<2x2x2xf32>
  %b = hip.transpose(%ctx) ins(%a : tensor<2x2x2xf32>) outs(%e1 : tensor<2x2x2xf32>) {perm = [1, 2, 0]} : tensor<2x2x2xf32>
  return %b : tensor<2x2x2xf32>
}

// -----

// Different contexts cannot be assumed to share the same execution state.
// CHECK-LABEL: func.func @different_contexts
// CHECK: hip.transpose
// CHECK: hip.transpose
func.func @different_contexts(%ctx0: !hip.context, %ctx1: !hip.context, %x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %e0 = tensor.empty() : tensor<3x2xf32>
  %a = hip.transpose(%ctx0) ins(%x : tensor<2x3xf32>) outs(%e0 : tensor<3x2xf32>) {perm = [1, 0]} : tensor<3x2xf32>
  %e1 = tensor.empty() : tensor<2x3xf32>
  %b = hip.transpose(%ctx1) ins(%a : tensor<3x2xf32>) outs(%e1 : tensor<2x3xf32>) {perm = [1, 0]} : tensor<2x3xf32>
  return %b : tensor<2x3xf32>
}

// -----

// The tensor rewrite must not erase buffer writes.
// CHECK-LABEL: func.func @buffers
// CHECK: hip.transpose
// CHECK: hip.transpose
func.func @buffers(%ctx: !hip.context, %x: memref<2x3xf32>, %tmp: memref<3x2xf32>, %out: memref<2x3xf32>) {
  hip.transpose(%ctx) ins(%x : memref<2x3xf32>) outs(%tmp : memref<3x2xf32>) {perm = [1, 0]}
  hip.transpose(%ctx) ins(%tmp : memref<3x2xf32>) outs(%out : memref<2x3xf32>) {perm = [1, 0]}
  return
}

// -----

// Dynamic dimensions are forwarded when the permuted types agree.
// CHECK-LABEL: func.func @dynamic
// CHECK-SAME: %[[X:[^, :]+]]: tensor<?x3xf32>
// CHECK-NOT: hip.transpose
// CHECK: return %[[X]] : tensor<?x3xf32>
func.func @dynamic(%ctx: !hip.context, %x: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x3xf32>
  %e0 = tensor.empty(%n) : tensor<3x?xf32>
  %a = hip.transpose(%ctx) ins(%x : tensor<?x3xf32>) outs(%e0 : tensor<3x?xf32>) {perm = [1, 0]} : tensor<3x?xf32>
  %e1 = tensor.empty(%n) : tensor<?x3xf32>
  %b = hip.transpose(%ctx) ins(%a : tensor<3x?xf32>) outs(%e1 : tensor<?x3xf32>) {perm = [1, 0]} : tensor<?x3xf32>
  return %b : tensor<?x3xf32>
}

// -----

// Do not discard a static refinement of an unknown input extent.
// CHECK-LABEL: func.func @refined_middle
// CHECK: hip.transpose
// CHECK: hip.transpose
func.func @refined_middle(%ctx: !hip.context, %x: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x3xf32>
  %e0 = tensor.empty() : tensor<3x2xf32>
  %a = hip.transpose(%ctx) ins(%x : tensor<?x3xf32>) outs(%e0 : tensor<3x2xf32>) {perm = [1, 0]} : tensor<3x2xf32>
  %e1 = tensor.empty(%n) : tensor<?x3xf32>
  %b = hip.transpose(%ctx) ins(%a : tensor<3x2xf32>) outs(%e1 : tensor<?x3xf32>) {perm = [1, 0]} : tensor<?x3xf32>
  return %b : tensor<?x3xf32>
}
