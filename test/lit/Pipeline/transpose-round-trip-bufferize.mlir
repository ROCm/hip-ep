// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --hip-promote-strided-operands --hip-use-output-allocator --split-input-file %s | FileCheck %s

// Forwarding a view must not pass a strided pointer to a HIP runtime consumer.
// CHECK-LABEL: func.func @strided_input
// CHECK-NOT: hip.transpose
// CHECK: %[[VIEW:.*]] = memref.subview
// CHECK: %[[COPY:.*]] = memref.alloc() : memref<1x4x2xf32>
// CHECK: memref.copy %[[VIEW]], %[[COPY]]
// CHECK: hip.sigmoid{{.*}}ins(%[[COPY]] : memref<1x4x2xf32>)
func.func @strided_input(%ctx: !hip.context, %x: tensor<1x4x6xf32>) -> tensor<1x4x2xf32> {
  %slice = tensor.extract_slice %x[0, 0, 2] [1, 4, 2] [1, 1, 1] : tensor<1x4x6xf32> to tensor<1x4x2xf32>
  %e0 = tensor.empty() : tensor<1x2x4xf32>
  %a = hip.transpose(%ctx) ins(%slice : tensor<1x4x2xf32>) outs(%e0 : tensor<1x2x4xf32>) {perm = [0, 2, 1]} : tensor<1x2x4xf32>
  %e1 = tensor.empty() : tensor<1x4x2xf32>
  %b = hip.transpose(%ctx) ins(%a : tensor<1x2x4xf32>) outs(%e1 : tensor<1x4x2xf32>) {perm = [0, 2, 1]} : tensor<1x4x2xf32>
  %e2 = tensor.empty() : tensor<1x4x2xf32>
  %out = hip.sigmoid(%ctx) ins(%b : tensor<1x4x2xf32>) outs(%e2 : tensor<1x4x2xf32>) : tensor<1x4x2xf32>
  return %out : tensor<1x4x2xf32>
}

// -----

// A forwarded graph input still needs a runtime-owned output.
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:[^, :]+]]: !hip.context, %[[X:[^, :]+]]: memref<2x3xf32>)
// CHECK-NOT: hip.transpose
// CHECK: %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<2x3xf32>
// CHECK: memref.copy %[[X]], %[[OUT]]
// CHECK: return %[[OUT]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %e0 = tensor.empty() : tensor<3x2xf32>
  %a = hip.transpose(%ctx) ins(%x : tensor<2x3xf32>) outs(%e0 : tensor<3x2xf32>) {perm = [1, 0]} : tensor<3x2xf32>
  %e1 = tensor.empty() : tensor<2x3xf32>
  %b = hip.transpose(%ctx) ins(%a : tensor<3x2xf32>) outs(%e1 : tensor<2x3xf32>) {perm = [1, 0]} : tensor<2x3xf32>
  return %b : tensor<2x3xf32>
}
