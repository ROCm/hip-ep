// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.random_normal_like is a destination-passing op. One-Shot Bufferize
// must rewrite its tensor operands to memrefs and drop the tensor result,
// writing through the init buffer.

// RUN: hip-mlir-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s

// CHECK-LABEL: func.func @random_normal_like(
// CHECK-SAME: %[[CTX:.+]]: !hip.context,
// CHECK-SAME: %[[INPUT:.+]]: memref<1x2x60xf32>) -> memref<1x2x60xf32> {
// CHECK: %[[OUT:.+]] = memref.alloc() {{.*}}: memref<1x2x60xf32>
// CHECK: hip.random_normal_like(%[[CTX]]) ins(%[[INPUT]] : memref<1x2x60xf32>) outs(%[[OUT]] : memref<1x2x60xf32>)
// CHECK: return %[[OUT]] : memref<1x2x60xf32>
func.func @random_normal_like(%ctx: !hip.context, %input: tensor<1x2x60xf32>) -> tensor<1x2x60xf32> {
  %init = tensor.empty() : tensor<1x2x60xf32>
  %out = hip.random_normal_like(%ctx) ins(%input : tensor<1x2x60xf32>) outs(%init : tensor<1x2x60xf32>) {mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32} : tensor<1x2x60xf32>
  return %out : tensor<1x2x60xf32>
}
