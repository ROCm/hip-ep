// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks that One-Shot Bufferize rewrites hipsr.pool_domain, its body, and its
// hipsr.pool_domain_yield terminator into memrefs.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" %s | FileCheck %s --check-prefixes=NOCOPY,NOTENSOR

// Nothing is written twice, so no buffer needs duplicating.
// NOCOPY-NOT: memref.copy

// CHECK-LABEL: func.func @mlp_domain(
// CHECK-SAME: %[[CTX:.+]]: !hip.context, %[[INPUT:.+]]: memref<4x256xf16>, %[[WEIGHT:.+]]: memref<256x512xf16>, %[[BIAS:.+]]: memref<4x512xf16>) -> memref<4x512xf16> {
// CHECK-NEXT: %[[OUT:.+]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]], %[[WEIGHT]], %[[BIAS]] : !hip.context, memref<4x256xf16>, memref<256x512xf16>, memref<4x512xf16>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.+]]: !hip.context, %[[IN:.+]]: memref<4x256xf16>, %[[W:.+]]: memref<256x512xf16>, %[[B:.+]]: memref<4x512xf16>):
// CHECK-NEXT: %[[MM:.+]] = memref.alloc(){{.*}} : memref<4x512xf16>
// CHECK-NEXT: hip.matmul(%[[DCTX]]) ins(%[[IN]], %[[W]] : memref<4x256xf16>, memref<256x512xf16>) outs(%[[MM]] : memref<4x512xf16>)
// CHECK-NEXT: %[[SUM:.+]] = memref.alloc(){{.*}} : memref<4x512xf16>
// CHECK-NEXT: hip.add(%[[DCTX]]) ins(%[[MM]], %[[B]] : memref<4x512xf16>, memref<4x512xf16>) outs(%[[SUM]] : memref<4x512xf16>)
// CHECK-NEXT: hipsr.pool_domain_yield %[[SUM]] : memref<4x512xf16>
// CHECK-NEXT: } -> memref<4x512xf16> {domain_id = 0 : i64}
// CHECK-NEXT: return %[[OUT]] : memref<4x512xf16>
// CHECK-NEXT: }
func.func @mlp_domain(%ctx: !hip.context, %input: tensor<4x256xf16>,
                      %weight: tensor<256x512xf16>, %bias: tensor<4x512xf16>)
    -> tensor<4x512xf16> {
  %out = hipsr.pool_domain(%ctx, %input, %weight, %bias
      : !hip.context, tensor<4x256xf16>, tensor<256x512xf16>,
        tensor<4x512xf16>) {
  ^bb0(%dctx: !hip.context, %in: tensor<4x256xf16>,
       %w: tensor<256x512xf16>, %b: tensor<4x512xf16>):
    %mm_init = tensor.empty() : tensor<4x512xf16>
    %mm = hip.matmul(%dctx) ins(%in, %w : tensor<4x256xf16>,
                                          tensor<256x512xf16>)
                            outs(%mm_init : tensor<4x512xf16>)
                            : tensor<4x512xf16>
    %sum_init = tensor.empty() : tensor<4x512xf16>
    %sum = hip.add(%dctx) ins(%mm, %b : tensor<4x512xf16>, tensor<4x512xf16>)
                          outs(%sum_init : tensor<4x512xf16>)
                          -> tensor<4x512xf16>
    hipsr.pool_domain_yield %sum : tensor<4x512xf16>
  } -> tensor<4x512xf16> {domain_id = 0 : i64}
  return %out : tensor<4x512xf16>
}
