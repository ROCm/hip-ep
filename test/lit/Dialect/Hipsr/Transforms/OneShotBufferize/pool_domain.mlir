// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Checks that One-Shot Bufferize rewrites hipsr.pool_domain, its body, and its
// hipsr.pool_domain_yield terminator into memrefs.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s --check-prefixes=NOCOPY,NOTENSOR

// Nothing is written twice, so no buffer needs duplicating.
// NOCOPY-NOT: memref.copy
// NOTENSOR-NOT: tensor<
// NOTENSOR-NOT: bufferization.to_

// CHECK-LABEL: func.func @cast_domain(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[INPUT:.+]]: memref<4x256xf16, #hipsr.mem<device>>) -> memref<4x256xi32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[OUT:.+]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, memref<4x256xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[DCTX:.+]]: !hipsr.context, %[[IN:.+]]: memref<4x256xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[MID:.+]] = memref.alloc(){{.*}} : memref<4x256xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[DCTX]]) ins(%[[IN]] : memref<4x256xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[MID]] : memref<4x256xf32, #hipsr.mem<device>>)
// CHECK-NEXT: %[[RES:.+]] = memref.alloc(){{.*}} : memref<4x256xi32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[DCTX]]) ins(%[[MID]] : memref<4x256xf32, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[RES]] : memref<4x256xi32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.pool_domain_yield %[[RES]] : memref<4x256xi32, #hipsr.mem<device>>
// CHECK-NEXT: } -> memref<4x256xi32, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: return %[[OUT]] : memref<4x256xi32, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @cast_domain(%ctx: !hipsr.context,
                       %input: tensor<4x256xf16, #hipsr.mem<device>>)
    -> tensor<4x256xi32, #hipsr.mem<device>> {
  %out = hipsr.pool_domain(%ctx, %input
      : !hipsr.context, tensor<4x256xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %in: tensor<4x256xf16, #hipsr.mem<device>>):
    %mid_init = tensor.empty() : tensor<4x256xf32, #hipsr.mem<device>>
    %mid = hipsr.cast(%dctx) ins(%in : tensor<4x256xf16, #hipsr.mem<device>>)
        outs(%mid_init : tensor<4x256xf32, #hipsr.mem<device>>)
        : tensor<4x256xf32, #hipsr.mem<device>>
    %res_init = tensor.empty() : tensor<4x256xi32, #hipsr.mem<device>>
    %res = hipsr.cast(%dctx) ins(%mid : tensor<4x256xf32, #hipsr.mem<device>>)
        outs(%res_init : tensor<4x256xi32, #hipsr.mem<device>>)
        : tensor<4x256xi32, #hipsr.mem<device>>
    hipsr.pool_domain_yield %res : tensor<4x256xi32, #hipsr.mem<device>>
  } -> tensor<4x256xi32, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %out : tensor<4x256xi32, #hipsr.mem<device>>
}
