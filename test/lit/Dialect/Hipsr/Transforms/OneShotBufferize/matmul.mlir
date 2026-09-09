// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// The plain matmul allocates from its result type. The batched one has a
// dynamic batch extent, so it reads that extent from the lhs buffer.
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[A:.+]]: memref<64x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[BATCHED:.+]]: memref<?x64x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[B:.+]]: memref<4096x1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: -> (memref<64x1024xf16, #hipsr.mem<device>>, memref<?x64x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc() {{.*}}: memref<64x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : memref<64x4096xf16, #hipsr.mem<device>>, memref<4096x1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<64x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[BATCH:.+]] = memref.dim %[[BATCHED]], %[[C0]] : memref<?x64x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[BATCHED_OUT:.+]] = memref.alloc(%[[BATCH]]) {{.*}}: memref<?x64x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.matmul(%[[CTX]]) ins(%[[BATCHED]], %[[B]] : memref<?x64x4096xf16, #hipsr.mem<device>>, memref<4096x1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[BATCHED_OUT]] : memref<?x64x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]], %[[BATCHED_OUT]] : memref<64x1024xf16, #hipsr.mem<device>>, memref<?x64x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @matmul(%ctx: !hipsr.context,
                  %a: tensor<64x4096xf16, #hipsr.mem<device>>,
                  %batched: tensor<?x64x4096xf16, #hipsr.mem<device>>,
                  %b: tensor<4096x1024xf16, #hipsr.mem<device>>)
    -> (tensor<64x1024xf16, #hipsr.mem<device>>, tensor<?x64x1024xf16, #hipsr.mem<device>>) {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<64x1024xf16, #hipsr.mem<device>>
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<64x4096xf16, #hipsr.mem<device>>, tensor<4096x1024xf16, #hipsr.mem<device>>)
      outs(%init : tensor<64x1024xf16, #hipsr.mem<device>>)
      : tensor<64x1024xf16, #hipsr.mem<device>>
  %batch = tensor.dim %batched, %c0 : tensor<?x64x4096xf16, #hipsr.mem<device>>
  %batched_init = tensor.empty(%batch) : tensor<?x64x1024xf16, #hipsr.mem<device>>
  %1 = hipsr.matmul(%ctx) ins(%batched, %b : tensor<?x64x4096xf16, #hipsr.mem<device>>, tensor<4096x1024xf16, #hipsr.mem<device>>)
      outs(%batched_init : tensor<?x64x1024xf16, #hipsr.mem<device>>)
      : tensor<?x64x1024xf16, #hipsr.mem<device>>
  return %0, %1 : tensor<64x1024xf16, #hipsr.mem<device>>, tensor<?x64x1024xf16, #hipsr.mem<device>>
}
