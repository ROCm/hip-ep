// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// Both adds lose their result and write through the outs buffer instead, so
// the return hands back those buffers. The static init allocates from its
// type. The dynamic one gets its extent from a memref.dim of the operand.
// CHECK-LABEL: func.func @add(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[LHS:.+]]: memref<4x1024xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[DYN:.+]]: memref<?x1024xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[RHS:.+]]: memref<1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: -> (memref<4x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc() {{.*}}: memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[ROWS:.+]] = memref.dim %[[DYN]], %[[C0]] : memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[DYN_OUT:.+]] = memref.alloc(%[[ROWS]]) {{.*}}: memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%[[CTX]]) ins(%[[DYN]], %[[RHS]] : memref<?x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[DYN_OUT]] : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]], %[[DYN_OUT]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @add(%ctx: !hipsr.context,
               %lhs: tensor<4x1024xf16, #hipsr.mem<device>>,
               %dyn: tensor<?x1024xf16, #hipsr.mem<device>>,
               %rhs: tensor<1024xf16, #hipsr.mem<device>>)
    -> (tensor<4x1024xf16, #hipsr.mem<device>>, tensor<?x1024xf16, #hipsr.mem<device>>) {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<4x1024xf16, #hipsr.mem<device>>
  %0 = hipsr.add(%ctx) ins(%lhs, %rhs : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x1024xf16, #hipsr.mem<device>>)
      : tensor<4x1024xf16, #hipsr.mem<device>>
  %rows = tensor.dim %dyn, %c0 : tensor<?x1024xf16, #hipsr.mem<device>>
  %dyn_init = tensor.empty(%rows) : tensor<?x1024xf16, #hipsr.mem<device>>
  %1 = hipsr.add(%ctx) ins(%dyn, %rhs : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>)
      outs(%dyn_init : tensor<?x1024xf16, #hipsr.mem<device>>)
      : tensor<?x1024xf16, #hipsr.mem<device>>
  return %0, %1 : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<?x1024xf16, #hipsr.mem<device>>
}
