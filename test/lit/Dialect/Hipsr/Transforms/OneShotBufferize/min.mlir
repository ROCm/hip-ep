// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// CHECK-LABEL: func.func @min_static(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[LHS:.+]]: memref<4x1024xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[RHS:.+]]: memref<1024xf16, #hipsr.mem<device>>) -> memref<4x1024xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc() {{.*}}: memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.min(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]] : memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @min_static(%ctx: !hipsr.context,
                      %lhs: tensor<4x1024xf16, #hipsr.mem<device>>,
                      %rhs: tensor<1024xf16, #hipsr.mem<device>>)
    -> tensor<4x1024xf16, #hipsr.mem<device>> {
  %init = tensor.empty() : tensor<4x1024xf16, #hipsr.mem<device>>
  %0 = hipsr.min(%ctx) ins(%lhs, %rhs : tensor<4x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x1024xf16, #hipsr.mem<device>>)
      : tensor<4x1024xf16, #hipsr.mem<device>>
  return %0 : tensor<4x1024xf16, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @min_dynamic(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[LHS:.+]]: memref<?x1024xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[RHS:.+]]: memref<1024xf16, #hipsr.mem<device>>) -> memref<?x1024xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[D0:.+]] = memref.dim %[[LHS]], %[[C0]] : memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.min(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : memref<?x1024xf16, #hipsr.mem<device>>, memref<1024xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<?x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]] : memref<?x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @min_dynamic(%ctx: !hipsr.context,
                       %lhs: tensor<?x1024xf16, #hipsr.mem<device>>,
                       %rhs: tensor<1024xf16, #hipsr.mem<device>>)
    -> tensor<?x1024xf16, #hipsr.mem<device>> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x1024xf16, #hipsr.mem<device>>
  %init = tensor.empty(%d0) : tensor<?x1024xf16, #hipsr.mem<device>>
  %0 = hipsr.min(%ctx) ins(%lhs, %rhs : tensor<?x1024xf16, #hipsr.mem<device>>, tensor<1024xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?x1024xf16, #hipsr.mem<device>>)
      : tensor<?x1024xf16, #hipsr.mem<device>>
  return %0 : tensor<?x1024xf16, #hipsr.mem<device>>
}
