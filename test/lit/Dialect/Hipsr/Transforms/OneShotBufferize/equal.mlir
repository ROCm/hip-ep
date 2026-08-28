// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// The mask buffer takes ui8 from the destination, not from the operands.
// CHECK-LABEL: func.func @equal(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[LHS:.+]]: memref<4x1024xi64, #hipsr.mem<device>>,
// CHECK-SAME:    %[[RHS:.+]]: memref<1024xi64, #hipsr.mem<device>>) -> memref<4x1024xui8, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[OUT:.+]] = memref.alloc() {{.*}}: memref<4x1024xui8, #hipsr.mem<device>>
// CHECK-NEXT:    hipsr.equal(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : memref<4x1024xi64, #hipsr.mem<device>>, memref<1024xi64, #hipsr.mem<device>>) outs(%[[OUT]] : memref<4x1024xui8, #hipsr.mem<device>>)
// CHECK-NEXT:    return %[[OUT]] : memref<4x1024xui8, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @equal(%ctx: !hipsr.context,
                 %lhs: tensor<4x1024xi64, #hipsr.mem<device>>,
                 %rhs: tensor<1024xi64, #hipsr.mem<device>>)
    -> tensor<4x1024xui8, #hipsr.mem<device>> {
  %init = tensor.empty() : tensor<4x1024xui8, #hipsr.mem<device>>
  %0 = hipsr.equal(%ctx) ins(%lhs, %rhs : tensor<4x1024xi64, #hipsr.mem<device>>, tensor<1024xi64, #hipsr.mem<device>>)
      outs(%init : tensor<4x1024xui8, #hipsr.mem<device>>)
      : tensor<4x1024xui8, #hipsr.mem<device>>
  return %0 : tensor<4x1024xui8, #hipsr.mem<device>>
}
