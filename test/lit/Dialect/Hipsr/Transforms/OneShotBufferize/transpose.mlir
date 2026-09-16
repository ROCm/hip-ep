// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// The destination shape differs from the input's, so the buffer cannot be the
// input's and comes from the destination.
// CHECK-LABEL: func.func @transpose(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.+]]: memref<3x1024xi64, #hipsr.mem<device>>) -> memref<1024x3xi64, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[OUT:.+]] = memref.alloc() {{.*}}: memref<1024x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:    hipsr.transpose(%[[CTX]]) ins(%[[INPUT]] : memref<3x1024xi64, #hipsr.mem<device>>) outs(%[[OUT]] : memref<1024x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>}
// CHECK-NEXT:    return %[[OUT]] : memref<1024x3xi64, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @transpose(%ctx: !hipsr.context,
                     %input: tensor<3x1024xi64, #hipsr.mem<device>>)
    -> tensor<1024x3xi64, #hipsr.mem<device>> {
  %init = tensor.empty() : tensor<1024x3xi64, #hipsr.mem<device>>
  %0 = hipsr.transpose(%ctx)
      ins(%input : tensor<3x1024xi64, #hipsr.mem<device>>)
      outs(%init : tensor<1024x3xi64, #hipsr.mem<device>>)
      {perm = array<i64: 1, 0>} : tensor<1024x3xi64, #hipsr.mem<device>>
  return %0 : tensor<1024x3xi64, #hipsr.mem<device>>
}
