// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// Gather cannot write in place: the result shape differs from the data's, so
// the buffer comes from the destination.
// CHECK-LABEL: func.func @gather(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[TABLE:.+]]: memref<8x4xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[IDS:.+]]: memref<3xi64, #hipsr.mem<device>>) -> memref<3x4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[OUT:.+]] = memref.alloc() {{.*}}: memref<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    hipsr.gather(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : memref<8x4xf16, #hipsr.mem<device>>, memref<3xi64, #hipsr.mem<device>>) outs(%[[OUT]] : memref<3x4xf16, #hipsr.mem<device>>) {axis = 0 : i64}
// CHECK-NEXT:    return %[[OUT]] : memref<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @gather(%ctx: !hipsr.context,
                  %table: tensor<8x4xf16, #hipsr.mem<device>>,
                  %ids: tensor<3xi64, #hipsr.mem<device>>)
    -> tensor<3x4xf16, #hipsr.mem<device>> {
  %init = tensor.empty() : tensor<3x4xf16, #hipsr.mem<device>>
  %0 = hipsr.gather(%ctx)
      ins(%table, %ids : tensor<8x4xf16, #hipsr.mem<device>>,
                         tensor<3xi64, #hipsr.mem<device>>)
      outs(%init : tensor<3x4xf16, #hipsr.mem<device>>) {axis = 0 : i64}
      : tensor<3x4xf16, #hipsr.mem<device>>
  return %0 : tensor<3x4xf16, #hipsr.mem<device>>
}
