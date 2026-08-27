// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// The runtime copies the data into the destination before it writes, so the
// destination gets a buffer of its own rather than the data's.
// CHECK-LABEL: func.func @scatter_nd(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[DATA:.+]]: memref<4x2xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[IDS:.+]]: memref<3x2xi64, #hipsr.mem<device>>,
// CHECK-SAME:    %[[UPDATES:.+]]: memref<3xf16, #hipsr.mem<device>>) -> memref<4x2xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[OUT:.+]] = memref.alloc() {{.*}}: memref<4x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:    hipsr.scatter_nd(%[[CTX]]) ins(%[[DATA]], %[[IDS]], %[[UPDATES]] : memref<4x2xf16, #hipsr.mem<device>>, memref<3x2xi64, #hipsr.mem<device>>, memref<3xf16, #hipsr.mem<device>>) outs(%[[OUT]] : memref<4x2xf16, #hipsr.mem<device>>)
// CHECK-NEXT:    return %[[OUT]] : memref<4x2xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @scatter_nd(%ctx: !hipsr.context,
                      %data: tensor<4x2xf16, #hipsr.mem<device>>,
                      %ids: tensor<3x2xi64, #hipsr.mem<device>>,
                      %updates: tensor<3xf16, #hipsr.mem<device>>)
    -> tensor<4x2xf16, #hipsr.mem<device>> {
  %init = tensor.empty() : tensor<4x2xf16, #hipsr.mem<device>>
  %0 = hipsr.scatter_nd(%ctx)
      ins(%data, %ids, %updates : tensor<4x2xf16, #hipsr.mem<device>>,
                                  tensor<3x2xi64, #hipsr.mem<device>>,
                                  tensor<3xf16, #hipsr.mem<device>>)
      outs(%init : tensor<4x2xf16, #hipsr.mem<device>>)
      : tensor<4x2xf16, #hipsr.mem<device>>
  return %0 : tensor<4x2xf16, #hipsr.mem<device>>
}
