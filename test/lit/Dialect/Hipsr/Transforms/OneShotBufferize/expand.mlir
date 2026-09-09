// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.expand keeps its data on the device and its extents on the host. Each
// operand keeps the memory space from its own encoding, so the two bufferize
// into different spaces.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// The first expand takes its shape from a host argument. The second builds a
// shape, which lands in a host allocation, while the data allocation next to
// it stays on the device.
// CHECK-LABEL: func.func @expand(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[INPUT:.+]]: memref<2x1xf32, #hipsr.mem<device>>,
// CHECK-SAME: %[[SHAPE:.+]]: memref<2xi64, #hipsr.mem<host>>,
// CHECK-SAME: %[[DYN:.+]]: memref<?x1xf32, #hipsr.mem<device>>)
// CHECK-SAME: -> (memref<2x4xf32, #hipsr.mem<device>>, memref<?x4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc() {{.*}}: memref<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.expand(%[[CTX]]) ins(%[[INPUT]], %[[SHAPE]] : memref<2x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>)
// CHECK-SAME: outs(%[[OUT]] : memref<2x4xf32, #hipsr.mem<device>>)
// CHECK-NEXT: %[[ROWS:.+]] = memref.dim %[[DYN]], %[[C0]] : memref<?x1xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[ROWS_I64:.+]] = arith.index_cast %[[ROWS]] : index to i64
// CHECK-NEXT: %[[BUILT:.+]] = memref.alloc() {{.*}}: memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT: %[[STORE0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[STORE1:.+]] = arith.constant 1 : index
// CHECK-NEXT: memref.store %[[ROWS_I64]], %[[BUILT]]{{\[}}%[[STORE0]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT: memref.store %[[C4]], %[[BUILT]]{{\[}}%[[STORE1]]] : memref<2xi64, #hipsr.mem<host>>
// CHECK-NEXT: %[[DYN_OUT:.+]] = memref.alloc(%[[ROWS]]) {{.*}}: memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.expand(%[[CTX]]) ins(%[[DYN]], %[[BUILT]] : memref<?x1xf32, #hipsr.mem<device>>, memref<2xi64, #hipsr.mem<host>>)
// CHECK-SAME: outs(%[[DYN_OUT]] : memref<?x4xf32, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]], %[[DYN_OUT]] : memref<2x4xf32, #hipsr.mem<device>>, memref<?x4xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @expand(%ctx: !hipsr.context,
                  %input: tensor<2x1xf32, #hipsr.mem<device>>,
                  %shape: tensor<2xi64, #hipsr.mem<host>>,
                  %dyn: tensor<?x1xf32, #hipsr.mem<device>>)
    -> (tensor<2x4xf32, #hipsr.mem<device>>, tensor<?x4xf32, #hipsr.mem<device>>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : i64
  %init = tensor.empty() : tensor<2x4xf32, #hipsr.mem<device>>
  %0 = hipsr.expand(%ctx) ins(%input, %shape : tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>)
      outs(%init : tensor<2x4xf32, #hipsr.mem<device>>)
      : tensor<2x4xf32, #hipsr.mem<device>>
  %rows = tensor.dim %dyn, %c0 : tensor<?x1xf32, #hipsr.mem<device>>
  %rows_i64 = arith.index_cast %rows : index to i64
  %built = tensor.from_elements %rows_i64, %c4 : tensor<2xi64, #hipsr.mem<host>>
  %dyn_init = tensor.empty(%rows) : tensor<?x4xf32, #hipsr.mem<device>>
  %1 = hipsr.expand(%ctx) ins(%dyn, %built : tensor<?x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>)
      outs(%dyn_init : tensor<?x4xf32, #hipsr.mem<device>>)
      : tensor<?x4xf32, #hipsr.mem<device>>
  return %0, %1 : tensor<2x4xf32, #hipsr.mem<device>>, tensor<?x4xf32, #hipsr.mem<device>>
}
