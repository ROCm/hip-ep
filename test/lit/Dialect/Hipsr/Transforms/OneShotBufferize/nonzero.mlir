// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.nonzero has two inits and two results, so it shows that each result
// takes the buffer of its own init. The bufferized op keeps no result.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// The coordinate buffer holds one column per input element, so the second
// nonzero pairs a dynamic indices init with a static one-element count.
// CHECK-LABEL: func.func @nonzero(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[MASK:.+]]: memref<2x3xi1, #hipsr.mem<device>>,
// CHECK-SAME: %[[DYN:.+]]: memref<?x?xi1, #hipsr.mem<device>>)
// CHECK-SAME: -> (memref<2x6xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>, memref<2x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[IDS:.+]] = memref.alloc() {{.*}}: memref<2x6xi64, #hipsr.mem<device>>
// CHECK-NEXT: %[[COUNT:.+]] = memref.alloc() {{.*}}: memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.nonzero(%[[CTX]]) ins(%[[MASK]] : memref<2x3xi1, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[IDS]], %[[COUNT]] : memref<2x6xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>)
// CHECK-NEXT: %[[D0:.+]] = memref.dim %[[DYN]], %[[C0]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT: %[[D1:.+]] = memref.dim %[[DYN]], %[[C1]] : memref<?x?xi1, #hipsr.mem<device>>
// CHECK-NEXT: %[[CAPACITY:.+]] = arith.muli %[[D0]], %[[D1]] : index
// CHECK-NEXT: %[[DYN_IDS:.+]] = memref.alloc(%[[CAPACITY]]) {{.*}}: memref<2x?xi64, #hipsr.mem<device>>
// CHECK-NEXT: %[[DYN_COUNT:.+]] = memref.alloc() {{.*}}: memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.nonzero(%[[CTX]]) ins(%[[DYN]] : memref<?x?xi1, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[DYN_IDS]], %[[DYN_COUNT]] : memref<2x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[IDS]], %[[COUNT]], %[[DYN_IDS]], %[[DYN_COUNT]] : memref<2x6xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>, memref<2x?xi64, #hipsr.mem<device>>, memref<1xi64, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @nonzero(%ctx: !hipsr.context,
                   %mask: tensor<2x3xi1, #hipsr.mem<device>>,
                   %dyn: tensor<?x?xi1, #hipsr.mem<device>>)
    -> (tensor<2x6xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>,
        tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ids_init = tensor.empty() : tensor<2x6xi64, #hipsr.mem<device>>
  %count_init = tensor.empty() : tensor<1xi64, #hipsr.mem<device>>
  %ids, %count = hipsr.nonzero(%ctx) ins(%mask : tensor<2x3xi1, #hipsr.mem<device>>)
      outs(%ids_init, %count_init : tensor<2x6xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>)
      : tensor<2x6xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
  %d0 = tensor.dim %dyn, %c0 : tensor<?x?xi1, #hipsr.mem<device>>
  %d1 = tensor.dim %dyn, %c1 : tensor<?x?xi1, #hipsr.mem<device>>
  %capacity = arith.muli %d0, %d1 : index
  %dyn_ids_init = tensor.empty(%capacity) : tensor<2x?xi64, #hipsr.mem<device>>
  %dyn_count_init = tensor.empty() : tensor<1xi64, #hipsr.mem<device>>
  %dyn_ids, %dyn_count = hipsr.nonzero(%ctx) ins(%dyn : tensor<?x?xi1, #hipsr.mem<device>>)
      outs(%dyn_ids_init, %dyn_count_init : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>)
      : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
  return %ids, %count, %dyn_ids, %dyn_count
      : tensor<2x6xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>,
        tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
}
