// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// One placeholder covers both destinations, so the region yields two shapes.
// The indices get a row per input axis and, in the worst case, a column per
// input element. The count is one number. Only the row count comes from the
// type; the capacity stays symbolic.
// CHECK-LABEL: func.func @nonzero_mask(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[MASK:.+]]: tensor<?x?xi8, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INITS:.+]]:2 = hipsr.placeholder(%[[CTX]]) ins(%[[MASK]] : tensor<?x?xi8, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[MASK_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: %[[ROWS:.+]] = shape.const_size 2
// CHECK-NEXT: %[[CAPACITY:.+]] = shape.num_elements %[[MASK_SHAPE]] : !shape.shape -> !shape.size
// CHECK-NEXT: %[[INDICES_SHAPE:.+]] = shape.from_extents %[[ROWS]], %[[CAPACITY]] : !shape.size, !shape.size
// CHECK-NEXT: %[[COUNT_SHAPE:.+]] = shape.const_shape [1] : !shape.shape
// CHECK-NEXT: hipsr.shape_yield %[[INDICES_SHAPE]], %[[COUNT_SHAPE]] : !shape.shape, !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.nonzero(%[[CTX]]) ins(%[[MASK]] : tensor<?x?xi8, #hipsr.mem<device>>) outs(%[[INITS]]#0, %[[INITS]]#1 : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @nonzero_mask(%ctx: !hipsr.context,
                        %mask: tensor<?x?xi8, #hipsr.mem<device>>) {
  %indices_init, %count_init = hipsr.placeholder(%ctx)
      ins(%mask : tensor<?x?xi8, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<2x?xi64, #hipsr.mem<device>>,
        tensor<1xi64, #hipsr.mem<device>>
  %indices, %count = hipsr.nonzero(%ctx)
      ins(%mask : tensor<?x?xi8, #hipsr.mem<device>>)
      outs(%indices_init, %count_init
           : tensor<2x?xi64, #hipsr.mem<device>>,
             tensor<1xi64, #hipsr.mem<device>>)
      : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
  return
}
