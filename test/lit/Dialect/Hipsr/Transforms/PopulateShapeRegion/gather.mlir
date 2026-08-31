// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// The region reads one extent per output dimension, in this order:
//
//   - the data dimensions before the gathered axis;
//   - every indices dimension;
//   - the data dimensions after the axis.
//
// Both ranks are known, so the loop that assembles the extents runs at compile
// time and a dynamic dimension stays symbolic.
// CHECK-LABEL: func.func @gather_embedding(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[TABLE:.+]]: tensor<8x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[IDS:.+]]: tensor<?x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[TABLE_SHAPE:.+]]: tensor<2xindex>, %[[IDS_SHAPE:.+]]: tensor<2xindex>):
// CHECK-NEXT: %[[IDS_D0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[IDS_E0:.+]] = shape.get_extent %[[IDS_SHAPE]], %[[IDS_D0]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[IDS_D1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[IDS_E1:.+]] = shape.get_extent %[[IDS_SHAPE]], %[[IDS_D1]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[TABLE_D1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[TABLE_E1:.+]] = shape.get_extent %[[TABLE_SHAPE]], %[[TABLE_D1]] : tensor<2xindex>, index -> index
// CHECK-NEXT: %[[SHAPE:.+]] = tensor.from_elements %[[IDS_E0]], %[[IDS_E1]], %[[TABLE_E1]] : tensor<3xindex>
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : tensor<3xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.gather(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64} : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @gather_embedding(%ctx: !hipsr.context,
                            %table: tensor<8x4096xf16, #hipsr.mem<device>>,
                            %ids: tensor<?x?xi64, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%table, %ids : tensor<8x4096xf16, #hipsr.mem<device>>,
                         tensor<?x?xi64, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x?x4096xf16, #hipsr.mem<device>>
  %result = hipsr.gather(%ctx)
      ins(%table, %ids : tensor<8x4096xf16, #hipsr.mem<device>>,
                         tensor<?x?xi64, #hipsr.mem<device>>)
      outs(%init : tensor<?x?x4096xf16, #hipsr.mem<device>>) {axis = 0 : i64}
      : tensor<?x?x4096xf16, #hipsr.mem<device>>
  return
}

// Gathering along an inner axis is the only case with data dimensions on both
// sides of the indices, so it is what pins the order down.
// CHECK-LABEL: func.func @gather_middle_axis(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[DATA:.+]]: tensor<8x4x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[IDS:.+]]: tensor<?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[DATA]], %[[IDS]] : tensor<8x4x4096xf16, #hipsr.mem<device>>, tensor<?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<8x?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[DATA_SHAPE:.+]]: tensor<3xindex>, %[[IDS_SHAPE:.+]]: tensor<1xindex>):
// CHECK-NEXT: %[[DATA_D0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DATA_E0:.+]] = shape.get_extent %[[DATA_SHAPE]], %[[DATA_D0]] : tensor<3xindex>, index -> index
// CHECK-NEXT: %[[IDS_D0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[IDS_E0:.+]] = shape.get_extent %[[IDS_SHAPE]], %[[IDS_D0]] : tensor<1xindex>, index -> index
// CHECK-NEXT: %[[DATA_D2:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[DATA_E2:.+]] = shape.get_extent %[[DATA_SHAPE]], %[[DATA_D2]] : tensor<3xindex>, index -> index
// CHECK-NEXT: %[[SHAPE:.+]] = tensor.from_elements %[[DATA_E0]], %[[IDS_E0]], %[[DATA_E2]] : tensor<3xindex>
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : tensor<3xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.gather(%[[CTX]]) ins(%[[DATA]], %[[IDS]] : tensor<8x4x4096xf16, #hipsr.mem<device>>, tensor<?xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<8x?x4096xf16, #hipsr.mem<device>>) {axis = 1 : i64} : tensor<8x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @gather_middle_axis(%ctx: !hipsr.context,
                              %data: tensor<8x4x4096xf16, #hipsr.mem<device>>,
                              %ids: tensor<?xi64, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%data, %ids : tensor<8x4x4096xf16, #hipsr.mem<device>>,
                        tensor<?xi64, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<8x?x4096xf16, #hipsr.mem<device>>
  %result = hipsr.gather(%ctx)
      ins(%data, %ids : tensor<8x4x4096xf16, #hipsr.mem<device>>,
                        tensor<?xi64, #hipsr.mem<device>>)
      outs(%init : tensor<8x?x4096xf16, #hipsr.mem<device>>) {axis = 1 : i64}
      : tensor<8x?x4096xf16, #hipsr.mem<device>>
  return
}
