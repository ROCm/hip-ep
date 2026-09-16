// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// The region splits the data shape around the gathered axis and puts the whole
// indices shape between the pieces. Nothing here depends on a rank, and a
// dynamic dimension stays symbolic.
// CHECK-LABEL: func.func @gather_embedding(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[TABLE:.+]]: tensor<8x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[IDS:.+]]: tensor<?x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[TABLE]], %[[IDS]] : tensor<8x4096xf16, #hipsr.mem<device>>, tensor<?x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[TABLE_SHAPE:.+]]: !shape.shape, %[[IDS_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: %[[AXIS:.+]] = shape.const_size 0
// CHECK-NEXT: %[[LEADING:.+]], %{{.+}} = "shape.split_at"(%[[TABLE_SHAPE]], %[[AXIS]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT: %[[PAST_AXIS:.+]] = shape.const_size 1
// CHECK-NEXT: %{{.+}}, %[[TRAILING:.+]] = "shape.split_at"(%[[TABLE_SHAPE]], %[[PAST_AXIS]]) : (!shape.shape, !shape.size) -> (!shape.shape, !shape.shape)
// CHECK-NEXT: %[[GATHERED:.+]] = shape.concat %[[LEADING]], %[[IDS_SHAPE]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: %[[SHAPE:.+]] = shape.concat %[[GATHERED]], %[[TRAILING]] : !shape.shape, !shape.shape -> !shape.shape
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
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
