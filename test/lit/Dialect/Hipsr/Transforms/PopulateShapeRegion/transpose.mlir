// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// The region reads one extent per output axis off the input's shape, so a
// dynamic extent moves with its axis.
// CHECK-LABEL: func.func @transpose_normal(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[INPUT:.+]]: tensor<3x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<3x?xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x3xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[IN_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: %[[AXIS1:.+]] = shape.const_size 1
// CHECK-NEXT: %[[COLS:.+]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS1]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[AXIS0:.+]] = shape.const_size 0
// CHECK-NEXT: %[[ROWS:.+]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS0]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[COLS]], %[[ROWS]] : !shape.size, !shape.size
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.transpose(%[[CTX]]) ins(%[[INPUT]] : tensor<3x?xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x3xi64, #hipsr.mem<device>>) {perm = array<i64: 1, 0>} : tensor<?x3xi64, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @transpose_normal(%ctx: !hipsr.context,
                            %input: tensor<3x?xi64, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<3x?xi64, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x3xi64, #hipsr.mem<device>>
  %result = hipsr.transpose(%ctx)
      ins(%input : tensor<3x?xi64, #hipsr.mem<device>>)
      outs(%init : tensor<?x3xi64, #hipsr.mem<device>>)
      {perm = array<i64: 1, 0>} : tensor<?x3xi64, #hipsr.mem<device>>
  return
}
