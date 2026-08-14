// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// Cast preserves its input shape.
// CHECK-LABEL: func.func @cast_normal(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[INPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @cast_normal(%ctx: !hipsr.context, %input: tensor<?x8xf32, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %result = hipsr.cast(%ctx)
      ins(%input : tensor<?x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
  return
}
