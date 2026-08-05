// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// Cast preserves its input shape.
// CHECK-LABEL: func.func @cast_normal(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[INPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @cast_normal(%ctx: !hipsr.context, %input: tensor<?x8xf32>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %result = hipsr.cast(%ctx)
      ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return
}
