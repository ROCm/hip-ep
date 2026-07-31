// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// Cast yields its normal placeholder input shape unchanged. The compute op
// remains regionless.
// CHECK-LABEL: func.func @cast_tensor(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<?x8xf32>) -> tensor<?x8xf16> {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16> shape_region {
// CHECK-NEXT: ^bb0(%[[INPUT_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT: hipsr.shape_yield %[[INPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<?x8xf16>
// CHECK-NEXT: }
func.func @cast_tensor(
    %ctx: !hipsr.context, %input: tensor<?x8xf32>) -> tensor<?x8xf16> {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf32>)
      {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}
