// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// An existing placeholder shape region is left unchanged.
// CHECK-LABEL: func.func @already_populated(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<4x8xf32>) {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4x8xf16> shape_region {
// CHECK-NEXT: ^bb0(%{{.+}}: !shape.shape):
// CHECK-NEXT: %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT: %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT: %[[OUTPUT_SHAPE:.+]] = shape.from_extents %[[C4]], %[[C8]] : index, index
// CHECK-NEXT: hipsr.shape_yield %[[OUTPUT_SHAPE]] : !shape.shape
// CHECK-NEXT: }
// CHECK-NEXT: %{{.+}} = hipsr.cast(%[[CTX]]) ins(%[[INPUT]] : tensor<4x8xf32>) outs(%[[INIT]] : tensor<4x8xf16>) : tensor<4x8xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @already_populated(
    %ctx: !hipsr.context, %input: tensor<4x8xf32>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4x8xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<4x8xf16> shape_region {
  ^bb0(%input_shape: !shape.shape):
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %output_shape = shape.from_extents %c4, %c8 : index, index
    hipsr.shape_yield %output_shape : !shape.shape
  }
  %result = hipsr.cast(%ctx)
      ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return
}
