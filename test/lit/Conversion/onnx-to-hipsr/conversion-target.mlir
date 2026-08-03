// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-onnx-to-hipsr %s | FileCheck %s

// Helper operations remain legal at any nesting depth inside hipsr.compute.
// CHECK-LABEL: func.func @recursive_legality(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<2x3xf16>,
// CHECK-SAME: %[[INIT:.+]]: tensor<6xf16>) -> tensor<6xf16> {
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3xf16>) outs(%[[INIT]] : tensor<6xf16>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[BODY_INPUT:.+]]: tensor<2x3xf16>, %{{.+}}: tensor<6xf16>):
// CHECK-NEXT: %[[NESTED:.+]] = scf.execute_region -> tensor<6xf16> {
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[BODY_INPUT]] {{\[\[}}0, 1]] : tensor<2x3xf16> into tensor<6xf16>
// CHECK-NEXT: scf.yield %[[FLAT]] : tensor<6xf16>
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.compute_yield %[[NESTED]] : tensor<6xf16>
// CHECK-NEXT: } : tensor<6xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<6xf16>
// CHECK-NEXT: }
func.func @recursive_legality(
    %ctx: !hipsr.context, %input: tensor<2x3xf16>,
    %init: tensor<6xf16>) -> tensor<6xf16> {
  %result = hipsr.compute(%ctx) ins(%input : tensor<2x3xf16>)
                                  outs(%init : tensor<6xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %body_input: tensor<2x3xf16>,
       %body_init: tensor<6xf16>):
    %nested = scf.execute_region -> tensor<6xf16> {
      %flat = tensor.collapse_shape %body_input [[0, 1]]
          : tensor<2x3xf16> into tensor<6xf16>
      scf.yield %flat : tensor<6xf16>
    }
    hipsr.compute_yield %nested : tensor<6xf16>
  } : tensor<6xf16>
  return %result : tensor<6xf16>
}
