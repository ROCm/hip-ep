// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-onnx-to-hipsr %s | FileCheck %s

// Scalar constants are legal at top level, and other helper operations remain
// legal at any nesting depth inside hipsr.compute.
// CHECK-LABEL: func.func @recursive_legality(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[INPUT:.+]]: tensor<2x3xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[INIT:.+]]: tensor<6xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
// CHECK-NEXT: arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<6xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[BODY_INPUT:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %{{.+}}: tensor<6xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[NESTED:.+]] = scf.execute_region -> tensor<6xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[FLAT:.+]] = tensor.collapse_shape %[[BODY_INPUT]] {{\[\[}}0, 1]] : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: scf.yield %[[FLAT]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
// CHECK-NEXT: hipsr.compute_yield %[[NESTED]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: return %[[RESULT]] : tensor<6xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @recursive_legality(
    %ctx: !hipsr.context, %input: tensor<2x3xf16, #hipsr.mem<device>>,
    %init: tensor<6xf16, #hipsr.mem<device>>) -> tensor<6xf16, #hipsr.mem<device>> {
  %scalar = arith.constant 1.0 : f32
  %result = hipsr.compute(%ctx) ins(%input : tensor<2x3xf16, #hipsr.mem<device>>)
                                  outs(%init : tensor<6xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %body_input: tensor<2x3xf16, #hipsr.mem<device>>,
       %body_init: tensor<6xf16, #hipsr.mem<device>>):
    %nested = scf.execute_region -> tensor<6xf16, #hipsr.mem<device>> {
      %flat = tensor.collapse_shape %body_input [[0, 1]]
          : tensor<2x3xf16, #hipsr.mem<device>> into tensor<6xf16, #hipsr.mem<device>>
      scf.yield %flat : tensor<6xf16, #hipsr.mem<device>>
    }
    hipsr.compute_yield %nested : tensor<6xf16, #hipsr.mem<device>>
  } : tensor<6xf16, #hipsr.mem<device>>
  return %result : tensor<6xf16, #hipsr.mem<device>>
}
