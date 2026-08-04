// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Multi-result placeholders yield one shape per result. A scalar still has one
// shape value, produced from an empty extent list.
// CHECK-LABEL: func.func @multiple_results_and_scalar(
// CHECK: %[[INITS:.+]]:2 = hipsr.placeholder
// CHECK-SAME: shape_region {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context):
// CHECK-NEXT: %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[C3:.+]] = arith.constant 3 : index
// CHECK-NEXT: %[[VALUES_SHAPE:.+]] = shape.from_extents %[[C2]], %[[C3]]
// CHECK-NEXT: %[[SCALAR_SHAPE:.+]] = shape.from_extents
// CHECK-NEXT: hipsr.shape_yield2 %[[VALUES_SHAPE]], %[[SCALAR_SHAPE]]
// CHECK-SAME: : !shape.shape, !shape.shape
// CHECK-NEXT: }
func.func @multiple_results_and_scalar(%ctx: !hipsr.context)
    -> (tensor<2x3xf16>, tensor<f16>) {
  %inits:2 = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<2x3xf16>, tensor<f16> shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %values_shape = "shape.from_extents"(%c2, %c3)
        : (index, index) -> !shape.shape
    %scalar_shape = "shape.from_extents"() : () -> !shape.shape
    hipsr.shape_yield2 %values_shape, %scalar_shape
        : !shape.shape, !shape.shape
  }
  %results:2 = hipsr.compute(%ctx) ins()
      outs(%inits#0, %inits#1 : tensor<2x3xf16>, tensor<f16>) {
  ^bb0(%body_ctx: !hipsr.context, %values_dest: tensor<2x3xf16>,
       %scalar_dest: tensor<f16>):
    hipsr.compute_yield %values_dest, %scalar_dest
        : tensor<2x3xf16>, tensor<f16>
  } : tensor<2x3xf16>, tensor<f16>
  return %results#0, %results#1 : tensor<2x3xf16>, tensor<f16>
}

// -----

// Shape-yield operands must use the Shape dialect's value shape type.
func.func @non_shape_operand(%ctx: !hipsr.context) -> tensor<f16> {
  %init = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<f16>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    %extent = arith.constant 1 : index
    // expected-error @+1 {{operand #0 must be variadic of , but got 'index'}}
    hipsr.shape_yield2 %extent : index
  }
  %result = hipsr.compute(%ctx) ins() outs(%init : tensor<f16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<f16>):
    hipsr.compute_yield %dest : tensor<f16>
  } : tensor<f16>
  return %result : tensor<f16>
}

// -----

// An empty block gets an implicit zero-operand yield, which cannot describe
// the placeholder's result.
func.func @implicit_empty_yield(%ctx: !hipsr.context) -> tensor<f16> {
  // expected-error @+1 {{must yield one !shape.shape per enclosing placeholder result; expected 1, got 0}}
  %init = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<f16> shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
  }
  %result = hipsr.compute(%ctx) ins() outs(%init : tensor<f16>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<f16>):
    hipsr.compute_yield %dest : tensor<f16>
  } : tensor<f16>
  return %result : tensor<f16>
}

// -----

// ShapeYield2 terminates placeholder shape regions only.
func.func @wrong_parent() {
  %shape = "shape.from_extents"() : () -> !shape.shape
  // expected-error @+1 {{expects parent op 'hipsr.placeholder'}}
  hipsr.shape_yield2 %shape : !shape.shape
}
