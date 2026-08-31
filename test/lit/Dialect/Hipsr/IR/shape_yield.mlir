// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Multi-result placeholders yield one shape per result. A scalar still has one
// shape value, produced from an empty extent list.
// CHECK-LABEL: func.func @multiple_results_and_scalar(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context) -> (tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INITS:.+]]:2 = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context):
// CHECK-NEXT: %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[C3:.+]] = arith.constant 3 : index
// CHECK-NEXT: %[[VALUES_SHAPE:.+]] = tensor.from_elements %[[C2]], %[[C3]] : tensor<2xindex>
// CHECK-NEXT: %[[SCALAR_SHAPE:.+]] = arith.constant dense<> : tensor<0xindex>
// CHECK-NEXT: hipsr.shape_yield %[[VALUES_SHAPE]], %[[SCALAR_SHAPE]]
// CHECK-SAME: : tensor<2xindex>, tensor<0xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULTS:.+]]:2 = hipsr.compute(%[[CTX]]) ins() outs(%[[INITS]]#0, %[[INITS]]#1 : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[VALUES_DEST:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[SCALAR_DEST:.+]]: tensor<f16, #hipsr.mem<device>>):
// CHECK-NEXT: hipsr.compute_yield %[[VALUES_DEST]], %[[SCALAR_DEST]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>
// CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @multiple_results_and_scalar(%ctx: !hipsr.context)
    -> (tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>) {
  %inits:2 = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>> shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %values_shape = tensor.from_elements %c2, %c3 : tensor<2xindex>
    %scalar_shape = arith.constant dense<> : tensor<0xindex>
    hipsr.shape_yield %values_shape, %scalar_shape
        : tensor<2xindex>, tensor<0xindex>
  }
  %results:2 = hipsr.compute(%ctx) ins()
      outs(%inits#0, %inits#1 : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %values_dest: tensor<2x3xf16, #hipsr.mem<device>>,
       %scalar_dest: tensor<f16, #hipsr.mem<device>>):
    hipsr.compute_yield %values_dest, %scalar_dest
        : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>
  } : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>
  return %results#0, %results#1 : tensor<2x3xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>
}

// -----

func.func @non_shape_operand(%ctx: !hipsr.context) -> tensor<f16, #hipsr.mem<device>> {
  %init = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<f16, #hipsr.mem<device>>
      shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    %extent = arith.constant 1 : index
    // expected-error @+1 {{operand #0 must be variadic of 1D tensor of index values, but got 'index'}}
    hipsr.shape_yield %extent : index
  }
  %result = hipsr.compute(%ctx) ins() outs(%init : tensor<f16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<f16, #hipsr.mem<device>>):
    hipsr.compute_yield %dest : tensor<f16, #hipsr.mem<device>>
  } : tensor<f16, #hipsr.mem<device>>
  return %result : tensor<f16, #hipsr.mem<device>>
}

// -----

// An empty block gets an implicit zero-operand yield, which cannot describe
// the placeholder's result.
func.func @implicit_empty_yield(%ctx: !hipsr.context) -> tensor<f16, #hipsr.mem<device>> {
  // expected-error @+1 {{must yield one extent tensor per enclosing placeholder result; expected 1, got 0}}
  %init = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<f16, #hipsr.mem<device>> shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
  }
  %result = hipsr.compute(%ctx) ins() outs(%init : tensor<f16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<f16, #hipsr.mem<device>>):
    hipsr.compute_yield %dest : tensor<f16, #hipsr.mem<device>>
  } : tensor<f16, #hipsr.mem<device>>
  return %result : tensor<f16, #hipsr.mem<device>>
}

// -----

// A yielded shape must describe its result, so a length that disagrees with
// the result rank is rejected here rather than at a later extent read.
func.func @extent_count_mismatch(%ctx: !hipsr.context)
    -> tensor<2x3xf16, #hipsr.mem<device>> {
  %init = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<2x3xf16, #hipsr.mem<device>> shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    %c2 = arith.constant 2 : index
    %shape = tensor.from_elements %c2 : tensor<1xindex>
    // expected-error @+1 {{shape #0 holds 1 extents but result #0 has rank 2}}
    hipsr.shape_yield %shape : tensor<1xindex>
  }
  %result = hipsr.compute(%ctx) ins() outs(%init : tensor<2x3xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<2x3xf16, #hipsr.mem<device>>):
    hipsr.compute_yield %dest : tensor<2x3xf16, #hipsr.mem<device>>
  } : tensor<2x3xf16, #hipsr.mem<device>>
  return %result : tensor<2x3xf16, #hipsr.mem<device>>
}

// -----

// A dynamic length states nothing, so the verifier has nothing to compare.
// CHECK-LABEL: func.func @dynamic_extent_count(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context) -> tensor<2x3xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<2x3xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context):
// CHECK-NEXT: %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT: %[[C3:.+]] = arith.constant 3 : index
// CHECK-NEXT: %[[STATIC_SHAPE:.+]] = tensor.from_elements %[[C2]], %[[C3]] : tensor<2xindex>
// CHECK-NEXT: %[[SHAPE:.+]] = tensor.cast %[[STATIC_SHAPE]] : tensor<2xindex> to tensor<?xindex>
// CHECK-NEXT: hipsr.shape_yield %[[SHAPE]] : tensor<?xindex>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.+]] = hipsr.compute(%[[CTX]]) ins() outs(%[[INIT]] : tensor<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%{{.+}}: !hipsr.context, %[[DEST:.+]]: tensor<2x3xf16, #hipsr.mem<device>>):
// CHECK-NEXT: hipsr.compute_yield %[[DEST]] : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: return %[[RESULT]] : tensor<2x3xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @dynamic_extent_count(%ctx: !hipsr.context)
    -> tensor<2x3xf16, #hipsr.mem<device>> {
  %init = hipsr.placeholder(%ctx)
      {placeholder_type = #hipsr.placeholder_type<barrier>}
      : tensor<2x3xf16, #hipsr.mem<device>> shape_region {
  ^bb0(%shape_ctx: !hipsr.context):
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %static_shape = tensor.from_elements %c2, %c3 : tensor<2xindex>
    %shape = tensor.cast %static_shape : tensor<2xindex> to tensor<?xindex>
    hipsr.shape_yield %shape : tensor<?xindex>
  }
  %result = hipsr.compute(%ctx) ins() outs(%init : tensor<2x3xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %dest: tensor<2x3xf16, #hipsr.mem<device>>):
    hipsr.compute_yield %dest : tensor<2x3xf16, #hipsr.mem<device>>
  } : tensor<2x3xf16, #hipsr.mem<device>>
  return %result : tensor<2x3xf16, #hipsr.mem<device>>
}

// -----

// ShapeYield terminates placeholder shape regions only.
func.func @wrong_parent() {
  %shape = arith.constant dense<> : tensor<0xindex>
  // expected-error @+1 {{expects parent op 'hipsr.placeholder'}}
  hipsr.shape_yield %shape : tensor<0xindex>
}
