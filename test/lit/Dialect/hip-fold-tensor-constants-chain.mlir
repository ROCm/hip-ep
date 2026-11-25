// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fold-tensor-constants %s | FileCheck %s
// RUN: hip-mlir-opt --hip-fold-tensor-constants='max-fold-elements=1' %s | FileCheck %s --check-prefix=CAPPED
// RUN: hip-mlir-opt --hip-fold-tensor-constants='max-fold-total-elements=4' %s | FileCheck %s --check-prefix=BUDGET

// What this file tests
// ====================
// The attention-mask constant chain of a real audio model
// (facebook/data2vec-audio-base-960h through the torch_hip plugin): an iota
// compared against zero, inverted, and broadcast to the attention shape,
// feeding every layer's hip.where condition. The whole chain is evaluated at
// compile time: range, less, not and expand disappear together with the
// expand_shape carriers between them, and the wheres read one inline
// splat-true carrier.

// CHECK-LABEL: func.func @audio_mask_chain
// CAPPED-LABEL: func.func @audio_mask_chain
// CAPPED: hip.range
// CAPPED: hip.expand
// BUDGET-LABEL: func.func @audio_mask_chain
// BUDGET: hip.range
// CHECK-NOT: hip.range
// CHECK-NOT: hip.less
// CHECK-NOT: hip.not
// CHECK-NOT: hip.expand
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[MASK:.*]] = hip.constant {value = dense<true> : tensor<1x1x169x169xi1>} : tensor<1x1x169x169xi1>
// CHECK: hip.where({{.*}}) ins(%[[MASK]],
// CHECK: hip.where({{.*}}) ins(%[[MASK]],
// CHECK-NOT: hip.range
func.func @audio_mask_chain(%ctx: !hip.context,
                            %x: tensor<1x1x169x169xf32>,
                            %y: tensor<1x1x169x169xf32>)
    -> (tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) {
  %start = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %limit = hip.constant {value = dense<169> : tensor<i64>} : tensor<i64>
  %delta = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %iota_init = tensor.empty() : tensor<169xi64>
  %iota = hip.range(%ctx) ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
      outs(%iota_init : tensor<169xi64>) : tensor<169xi64>
  %e0 = tensor.expand_shape %iota [[0, 1]] output_shape [1, 169] : tensor<169xi64> into tensor<1x169xi64>
  %e1 = tensor.expand_shape %e0 [[0, 1], [2]] output_shape [1, 1, 169] : tensor<1x169xi64> into tensor<1x1x169xi64>
  %e2 = tensor.expand_shape %e1 [[0], [1], [2, 3]] output_shape [1, 1, 169, 1] : tensor<1x1x169xi64> into tensor<1x1x169x1xi64>
  %zero = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %less_init = tensor.empty() : tensor<1x1x169x1xi1>
  %less = hip.less(%ctx) ins(%e2, %zero : tensor<1x1x169x1xi64>, tensor<i64>)
      outs(%less_init : tensor<1x1x169x1xi1>) : tensor<1x1x169x1xi1>
  %not_init = tensor.empty() : tensor<1x1x169x1xi1>
  %not = hip.not(%ctx) ins(%less : tensor<1x1x169x1xi1>)
      outs(%not_init : tensor<1x1x169x1xi1>) : tensor<1x1x169x1xi1>
  %shape = hip.constant {value = dense<[1, 1, 169, 169]> : tensor<4xi64>} : tensor<4xi64>
  %mask_init = tensor.empty() : tensor<1x1x169x169xi1>
  %mask = hip.expand(%ctx) ins(%not, %shape : tensor<1x1x169x1xi1>, tensor<4xi64>)
      outs(%mask_init : tensor<1x1x169x169xi1>) : tensor<1x1x169x169xi1>
  %init0 = tensor.empty() : tensor<1x1x169x169xf32>
  %layer0 = hip.where(%ctx) ins(%mask, %x, %y :
      tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>)
      outs(%init0 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
  %init1 = tensor.empty() : tensor<1x1x169x169xf32>
  %layer1 = hip.where(%ctx) ins(%mask, %y, %x :
      tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>)
      outs(%init1 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
  return %layer0, %layer1 : tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>
}

// The evaluation budget is module-wide and charges failed attempts: placed
// immediately after the audio chain, whose attempts all FAIL under
// max-fold-total-elements=4 (the range attempt alone charges 169), so if
// failures were not charged this 4-element fold would still fit the budget.
// The default run folds it.
// CHECK-LABEL: func.func @budget_cross_function
// CHECK-NOT: hip.not
// BUDGET-LABEL: func.func @budget_cross_function
// BUDGET: hip.not
func.func @budget_cross_function(%ctx: !hip.context, %x: tensor<4xf32>,
                                 %y: tensor<4xf32>) -> tensor<4xf32> {
  %input = hip.constant {value = dense<false> : tensor<4xi1>} : tensor<4xi1>
  %not_init = tensor.empty() : tensor<4xi1>
  %condition = hip.not(%ctx) ins(%input : tensor<4xi1>)
      outs(%not_init : tensor<4xi1>) : tensor<4xi1>
  %where_init = tensor.empty() : tensor<4xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%where_init : tensor<4xf32>) : tensor<4xf32>
  return %result : tensor<4xf32>
}

// A non-splat broadcast is materialized element-exactly: expand [F,T] from
// [1,2] to [2,2] duplicates the row.
// CHECK-LABEL: func.func @fold_expand_non_splat
// CHECK-NOT: hip.expand
// CHECK: hip.constant {value = dense<{{\[\[}}false, true], [false, true]]> : tensor<2x2xi1>} : tensor<2x2xi1>
func.func @fold_expand_non_splat(%ctx: !hip.context, %x: tensor<2x2xf32>,
                                 %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %input = hip.constant {value = dense<[[false, true]]> : tensor<1x2xi1>} : tensor<1x2xi1>
  %shape = hip.constant {value = dense<[2, 2]> : tensor<2xi64>} : tensor<2xi64>
  %expand_init = tensor.empty() : tensor<2x2xi1>
  %mask = hip.expand(%ctx) ins(%input, %shape : tensor<1x2xi1>, tensor<2xi64>)
      outs(%expand_init : tensor<2x2xi1>) : tensor<2x2xi1>
  %where_init = tensor.empty() : tensor<2x2xf32>
  %result = hip.where(%ctx) ins(%mask, %x, %y :
      tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%where_init : tensor<2x2xf32>) : tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

// A shape operand that disagrees with the static result type bails.
// CHECK-LABEL: func.func @keep_expand_shape_mismatch
// CHECK: hip.expand
func.func @keep_expand_shape_mismatch(%ctx: !hip.context, %x: tensor<3x2xf32>,
                                      %y: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = hip.constant {value = dense<true> : tensor<1x2xi1>} : tensor<1x2xi1>
  %shape = hip.constant {value = dense<[2, 2]> : tensor<2xi64>} : tensor<2xi64>
  %expand_init = tensor.empty() : tensor<3x2xi1>
  %mask = hip.expand(%ctx) ins(%input, %shape : tensor<1x2xi1>, tensor<2xi64>)
      outs(%expand_init : tensor<3x2xi1>) : tensor<3x2xi1>
  %where_init = tensor.empty() : tensor<3x2xf32>
  %result = hip.where(%ctx) ins(%mask, %x, %y :
      tensor<3x2xi1>, tensor<3x2xf32>, tensor<3x2xf32>)
      outs(%where_init : tensor<3x2xf32>) : tensor<3x2xf32>
  return %result : tensor<3x2xf32>
}

// Under max-fold-elements=1 the two-element candidate is rejected by the
// per-payload cap before anything is evaluated, so nothing folds. The
// default run folds the chain.
// CHECK-LABEL: func.func @capped_see_through
// CHECK-NOT: hip.not
// CAPPED-LABEL: func.func @capped_see_through
// CAPPED: tensor.expand_shape
// CAPPED: hip.not
func.func @capped_see_through(%ctx: !hip.context, %x: tensor<1x2xf32>,
                              %y: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %input = hip.constant {value = dense<false> : tensor<2xi1>} : tensor<2xi1>
  %expanded = tensor.expand_shape %input [[0, 1]] output_shape [1, 2] : tensor<2xi1> into tensor<1x2xi1>
  %not_init = tensor.empty() : tensor<1x2xi1>
  %condition = hip.not(%ctx) ins(%expanded : tensor<1x2xi1>)
      outs(%not_init : tensor<1x2xi1>) : tensor<1x2xi1>
  %where_init = tensor.empty() : tensor<1x2xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<1x2xi1>, tensor<1x2xf32>, tensor<1x2xf32>)
      outs(%where_init : tensor<1x2xf32>) : tensor<1x2xf32>
  return %result : tensor<1x2xf32>
}

// Specifically the expand_shape see-through cap: the reshape source (2
// elements) exceeds max-fold-elements=1 while the expand RESULT is empty
// (0 elements, passes the per-payload cap), so the only guard that can stop
// this fold under CAPPED is the reshape materialization check. The default
// run folds the expand to an empty carrier.
// CHECK-LABEL: func.func @capped_reshape_guard
// CHECK-NOT: hip.expand
// CHECK: hip.constant {value = dense<> : tensor<0x2xi1>} : tensor<0x2xi1>
// CAPPED-LABEL: func.func @capped_reshape_guard
// CAPPED: tensor.expand_shape
// CAPPED: hip.expand
func.func @capped_reshape_guard(%ctx: !hip.context, %x: tensor<0x2xf32>,
                                %y: tensor<0x2xf32>) -> tensor<0x2xf32> {
  %input = hip.constant {value = dense<false> : tensor<2xi1>} : tensor<2xi1>
  %reshaped = tensor.expand_shape %input [[0, 1]] output_shape [1, 2] : tensor<2xi1> into tensor<1x2xi1>
  %shape = hip.constant {value = dense<[0, 2]> : tensor<2xi64>} : tensor<2xi64>
  %expand_init = tensor.empty() : tensor<0x2xi1>
  %mask = hip.expand(%ctx) ins(%reshaped, %shape : tensor<1x2xi1>, tensor<2xi64>)
      outs(%expand_init : tensor<0x2xi1>) : tensor<0x2xi1>
  %where_init = tensor.empty() : tensor<0x2xf32>
  %result = hip.where(%ctx) ins(%mask, %x, %y :
      tensor<0x2xi1>, tensor<0x2xf32>, tensor<0x2xf32>)
      outs(%where_init : tensor<0x2xf32>) : tensor<0x2xf32>
  return %result : tensor<0x2xf32>
}
