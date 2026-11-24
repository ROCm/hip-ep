// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-fold-tensor-constants %s | FileCheck %s

// What this file tests
// ====================
// hip-fold-tensor-constants evaluates element-wise boolean HIP ops whose data
// inputs are inline hip.constant carriers of exactly the static result type,
// replaces every audited read-only use with a new inline hip.constant, and
// erases consumed carriers only once they are dead. Negative cases pin the
// refusals: terminator uses, DPS init uses, non-constant inputs, broadcast
// forms, and the module-wide duplicate-serialization-order gate (kept in its
// own split chunk so it does not disable folding for the other cases).

// Fold a splat boolean NOT used only as a where condition. The consumed input
// carrier is retired with the folded op; the replacement has no
// serialization_order.
// CHECK-LABEL: func.func @fold_splat_not
// CHECK-NOT: hip.not
// CHECK-NOT: dense<false>
// CHECK: %[[COND:.*]] = hip.constant {value = dense<true> : tensor<4xi1>} : tensor<4xi1>
// CHECK: hip.where({{.*}}) ins(%[[COND]],
// CHECK-NOT: hip.not
func.func @fold_splat_not(%ctx: !hip.context, %x: tensor<4xf32>,
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

// Non-splat payloads are evaluated element-wise.
// CHECK-LABEL: func.func @fold_dense_not
// CHECK-NOT: hip.not
// CHECK: %[[COND:.*]] = hip.constant {value = dense<[true, false]> : tensor<2xi1>} : tensor<2xi1>
// CHECK: hip.where({{.*}}) ins(%[[COND]],
// CHECK-NOT: hip.not
func.func @fold_dense_not(%ctx: !hip.context, %x: tensor<2xf32>,
                          %y: tensor<2xf32>) -> tensor<2xf32> {
  %input = hip.constant {value = dense<[false, true]> : tensor<2xi1>} : tensor<2xi1>
  %not_init = tensor.empty() : tensor<2xi1>
  %condition = hip.not(%ctx) ins(%input : tensor<2xi1>)
      outs(%not_init : tensor<2xi1>) : tensor<2xi1>
  %where_init = tensor.empty() : tensor<2xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<2xi1>, tensor<2xf32>, tensor<2xf32>)
      outs(%where_init : tensor<2xf32>) : tensor<2xf32>
  return %result : tensor<2xf32>
}

// A producer-before-consumer chain folds in one sweep: the and result becomes
// a constant that the or then consumes. and([T,F],[T,T]) = [T,F];
// or([T,F],[F,F]) = [T,F].
// CHECK-LABEL: func.func @fold_and_or_chain
// CHECK-NOT: hip.and
// CHECK-NOT: hip.or
// CHECK: %[[COND:.*]] = hip.constant {value = dense<[true, false]> : tensor<2xi1>} : tensor<2xi1>
// CHECK: hip.where({{.*}}) ins(%[[COND]],
// CHECK-NOT: hip.and
// CHECK-NOT: hip.or
func.func @fold_and_or_chain(%ctx: !hip.context, %x: tensor<2xf32>,
                             %y: tensor<2xf32>) -> tensor<2xf32> {
  %a = hip.constant {value = dense<[true, false]> : tensor<2xi1>} : tensor<2xi1>
  %b = hip.constant {value = dense<true> : tensor<2xi1>} : tensor<2xi1>
  %c = hip.constant {value = dense<false> : tensor<2xi1>} : tensor<2xi1>
  %and_init = tensor.empty() : tensor<2xi1>
  %and = hip.and(%ctx) ins(%a, %b : tensor<2xi1>, tensor<2xi1>)
      outs(%and_init : tensor<2xi1>) : tensor<2xi1>
  %or_init = tensor.empty() : tensor<2xi1>
  %or = hip.or(%ctx) ins(%and, %c : tensor<2xi1>, tensor<2xi1>)
      outs(%or_init : tensor<2xi1>) : tensor<2xi1>
  %where_init = tensor.empty() : tensor<2xf32>
  %result = hip.where(%ctx) ins(%or, %x, %y :
      tensor<2xi1>, tensor<2xf32>, tensor<2xf32>)
      outs(%where_init : tensor<2xf32>) : tensor<2xf32>
  return %result : tensor<2xf32>
}

// A consumed carrier with another user stays alive; the fold still happens.
// CHECK-LABEL: func.func @fold_keeps_shared_carrier
// CHECK: %[[INPUT:.*]] = hip.constant {value = dense<false> : tensor<4xi1>}
// CHECK-NOT: hip.not
// CHECK: hip.where({{.*}}) ins(%{{[0-9a-z]+}},
// CHECK: return %[[INPUT]],
func.func @fold_keeps_shared_carrier(%ctx: !hip.context, %x: tensor<4xf32>,
                                     %y: tensor<4xf32>)
    -> (tensor<4xi1>, tensor<4xf32>) {
  %input = hip.constant {value = dense<false> : tensor<4xi1>} : tensor<4xi1>
  %not_init = tensor.empty() : tensor<4xi1>
  %condition = hip.not(%ctx) ins(%input : tensor<4xi1>)
      outs(%not_init : tensor<4xi1>) : tensor<4xi1>
  %where_init = tensor.empty() : tensor<4xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%where_init : tensor<4xf32>) : tensor<4xf32>
  return %input, %result : tensor<4xi1>, tensor<4xf32>
}

// A terminator use is not a foldable read: graph outputs stay
// runtime-allocated, so the candidate is left unchanged.
// CHECK-LABEL: func.func @keep_return_use
// CHECK: %[[NOT:.*]] = hip.not
// CHECK: return %[[NOT]],
func.func @keep_return_use(%ctx: !hip.context, %x: tensor<4xf32>,
                           %y: tensor<4xf32>)
    -> (tensor<4xi1>, tensor<4xf32>) {
  %input = hip.constant {value = dense<false> : tensor<4xi1>} : tensor<4xi1>
  %not_init = tensor.empty() : tensor<4xi1>
  %condition = hip.not(%ctx) ins(%input : tensor<4xi1>)
      outs(%not_init : tensor<4xi1>) : tensor<4xi1>
  %where_init = tensor.empty() : tensor<4xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%where_init : tensor<4xf32>) : tensor<4xf32>
  return %condition, %result : tensor<4xi1>, tensor<4xf32>
}

// A DPS init use is not a foldable read: replacing the init with a constant
// would make bufferization write into constant storage.
// CHECK-LABEL: func.func @keep_init_use
// CHECK: hip.not
// CHECK: hip.where
func.func @keep_init_use(%ctx: !hip.context, %cond: tensor<4xi1>,
                         %x: tensor<4xi1>, %y: tensor<4xi1>) -> tensor<4xi1> {
  %input = hip.constant {value = dense<false> : tensor<4xi1>} : tensor<4xi1>
  %not_init = tensor.empty() : tensor<4xi1>
  %not = hip.not(%ctx) ins(%input : tensor<4xi1>)
      outs(%not_init : tensor<4xi1>) : tensor<4xi1>
  %result = hip.where(%ctx) ins(%cond, %x, %y :
      tensor<4xi1>, tensor<4xi1>, tensor<4xi1>)
      outs(%not : tensor<4xi1>) : tensor<4xi1>
  return %result : tensor<4xi1>
}

// A non-constant input blocks evaluation.
// CHECK-LABEL: func.func @keep_non_constant_input
// CHECK: hip.not
func.func @keep_non_constant_input(%ctx: !hip.context, %input: tensor<4xi1>,
                                   %x: tensor<4xf32>, %y: tensor<4xf32>)
    -> tensor<4xf32> {
  %not_init = tensor.empty() : tensor<4xi1>
  %condition = hip.not(%ctx) ins(%input : tensor<4xi1>)
      outs(%not_init : tensor<4xi1>) : tensor<4xi1>
  %where_init = tensor.empty() : tensor<4xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%where_init : tensor<4xf32>) : tensor<4xf32>
  return %result : tensor<4xf32>
}

// A tensor.expand_shape between the carrier and the consumer is seen through:
// the payload is reshaped, the not folds, and the now-dead expand_shape and
// carrier are retired with it.
// CHECK-LABEL: func.func @fold_through_expand_shape
// CHECK-NOT: hip.not
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[COND:.*]] = hip.constant {value = dense<true> : tensor<1x4xi1>} : tensor<1x4xi1>
// CHECK: hip.where({{.*}}) ins(%[[COND]],
// CHECK-NOT: hip.not
func.func @fold_through_expand_shape(%ctx: !hip.context, %x: tensor<1x4xf32>,
                                     %y: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %input = hip.constant {value = dense<false> : tensor<4xi1>} : tensor<4xi1>
  %expanded = tensor.expand_shape %input [[0, 1]] output_shape [1, 4] : tensor<4xi1> into tensor<1x4xi1>
  %not_init = tensor.empty() : tensor<1x4xi1>
  %condition = hip.not(%ctx) ins(%expanded : tensor<1x4xi1>)
      outs(%not_init : tensor<1x4xi1>) : tensor<1x4xi1>
  %where_init = tensor.empty() : tensor<1x4xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<1x4xi1>, tensor<1x4xf32>, tensor<1x4xf32>)
      outs(%where_init : tensor<1x4xf32>) : tensor<1x4xf32>
  return %result : tensor<1x4xf32>
}

// An i64 iota chain folds: range(0,4,1) then less(iota, 2) with a rank-0
// scalar rhs. The range folds first (its only use is a DPS input), then the
// less reads the folded carrier.
// CHECK-LABEL: func.func @fold_range_less
// CHECK-NOT: hip.range
// CHECK-NOT: hip.less
// CHECK: %[[COND:.*]] = hip.constant {value = dense<[true, true, false, false]> : tensor<4xi1>} : tensor<4xi1>
// CHECK: hip.where({{.*}}) ins(%[[COND]],
func.func @fold_range_less(%ctx: !hip.context, %x: tensor<4xf32>,
                           %y: tensor<4xf32>) -> tensor<4xf32> {
  %start = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %limit = hip.constant {value = dense<4> : tensor<i64>} : tensor<i64>
  %delta = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %iota_init = tensor.empty() : tensor<4xi64>
  %iota = hip.range(%ctx) ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
      outs(%iota_init : tensor<4xi64>) : tensor<4xi64>
  %two = hip.constant {value = dense<2> : tensor<i64>} : tensor<i64>
  %less_init = tensor.empty() : tensor<4xi1>
  %condition = hip.less(%ctx) ins(%iota, %two : tensor<4xi64>, tensor<i64>)
      outs(%less_init : tensor<4xi1>) : tensor<4xi1>
  %where_init = tensor.empty() : tensor<4xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%where_init : tensor<4xf32>) : tensor<4xf32>
  return %result : tensor<4xf32>
}

// Negative delta: range(5,0,-2) = [5,3,1]; less(iota, 4) = [F,T,T].
// CHECK-LABEL: func.func @fold_range_negative_delta
// CHECK-NOT: hip.range
// CHECK: hip.constant {value = dense<[false, true, true]> : tensor<3xi1>} : tensor<3xi1>
func.func @fold_range_negative_delta(%ctx: !hip.context, %x: tensor<3xf32>,
                                     %y: tensor<3xf32>) -> tensor<3xf32> {
  %start = hip.constant {value = dense<5> : tensor<i64>} : tensor<i64>
  %limit = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %delta = hip.constant {value = dense<-2> : tensor<i64>} : tensor<i64>
  %iota_init = tensor.empty() : tensor<3xi64>
  %iota = hip.range(%ctx) ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
      outs(%iota_init : tensor<3xi64>) : tensor<3xi64>
  %four = hip.constant {value = dense<4> : tensor<i64>} : tensor<i64>
  %less_init = tensor.empty() : tensor<3xi1>
  %condition = hip.less(%ctx) ins(%iota, %four : tensor<3xi64>, tensor<i64>)
      outs(%less_init : tensor<3xi1>) : tensor<3xi1>
  %where_init = tensor.empty() : tensor<3xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<3xi1>, tensor<3xf32>, tensor<3xf32>)
      outs(%where_init : tensor<3xf32>) : tensor<3xf32>
  return %result : tensor<3xf32>
}

// delta = 0 is undefined Range input: the evaluator bails and the whole
// chain stays.
// CHECK-LABEL: func.func @keep_range_zero_delta
// CHECK: hip.range
// CHECK: hip.less
func.func @keep_range_zero_delta(%ctx: !hip.context, %x: tensor<4xf32>,
                                 %y: tensor<4xf32>) -> tensor<4xf32> {
  %start = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %limit = hip.constant {value = dense<4> : tensor<i64>} : tensor<i64>
  %delta = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %iota_init = tensor.empty() : tensor<4xi64>
  %iota = hip.range(%ctx) ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
      outs(%iota_init : tensor<4xi64>) : tensor<4xi64>
  %two = hip.constant {value = dense<2> : tensor<i64>} : tensor<i64>
  %less_init = tensor.empty() : tensor<4xi1>
  %condition = hip.less(%ctx) ins(%iota, %two : tensor<4xi64>, tensor<i64>)
      outs(%less_init : tensor<4xi1>) : tensor<4xi1>
  %where_init = tensor.empty() : tensor<4xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%where_init : tensor<4xf32>) : tensor<4xf32>
  return %result : tensor<4xf32>
}

// A broadcast form is refused: the exact-type gate requires every input to
// match the result type, and dense<true> over tensor<1xi1> does not.
// CHECK-LABEL: func.func @keep_broadcast_and
// CHECK: hip.and
func.func @keep_broadcast_and(%ctx: !hip.context, %x: tensor<2xf32>,
                              %y: tensor<2xf32>) -> tensor<2xf32> {
  %a = hip.constant {value = dense<true> : tensor<1xi1>} : tensor<1xi1>
  %b = hip.constant {value = dense<[true, false]> : tensor<2xi1>} : tensor<2xi1>
  %and_init = tensor.empty() : tensor<2xi1>
  %condition = hip.and(%ctx) ins(%a, %b : tensor<1xi1>, tensor<2xi1>)
      outs(%and_init : tensor<2xi1>) : tensor<2xi1>
  %where_init = tensor.empty() : tensor<2xf32>
  %result = hip.where(%ctx) ins(%condition, %x, %y :
      tensor<2xi1>, tensor<2xf32>, tensor<2xf32>)
      outs(%where_init : tensor<2xf32>) : tensor<2xf32>
  return %result : tensor<2xf32>
}

// -----

// Duplicate explicit serialization orders make the pass do nothing for the
// whole module, so the externalizer's duplicate-order diagnostic is not
// hidden by a retired carrier. This chunk is a separate split module so the
// gate does not disable the cases above.
// CHECK-LABEL: func.func @keep_duplicate_order
// CHECK: hip.constant {serialization_order = 7
// CHECK: hip.constant {serialization_order = 7
// CHECK: hip.not
func.func @keep_duplicate_order(%ctx: !hip.context, %x: tensor<1xf32>,
                                %y: tensor<1xf32>) -> tensor<1xf32> {
  %input = hip.constant {
    serialization_order = 7 : i64, value = dense<false> : tensor<1xi1>
  } : tensor<1xi1>
  %other = hip.constant {
    serialization_order = 7 : i64, value = dense<true> : tensor<1xi1>
  } : tensor<1xi1>
  %not_init = tensor.empty() : tensor<1xi1>
  %condition = hip.not(%ctx) ins(%input : tensor<1xi1>)
      outs(%not_init : tensor<1xi1>) : tensor<1xi1>
  %and_init = tensor.empty() : tensor<1xi1>
  %cond2 = hip.and(%ctx) ins(%condition, %other : tensor<1xi1>, tensor<1xi1>)
      outs(%and_init : tensor<1xi1>) : tensor<1xi1>
  %where_init = tensor.empty() : tensor<1xf32>
  %result = hip.where(%ctx) ins(%cond2, %x, %y :
      tensor<1xi1>, tensor<1xf32>, tensor<1xf32>)
      outs(%where_init : tensor<1xf32>) : tensor<1xf32>
  return %result : tensor<1xf32>
}

// The gate is module-wide: this function would fold on its own, but shares
// the module with the duplicate orders above and must stay unchanged.
// CHECK-LABEL: func.func @keep_gated_foldable
// CHECK: hip.not
func.func @keep_gated_foldable(%ctx: !hip.context, %x: tensor<4xf32>,
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
