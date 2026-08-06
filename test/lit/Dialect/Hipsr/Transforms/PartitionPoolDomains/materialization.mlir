// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file -hipsr-partition-pool-domains %s | FileCheck %s

// Normal placeholders stay in the current domain, while barriers advance to
// the next domain. Shape edges at both boundaries use exported data values.
// CHECK-LABEL: func.func @mixed_chain(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32>) -> tensor<4xf16> {
// CHECK-NEXT: %[[DOMAIN0:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[INIT0:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[DATA0:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32>) outs(%[[INIT0]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[DATA0]] : tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]] : !hipsr.context, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[INPUT1:.*]]: tensor<4xf16>):
// CHECK-NEXT: %[[INIT1:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[INPUT1]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32>
// CHECK-NEXT: %[[DATA1:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[INPUT1]] : tensor<4xf16>) outs(%[[INIT1]] : tensor<4xf32>) : tensor<4xf32>
// CHECK-NEXT: %[[INIT2:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[INIT1]] : tensor<4xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[DATA2:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[DATA1]] : tensor<4xf32>) outs(%[[INIT2]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[DATA2]] : tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16> {domain_id = 1 : i64}
// CHECK-NEXT: %[[DOMAIN2:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN1]] : !hipsr.context, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[CTX2:.*]]: !hipsr.context, %[[INPUT2:.*]]: tensor<4xf16>):
// CHECK-NEXT: %[[INIT3:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[INPUT2]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32>
// CHECK-NEXT: %[[DATA3:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[INPUT2]] : tensor<4xf16>) outs(%[[INIT3]] : tensor<4xf32>) : tensor<4xf32>
// CHECK-NEXT: %[[INIT4:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[INIT3]] : tensor<4xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[DATA4:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[DATA3]] : tensor<4xf32>) outs(%[[INIT4]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[DATA4]] : tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16> {domain_id = 2 : i64}
// CHECK-NEXT: return %[[DOMAIN2]] : tensor<4xf16>
// CHECK-NEXT: }
func.func @mixed_chain(
    %ctx: !hipsr.context, %input: tensor<4xf32>) -> tensor<4xf16> {
  %init0 = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %data0 = hipsr.cast(%ctx) ins(%input : tensor<4xf32>)
      outs(%init0 : tensor<4xf16>) : tensor<4xf16>
  %init1 = hipsr.placeholder(%ctx)
      ins(%init0 : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32>
  %data1 = hipsr.cast(%ctx) ins(%data0 : tensor<4xf16>)
      outs(%init1 : tensor<4xf32>) : tensor<4xf32>
  %init2 = hipsr.placeholder(%ctx)
      ins(%init1 : tensor<4xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %data2 = hipsr.cast(%ctx) ins(%data1 : tensor<4xf32>)
      outs(%init2 : tensor<4xf16>) : tensor<4xf16>
  %init3 = hipsr.placeholder(%ctx)
      ins(%init2 : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32>
  %data3 = hipsr.cast(%ctx) ins(%data2 : tensor<4xf16>)
      outs(%init3 : tensor<4xf32>) : tensor<4xf32>
  %init4 = hipsr.placeholder(%ctx)
      ins(%init3 : tensor<4xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %data4 = hipsr.cast(%ctx) ins(%data3 : tensor<4xf32>)
      outs(%init4 : tensor<4xf16>) : tensor<4xf16>
  return %data4 : tensor<4xf16>
}

// -----

// Parallel barriers share a domain. One branch grows deeper, a later
// independent branch returns to domain zero, and normal joins use the deepest
// input domain.
// CHECK-LABEL: func.func @multi_branch(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32>) -> tensor<4xf16> {
// CHECK-NEXT: %[[DOMAIN0:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[ROOT_INIT:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[ROOT:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32>) outs(%[[ROOT_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: %[[INDEPENDENT_INIT:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[INDEPENDENT:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32>) outs(%[[INDEPENDENT_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[ROOT]], %[[INDEPENDENT]] : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16>, tensor<4xf16> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]]#0 : !hipsr.context, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[ROOT1:.*]]: tensor<4xf16>):
// CHECK-NEXT: %[[LHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT1]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
// CHECK-NEXT: %[[LHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT1]] : tensor<4xf16>) outs(%[[LHS_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: %[[LHS_NORMAL_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[LHS_INIT]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[LHS_NORMAL:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[LHS]] : tensor<4xf16>) outs(%[[LHS_NORMAL_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: %[[RHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT1]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
// CHECK-NEXT: %[[RHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT1]] : tensor<4xf16>) outs(%[[RHS_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: %[[RHS_NORMAL_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[RHS_INIT]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[RHS_NORMAL:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[RHS]] : tensor<4xf16>) outs(%[[RHS_NORMAL_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[LHS_NORMAL]], %[[RHS_NORMAL]] : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16>, tensor<4xf16> {domain_id = 1 : i64}
// CHECK-NEXT: %[[DOMAIN2:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN1]]#0, %[[DOMAIN1]]#1, %[[DOMAIN0]]#1 : !hipsr.context, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[CTX2:.*]]: !hipsr.context, %[[LHS2:.*]]: tensor<4xf16>, %[[RHS2:.*]]: tensor<4xf16>, %[[INDEPENDENT2:.*]]: tensor<4xf16>):
// CHECK-NEXT: %[[LHS_DEEP_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[LHS2]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
// CHECK-NEXT: %[[LHS_DEEP:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[LHS2]] : tensor<4xf16>) outs(%[[LHS_DEEP_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: %[[DEEP_JOIN_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[LHS_DEEP_INIT]], %[[RHS2]] : tensor<4xf16>, tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[DEEP_JOIN:.*]] = hipsr.add(%[[CTX2]]) ins(%[[LHS_DEEP]], %[[RHS2]] : tensor<4xf16>, tensor<4xf16>) outs(%[[DEEP_JOIN_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: %[[JOIN_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[DEEP_JOIN_INIT]], %[[INDEPENDENT2]] : tensor<4xf16>, tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[JOIN:.*]] = hipsr.add(%[[CTX2]]) ins(%[[DEEP_JOIN]], %[[INDEPENDENT2]] : tensor<4xf16>, tensor<4xf16>) outs(%[[JOIN_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[JOIN]] : tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16> {domain_id = 2 : i64}
// CHECK-NEXT: %[[DOMAIN3:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN2]] : !hipsr.context, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[CTX3:.*]]: !hipsr.context, %[[INPUT3:.*]]: tensor<4xf16>):
// CHECK-NEXT: %[[RESULT_INIT:.*]] = hipsr.placeholder(%[[CTX3]]) ins(%[[INPUT3]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.cast(%[[CTX3]]) ins(%[[INPUT3]] : tensor<4xf16>) outs(%[[RESULT_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[RESULT]] : tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16> {domain_id = 3 : i64}
// CHECK-NEXT: return %[[DOMAIN3]] : tensor<4xf16>
// CHECK-NEXT: }
func.func @multi_branch(
    %ctx: !hipsr.context, %input: tensor<4xf32>) -> tensor<4xf16> {
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %root = hipsr.cast(%ctx) ins(%input : tensor<4xf32>)
      outs(%root_init : tensor<4xf16>) : tensor<4xf16>
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
  %lhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16>)
      outs(%lhs_init : tensor<4xf16>) : tensor<4xf16>
  %lhs_normal_init = hipsr.placeholder(%ctx)
      ins(%lhs_init : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %lhs_normal = hipsr.cast(%ctx) ins(%lhs : tensor<4xf16>)
      outs(%lhs_normal_init : tensor<4xf16>) : tensor<4xf16>
  %lhs_deep_init = hipsr.placeholder(%ctx)
      ins(%lhs_normal_init : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
  %lhs_deep = hipsr.cast(%ctx) ins(%lhs_normal : tensor<4xf16>)
      outs(%lhs_deep_init : tensor<4xf16>) : tensor<4xf16>
  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
  %rhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16>)
      outs(%rhs_init : tensor<4xf16>) : tensor<4xf16>
  %rhs_normal_init = hipsr.placeholder(%ctx)
      ins(%rhs_init : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %rhs_normal = hipsr.cast(%ctx) ins(%rhs : tensor<4xf16>)
      outs(%rhs_normal_init : tensor<4xf16>) : tensor<4xf16>
  %independent_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %independent = hipsr.cast(%ctx) ins(%input : tensor<4xf32>)
      outs(%independent_init : tensor<4xf16>) : tensor<4xf16>
  %deep_join_init = hipsr.placeholder(%ctx)
      ins(%lhs_deep_init, %rhs_normal_init : tensor<4xf16>, tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %deep_join = hipsr.add(%ctx)
      ins(%lhs_deep, %rhs_normal : tensor<4xf16>, tensor<4xf16>)
      outs(%deep_join_init : tensor<4xf16>) : tensor<4xf16>
  %join_init = hipsr.placeholder(%ctx)
      ins(%deep_join_init, %independent_init
          : tensor<4xf16>, tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %join = hipsr.add(%ctx)
      ins(%deep_join, %independent : tensor<4xf16>, tensor<4xf16>)
      outs(%join_init : tensor<4xf16>) : tensor<4xf16>
  %result_init = hipsr.placeholder(%ctx)
      ins(%join_init : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
  %result = hipsr.cast(%ctx) ins(%join : tensor<4xf16>)
      outs(%result_init : tensor<4xf16>) : tensor<4xf16>
  return %result : tensor<4xf16>
}

// -----

// Multi-result DPS values cross one boundary independently and keep their
// result order.
// CHECK-LABEL: func.func @multi_result_boundaries(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf16>)
// CHECK-SAME: -> (tensor<4xf16>, tensor<4xf16>) {
// CHECK-NEXT: %[[DOMAIN0:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf16>):
// CHECK-NEXT: %[[ROOT_INITS:.*]]:2 = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: %[[ROOT:.*]]:2 = hipsr.compute(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf16>) outs(%[[ROOT_INITS]]#0, %[[ROOT_INITS]]#1 : tensor<4xf16>, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.*]]: !hipsr.context, %[[BODY_INPUT:.*]]: tensor<4xf16>, %[[LHS_DEST:.*]]: tensor<4xf16>, %[[RHS_DEST:.*]]: tensor<4xf16>):
// CHECK-NEXT: hipsr.compute_yield %[[LHS_DEST]], %[[RHS_DEST]] : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: } : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[ROOT]]#0, %[[ROOT]]#1 : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16>, tensor<4xf16> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]]#0, %[[DOMAIN0]]#1 : !hipsr.context, tensor<4xf16>, tensor<4xf16>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[LHS_INPUT:.*]]: tensor<4xf16>, %[[RHS_INPUT:.*]]: tensor<4xf16>):
// CHECK-NEXT: %[[LHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[LHS_INPUT]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
// CHECK-NEXT: %[[LHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[LHS_INPUT]] : tensor<4xf16>) outs(%[[LHS_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: %[[RHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[RHS_INPUT]] : tensor<4xf16>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
// CHECK-NEXT: %[[RHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[RHS_INPUT]] : tensor<4xf16>) outs(%[[RHS_INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NEXT: hipsr.pool_domain_yield %[[LHS]], %[[RHS]] : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: } -> tensor<4xf16>, tensor<4xf16> {domain_id = 1 : i64}
// CHECK-NEXT: return %[[DOMAIN1]]#0, %[[DOMAIN1]]#1 : tensor<4xf16>, tensor<4xf16>
// CHECK-NEXT: }
func.func @multi_result_boundaries(
    %ctx: !hipsr.context, %input: tensor<4xf16>)
    -> (tensor<4xf16>, tensor<4xf16>) {
  %root_inits:2 = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<4xf16>, tensor<4xf16>
  %root:2 = hipsr.compute(%ctx)
      ins(%input : tensor<4xf16>)
      outs(%root_inits#0, %root_inits#1 : tensor<4xf16>, tensor<4xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %body_input: tensor<4xf16>,
       %lhs_dest: tensor<4xf16>, %rhs_dest: tensor<4xf16>):
    hipsr.compute_yield %lhs_dest, %rhs_dest
        : tensor<4xf16>, tensor<4xf16>
  } : tensor<4xf16>, tensor<4xf16>
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_inits#0 : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
  %lhs = hipsr.cast(%ctx) ins(%root#0 : tensor<4xf16>)
      outs(%lhs_init : tensor<4xf16>) : tensor<4xf16>
  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_inits#1 : tensor<4xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16>
  %rhs = hipsr.cast(%ctx) ins(%root#1 : tensor<4xf16>)
      outs(%rhs_init : tensor<4xf16>) : tensor<4xf16>
  return %lhs, %rhs : tensor<4xf16>, tensor<4xf16>
}

// -----

// A domain without escaping results is retained without an explicit yield.
// CHECK-LABEL: func.func @no_result_domain(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32>) {
// CHECK-NEXT: hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32>) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_CTX:.*]]: !hipsr.context, %[[DOMAIN_INPUT:.*]]: tensor<4xf32>):
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder(%[[DOMAIN_CTX]]) ins(%[[DOMAIN_INPUT]] : tensor<4xf32>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
// CHECK-NEXT: %[[UNUSED:.*]] = hipsr.cast(%[[DOMAIN_CTX]]) ins(%[[DOMAIN_INPUT]] : tensor<4xf32>) outs(%[[INIT]] : tensor<4xf16>) : tensor<4xf16>
// CHECK-NOT: hipsr.pool_domain_yield
// CHECK-NEXT: } {domain_id = 0 : i64}
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @no_result_domain(
    %ctx: !hipsr.context, %input: tensor<4xf32>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16>
  %unused = hipsr.cast(%ctx) ins(%input : tensor<4xf32>)
      outs(%init : tensor<4xf16>) : tensor<4xf16>
  return
}

// -----

// Empty functions are unchanged.
// CHECK-LABEL: func.func @empty(%{{.*}}: !hipsr.context) {
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @empty(%ctx: !hipsr.context) {
  return
}

// -----

// Declarations are unchanged.
// CHECK-LABEL: func.func private @declaration(i32) -> i32
func.func private @declaration(i32) -> i32
