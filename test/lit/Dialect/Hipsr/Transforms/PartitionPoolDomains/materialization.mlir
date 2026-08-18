// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file -hipsr-partition-pool-domains %s | FileCheck %s

// Diagram notation:
//   name [N, D0] = a normal placeholder and its DPS consumer in domain 0
//   name [B, D1] = a barrier placeholder and its DPS consumer in domain 1
// Diagrams flow from top to bottom. Between paired nodes, each arrow carries
// matching shape and data dependencies.
//
// Normal placeholders stay in the current domain, while barriers advance to
// the next domain.
//
//          input
//            |
//            v
//   data0 [N, D0]
//            |
//            v
//   data1 [B, D1]
//            |
//            v
//   data2 [N, D1]
//            |
//            v
//   data3 [B, D2]
//            |
//            v
//   data4 [N, D2]
//
// CHECK-LABEL: func.func @mixed_chain(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[DOMAIN0:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT: %[[INIT0:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[DATA0:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[INIT0]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[INIT0]], %[[DATA0]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]]#0, %[[DOMAIN0]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[SHAPE_INPUT1:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[DATA_INPUT1:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[INIT1:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[SHAPE_INPUT1]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[DATA1:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[DATA_INPUT1]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[INIT1]] : tensor<4xf32, #hipsr.mem<device>>) : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[INIT2:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[INIT1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[DATA2:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[DATA1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[INIT2]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[INIT2]], %[[DATA2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT: %[[DOMAIN2:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN1]]#0, %[[DOMAIN1]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX2:.*]]: !hipsr.context, %[[SHAPE_INPUT2:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[DATA_INPUT2:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[INIT3:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[SHAPE_INPUT2]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[DATA3:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[DATA_INPUT2]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[INIT3]] : tensor<4xf32, #hipsr.mem<device>>) : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[INIT4:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[INIT3]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[DATA4:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[DATA3]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[INIT4]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[DATA4]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT: return %[[DOMAIN2]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @mixed_chain(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %init0 = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %data0 = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%init0 : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %init1 = hipsr.placeholder(%ctx)
      ins(%init0 : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32, #hipsr.mem<device>>
  %data1 = hipsr.cast(%ctx) ins(%data0 : tensor<4xf16, #hipsr.mem<device>>)
      outs(%init1 : tensor<4xf32, #hipsr.mem<device>>) : tensor<4xf32, #hipsr.mem<device>>
  %init2 = hipsr.placeholder(%ctx)
      ins(%init1 : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %data2 = hipsr.cast(%ctx) ins(%data1 : tensor<4xf32, #hipsr.mem<device>>)
      outs(%init2 : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %init3 = hipsr.placeholder(%ctx)
      ins(%init2 : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32, #hipsr.mem<device>>
  %data3 = hipsr.cast(%ctx) ins(%data2 : tensor<4xf16, #hipsr.mem<device>>)
      outs(%init3 : tensor<4xf32, #hipsr.mem<device>>) : tensor<4xf32, #hipsr.mem<device>>
  %init4 = hipsr.placeholder(%ctx)
      ins(%init3 : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %data4 = hipsr.cast(%ctx) ins(%data3 : tensor<4xf32, #hipsr.mem<device>>)
      outs(%init4 : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  return %data4 : tensor<4xf16, #hipsr.mem<device>>
}

// -----

// Parallel barriers share a domain. One branch grows deeper, a later
// independent branch returns to domain zero, and normal joins use the deepest
// input domain. The independent branch comes directly from the input and joins
// in the second panel.
//
//                             input
//                               |
//                               v
//                         root [N, D0]
//                               |
//                  +------------+------------+
//                  |                         |
//                  v                         v
//             lhs [B, D1]              rhs [B, D1]
//                  |                         |
//                  v                         v
//      lhs_normal [N, D1]      rhs_normal [N, D1]
//                  |                         |
//                  v                         |
//       lhs_deep [B, D2]                    |
//                  |                         |
//                  +------------+------------+
//                               |
//                               v
//                    deep_join [N, D2]
//
//   continued:
//
//       deep_join [N, D2]       independent [N, D0]
//                  |                         |
//                  +------------+------------+
//                               |
//                               v
//                         join [N, D2]
//                               |
//                               v
//                       result [B, D3]
//
// CHECK-LABEL: func.func @multi_branch(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[DOMAIN0:.*]]:4 = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT: %[[ROOT_INIT:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[ROOT:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[ROOT_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[INDEPENDENT_INIT:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[INDEPENDENT:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[INDEPENDENT_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[ROOT_INIT]], %[[ROOT]], %[[INDEPENDENT_INIT]], %[[INDEPENDENT]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]]:4 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]]#0, %[[DOMAIN0]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[ROOT_SHAPE1:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[ROOT_DATA1:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[LHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT_SHAPE1]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT_DATA1]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[LHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LHS_NORMAL_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[LHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LHS_NORMAL:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[LHS]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[LHS_NORMAL_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT_SHAPE1]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT_DATA1]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS_NORMAL_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS_NORMAL:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[RHS]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[RHS_NORMAL_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[LHS_NORMAL_INIT]], %[[LHS_NORMAL]], %[[RHS_NORMAL_INIT]], %[[RHS_NORMAL]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT: %[[DOMAIN2:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN1]]#0, %[[DOMAIN1]]#1, %[[DOMAIN1]]#2, %[[DOMAIN1]]#3, %[[DOMAIN0]]#2, %[[DOMAIN0]]#3 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX2:.*]]: !hipsr.context, %[[LHS_SHAPE2:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[LHS_DATA2:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[RHS_SHAPE2:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[RHS_DATA2:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[INDEPENDENT_SHAPE2:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[INDEPENDENT_DATA2:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[LHS_DEEP_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[LHS_SHAPE2]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LHS_DEEP:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[LHS_DATA2]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[LHS_DEEP_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[DEEP_JOIN_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[LHS_DEEP_INIT]], %[[RHS_SHAPE2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[DEEP_JOIN:.*]] = hipsr.add(%[[CTX2]]) ins(%[[LHS_DEEP]], %[[RHS_DATA2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[DEEP_JOIN_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[JOIN_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[DEEP_JOIN_INIT]], %[[INDEPENDENT_SHAPE2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[JOIN:.*]] = hipsr.add(%[[CTX2]]) ins(%[[DEEP_JOIN]], %[[INDEPENDENT_DATA2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[JOIN_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[JOIN_INIT]], %[[JOIN]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT: %[[DOMAIN3:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN2]]#0, %[[DOMAIN2]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX3:.*]]: !hipsr.context, %[[SHAPE_INPUT3:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[DATA_INPUT3:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[RESULT_INIT:.*]] = hipsr.placeholder(%[[CTX3]]) ins(%[[SHAPE_INPUT3]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RESULT:.*]] = hipsr.cast(%[[CTX3]]) ins(%[[DATA_INPUT3]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[RESULT_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[RESULT]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 3 : i64}
// CHECK-NEXT: return %[[DOMAIN3]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @multi_branch(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%root_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %lhs_normal_init = hipsr.placeholder(%ctx)
      ins(%lhs_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs_normal = hipsr.cast(%ctx) ins(%lhs : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_normal_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %lhs_deep_init = hipsr.placeholder(%ctx)
      ins(%lhs_normal_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs_deep = hipsr.cast(%ctx) ins(%lhs_normal : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_deep_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %rhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%rhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %rhs_normal_init = hipsr.placeholder(%ctx)
      ins(%rhs_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %rhs_normal = hipsr.cast(%ctx) ins(%rhs : tensor<4xf16, #hipsr.mem<device>>)
      outs(%rhs_normal_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %independent_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %independent = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%independent_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %deep_join_init = hipsr.placeholder(%ctx)
      ins(%lhs_deep_init, %rhs_normal_init : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %deep_join = hipsr.add(%ctx)
      ins(%lhs_deep, %rhs_normal : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      outs(%deep_join_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %join_init = hipsr.placeholder(%ctx)
      ins(%deep_join_init, %independent_init
          : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %join = hipsr.add(%ctx)
      ins(%deep_join, %independent : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      outs(%join_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %result_init = hipsr.placeholder(%ctx)
      ins(%join_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %result = hipsr.cast(%ctx) ins(%join : tensor<4xf16, #hipsr.mem<device>>)
      outs(%result_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  return %result : tensor<4xf16, #hipsr.mem<device>>
}

// -----

// A barrier join puts the merge point one domain deeper than both parallel
// arms. Shape and data values for both arms cross the second boundary.
//
//                 root [N, D0]
//                       |
//             +---------+---------+
//             |                   |
//             v                   v
//       lhs [B, D1]         rhs [B, D1]
//             |                   |
//             +---------+---------+
//                       |
//                       v
//                join [B, D2]
//
// CHECK-LABEL: func.func @diamond(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[DOMAIN0:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT: %[[ROOT_INIT:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[ROOT:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[ROOT_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[ROOT_INIT]], %[[ROOT]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]]:4 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]]#0, %[[DOMAIN0]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[ROOT_SHAPE:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[ROOT_DATA:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[LHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT_DATA]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[LHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT_DATA]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[LHS_INIT]], %[[LHS]], %[[RHS_INIT]], %[[RHS]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT: %[[DOMAIN2:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN1]]#0, %[[DOMAIN1]]#2, %[[DOMAIN1]]#1, %[[DOMAIN1]]#3 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX2:.*]]: !hipsr.context, %[[LHS_SHAPE:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[RHS_SHAPE:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[LHS_DATA:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[RHS_DATA:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[JOIN_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[JOIN:.*]] = hipsr.add(%[[CTX2]]) ins(%[[LHS_DATA]], %[[RHS_DATA]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[JOIN_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[JOIN]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT: return %[[DOMAIN2]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @diamond(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%root_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %rhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%rhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %join_init = hipsr.placeholder(%ctx)
      ins(%lhs_init, %rhs_init : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %join = hipsr.add(%ctx)
      ins(%lhs, %rhs : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      outs(%join_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  return %join : tensor<4xf16, #hipsr.mem<device>>
}

// -----

// Two diamonds cascade. Each barrier fan-out starts a new domain, while each
// normal join stays with its two arms.
//
//                         root [N, D0]
//                               |
//                  +------------+------------+
//                  |                         |
//                  v                         v
//          upper_lhs [B, D1]        upper_rhs [B, D1]
//                  |                         |
//                  +------------+------------+
//                               |
//                               v
//                    upper_join [N, D1]
//                               |
//                  +------------+------------+
//                  |                         |
//                  v                         v
//          lower_lhs [B, D2]        lower_rhs [B, D2]
//                  |                         |
//                  +------------+------------+
//                               |
//                               v
//                    lower_join [N, D2]
//
// CHECK-LABEL: func.func @cascaded_diamonds(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[DOMAIN0:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT: %[[ROOT_INIT:.*]] = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[ROOT:.*]] = hipsr.cast(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[ROOT_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[ROOT_INIT]], %[[ROOT]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]]#0, %[[DOMAIN0]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[ROOT_SHAPE:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[ROOT_DATA:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[UPPER_LHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[UPPER_LHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT_DATA]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[UPPER_LHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[UPPER_RHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[ROOT_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[UPPER_RHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[ROOT_DATA]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[UPPER_RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[UPPER_JOIN_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[UPPER_LHS_INIT]], %[[UPPER_RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[UPPER_JOIN:.*]] = hipsr.add(%[[CTX1]]) ins(%[[UPPER_LHS]], %[[UPPER_RHS]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[UPPER_JOIN_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[UPPER_JOIN_INIT]], %[[UPPER_JOIN]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT: %[[DOMAIN2:.*]] = hipsr.pool_domain(%[[CTX]], %[[DOMAIN1]]#0, %[[DOMAIN1]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX2:.*]]: !hipsr.context, %[[UPPER_SHAPE:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[UPPER_DATA:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[LOWER_LHS_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[UPPER_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LOWER_LHS:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[UPPER_DATA]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[LOWER_LHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LOWER_RHS_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[UPPER_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LOWER_RHS:.*]] = hipsr.cast(%[[CTX2]]) ins(%[[UPPER_DATA]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[LOWER_RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LOWER_JOIN_INIT:.*]] = hipsr.placeholder(%[[CTX2]]) ins(%[[LOWER_LHS_INIT]], %[[LOWER_RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LOWER_JOIN:.*]] = hipsr.add(%[[CTX2]]) ins(%[[LOWER_LHS]], %[[LOWER_RHS]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[LOWER_JOIN_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[LOWER_JOIN]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT: return %[[DOMAIN2]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @cascaded_diamonds(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%root_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %upper_lhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %upper_lhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%upper_lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %upper_rhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %upper_rhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%upper_rhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %upper_join_init = hipsr.placeholder(%ctx)
      ins(%upper_lhs_init, %upper_rhs_init : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %upper_join = hipsr.add(%ctx)
      ins(%upper_lhs, %upper_rhs : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      outs(%upper_join_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %lower_lhs_init = hipsr.placeholder(%ctx)
      ins(%upper_join_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lower_lhs = hipsr.cast(%ctx) ins(%upper_join : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lower_lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %lower_rhs_init = hipsr.placeholder(%ctx)
      ins(%upper_join_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lower_rhs = hipsr.cast(%ctx) ins(%upper_join : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lower_rhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %lower_join_init = hipsr.placeholder(%ctx)
      ins(%lower_lhs_init, %lower_rhs_init : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %lower_join = hipsr.add(%ctx)
      ins(%lower_lhs, %lower_rhs : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
      outs(%lower_join_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  return %lower_join : tensor<4xf16, #hipsr.mem<device>>
}

// -----

// Multi-result placeholder and DPS values cross one boundary independently
// and keep their result order. root#i denotes placeholder result i and its
// matching DPS result.
//
//                           input
//                             |
//                +------------+------------+
//                |                         |
//                v                         v
//       root#0 [N, D0]            root#1 [N, D0]
//                |                         |
//                v                         v
//          lhs [B, D1]              rhs [B, D1]
//                |                         |
//                +------------+------------+
//                             |
//                             v
//                     return (lhs, rhs)
//
// CHECK-LABEL: func.func @multi_result_boundaries(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf16, #hipsr.mem<device>>)
// CHECK-SAME: -> (tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[DOMAIN0:.*]]:4 = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX0:.*]]: !hipsr.context, %[[INPUT0:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[ROOT_INITS:.*]]:2 = hipsr.placeholder(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[ROOT:.*]]:2 = hipsr.compute(%[[CTX0]]) ins(%[[INPUT0]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[ROOT_INITS]]#0, %[[ROOT_INITS]]#1 : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[BODY_CTX:.*]]: !hipsr.context, %[[BODY_INPUT:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[LHS_DEST:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[RHS_DEST:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: hipsr.compute_yield %[[LHS_DEST]], %[[RHS_DEST]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[ROOT_INITS]]#0, %[[ROOT_INITS]]#1, %[[ROOT]]#0, %[[ROOT]]#1 : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT: %[[DOMAIN1:.*]]:2 = hipsr.pool_domain(%[[CTX]], %[[DOMAIN0]]#0, %[[DOMAIN0]]#2, %[[DOMAIN0]]#1, %[[DOMAIN0]]#3 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[CTX1:.*]]: !hipsr.context, %[[LHS_SHAPE:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[LHS_INPUT:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[RHS_SHAPE:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[RHS_INPUT:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT: %[[LHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[LHS_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[LHS_INPUT]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[LHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS_INIT:.*]] = hipsr.placeholder(%[[CTX1]]) ins(%[[RHS_SHAPE]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[RHS:.*]] = hipsr.cast(%[[CTX1]]) ins(%[[RHS_INPUT]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[RHS_INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.pool_domain_yield %[[LHS]], %[[RHS]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT: return %[[DOMAIN1]]#0, %[[DOMAIN1]]#1 : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @multi_result_boundaries(
    %ctx: !hipsr.context, %input: tensor<4xf16, #hipsr.mem<device>>)
    -> (tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
  %root_inits:2 = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
  %root:2 = hipsr.compute(%ctx)
      ins(%input : tensor<4xf16, #hipsr.mem<device>>)
      outs(%root_inits#0, %root_inits#1 : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %body_input: tensor<4xf16, #hipsr.mem<device>>,
       %lhs_dest: tensor<4xf16, #hipsr.mem<device>>, %rhs_dest: tensor<4xf16, #hipsr.mem<device>>):
    hipsr.compute_yield %lhs_dest, %rhs_dest
        : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
  } : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_inits#0 : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root#0 : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_inits#1 : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %rhs = hipsr.cast(%ctx) ins(%root#1 : tensor<4xf16, #hipsr.mem<device>>)
      outs(%rhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  return %lhs, %rhs : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
}

// -----

// A domain without escaping results is retained without an explicit yield.
//
//          input
//            |
//            v
//   unused [N, D0]
//            |
//            v
//   (no escaping value)
//
//   return (no operands)
//
// CHECK-LABEL: func.func @no_result_domain(
// CHECK-SAME: %[[CTX:.*]]: !hipsr.context, %[[INPUT:.*]]: tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: ^bb0(%[[DOMAIN_CTX:.*]]: !hipsr.context, %[[DOMAIN_INPUT:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT: %[[INIT:.*]] = hipsr.placeholder(%[[DOMAIN_CTX]]) ins(%[[DOMAIN_INPUT]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[UNUSED:.*]] = hipsr.cast(%[[DOMAIN_CTX]]) ins(%[[DOMAIN_INPUT]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NOT: hipsr.pool_domain_yield
// CHECK-NEXT: } {domain_id = 0 : i64}
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @no_result_domain(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %unused = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  return
}

// -----

// Empty functions are unchanged.
//
//   (no partitionable operations)
//                |
//                v
//             return
//   pool domains: none
//
// CHECK-LABEL: func.func @empty(%{{.*}}: !hipsr.context) {
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @empty(%ctx: !hipsr.context) {
  return
}

// -----

// Declarations are unchanged.
//
//          i32 input
//              |
//              v
//   @declaration (no body)
//              |
//              v
//         i32 result
//   pool domains: none
//
// CHECK-LABEL: func.func private @declaration(i32) -> i32
func.func private @declaration(i32) -> i32
