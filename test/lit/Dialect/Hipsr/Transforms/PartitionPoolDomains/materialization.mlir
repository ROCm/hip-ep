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
// CHECK-LABEL:   func.func @mixed_chain(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_0:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_0:.*]] = hipsr.cast(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_0]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_0]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_1:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_2:.*]]: !hipsr.context, %[[VAL_3:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_1:.*]] = hipsr.placeholder(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_1:.*]] = hipsr.cast(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_1]] : tensor<4xf32, #hipsr.mem<device>>) : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_2:.*]] = hipsr.placeholder(%[[VAL_2]]) ins(%[[PLACEHOLDER_1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_2:.*]] = hipsr.cast(%[[VAL_2]]) ins(%[[CAST_1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_2]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_2]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_2:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_1]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_4:.*]]: !hipsr.context, %[[VAL_5:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_3:.*]] = hipsr.placeholder(%[[VAL_4]]) ins(%[[VAL_5]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_3:.*]] = hipsr.cast(%[[VAL_4]]) ins(%[[VAL_5]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_3]] : tensor<4xf32, #hipsr.mem<device>>) : tensor<4xf32, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_4:.*]] = hipsr.placeholder(%[[VAL_4]]) ins(%[[PLACEHOLDER_3]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_4:.*]] = hipsr.cast(%[[VAL_4]]) ins(%[[CAST_3]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_4]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_4]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_2]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }

func.func @mixed_chain(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %init0 = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %data0 = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%init0 : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %init1 = hipsr.placeholder(%ctx)
      ins(%data0 : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf32, #hipsr.mem<device>>
  %data1 = hipsr.cast(%ctx) ins(%data0 : tensor<4xf16, #hipsr.mem<device>>)
      outs(%init1 : tensor<4xf32, #hipsr.mem<device>>) : tensor<4xf32, #hipsr.mem<device>>
  %init2 = hipsr.placeholder(%ctx)
      ins(%init1 : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %data2 = hipsr.cast(%ctx) ins(%data1 : tensor<4xf32, #hipsr.mem<device>>)
      outs(%init2 : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %init3 = hipsr.placeholder(%ctx)
      ins(%data2 : tensor<4xf16, #hipsr.mem<device>>)
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
// CHECK-LABEL:   func.func @multi_branch(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]]:3 = hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_0:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_0:.*]] = hipsr.cast(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_0]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_1:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_1:.*]] = hipsr.cast(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_1]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_0]], %[[PLACEHOLDER_1]], %[[CAST_1]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_1:.*]]:3 = hipsr.pool_domain(%[[ARG0]], %[[VAL_2:.*]]#0 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_3:.*]]: !hipsr.context, %[[VAL_4:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_2:.*]] = hipsr.placeholder(%[[VAL_3]]) ins(%[[VAL_4]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_2:.*]] = hipsr.cast(%[[VAL_3]]) ins(%[[VAL_4]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_2]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_3:.*]] = hipsr.placeholder(%[[VAL_3]]) ins(%[[PLACEHOLDER_2]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_3:.*]] = hipsr.cast(%[[VAL_3]]) ins(%[[CAST_2]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_3]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_4:.*]] = hipsr.placeholder(%[[VAL_3]]) ins(%[[VAL_4]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_4:.*]] = hipsr.cast(%[[VAL_3]]) ins(%[[VAL_4]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_4]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_5:.*]] = hipsr.placeholder(%[[VAL_3]]) ins(%[[PLACEHOLDER_4]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_5:.*]] = hipsr.cast(%[[VAL_3]]) ins(%[[CAST_4]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_5]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_3]], %[[PLACEHOLDER_5]], %[[CAST_5]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_2:.*]] = hipsr.pool_domain(%[[ARG0]], %[[VAL_5:.*]]#0, %[[VAL_5]]#1, %[[VAL_5]]#2, %[[VAL_6:.*]]#1, %[[VAL_6]]#2 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_7:.*]]: !hipsr.context, %[[VAL_8:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_9:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_10:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_11:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_12:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_6:.*]] = hipsr.placeholder(%[[VAL_7]]) ins(%[[VAL_8]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_6:.*]] = hipsr.cast(%[[VAL_7]]) ins(%[[VAL_8]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_6]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_7:.*]] = hipsr.placeholder(%[[VAL_7]]) ins(%[[PLACEHOLDER_6]], %[[VAL_9]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ADD_0:.*]] = hipsr.add(%[[VAL_7]]) ins(%[[CAST_6]], %[[VAL_10]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_7]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_8:.*]] = hipsr.placeholder(%[[VAL_7]]) ins(%[[PLACEHOLDER_7]], %[[VAL_11]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ADD_1:.*]] = hipsr.add(%[[VAL_7]]) ins(%[[ADD_0]], %[[VAL_12]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_8]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ADD_1]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_3:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_2]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_13:.*]]: !hipsr.context, %[[VAL_14:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_9:.*]] = hipsr.placeholder(%[[VAL_13]]) ins(%[[VAL_14]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_7:.*]] = hipsr.cast(%[[VAL_13]]) ins(%[[VAL_14]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_9]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_7]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 3 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_3]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }

func.func @multi_branch(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%root_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %lhs_normal_init = hipsr.placeholder(%ctx)
      ins(%lhs_init : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs_normal = hipsr.cast(%ctx) ins(%lhs : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_normal_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %lhs_deep_init = hipsr.placeholder(%ctx)
      ins(%lhs_normal : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs_deep = hipsr.cast(%ctx) ins(%lhs_normal : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_deep_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root : tensor<4xf16, #hipsr.mem<device>>)
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
      ins(%join : tensor<4xf16, #hipsr.mem<device>>)
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
// CHECK-LABEL:   func.func @diamond(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_0:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_0:.*]] = hipsr.cast(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_0]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_0]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_1:.*]]:2 = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_2:.*]]: !hipsr.context, %[[VAL_3:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_1:.*]] = hipsr.placeholder(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_1:.*]] = hipsr.cast(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_1]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_2:.*]] = hipsr.placeholder(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_2:.*]] = hipsr.cast(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_2]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_1]], %[[CAST_2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_2:.*]] = hipsr.pool_domain(%[[ARG0]], %[[VAL_4:.*]]#0, %[[VAL_4]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_5:.*]]: !hipsr.context, %[[VAL_6:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_7:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_3:.*]] = hipsr.placeholder(%[[VAL_5]]) ins(%[[VAL_6]], %[[VAL_7]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ADD_0:.*]] = hipsr.add(%[[VAL_5]]) ins(%[[VAL_6]], %[[VAL_7]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_3]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ADD_0]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_2]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }

func.func @diamond(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%root_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %rhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%rhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %join_init = hipsr.placeholder(%ctx)
      ins(%lhs, %rhs : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>)
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
// CHECK-LABEL:   func.func @cascaded_diamonds(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]] = hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_0:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_0:.*]] = hipsr.cast(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_0]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_0]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_1:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_0]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_2:.*]]: !hipsr.context, %[[VAL_3:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_1:.*]] = hipsr.placeholder(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_1:.*]] = hipsr.cast(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_1]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_2:.*]] = hipsr.placeholder(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_2:.*]] = hipsr.cast(%[[VAL_2]]) ins(%[[VAL_3]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_2]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_3:.*]] = hipsr.placeholder(%[[VAL_2]]) ins(%[[PLACEHOLDER_1]], %[[PLACEHOLDER_2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ADD_0:.*]] = hipsr.add(%[[VAL_2]]) ins(%[[CAST_1]], %[[CAST_2]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_3]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ADD_0]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_2:.*]] = hipsr.pool_domain(%[[ARG0]], %[[POOL_DOMAIN_1]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_4:.*]]: !hipsr.context, %[[VAL_5:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_4:.*]] = hipsr.placeholder(%[[VAL_4]]) ins(%[[VAL_5]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_3:.*]] = hipsr.cast(%[[VAL_4]]) ins(%[[VAL_5]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_4]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_5:.*]] = hipsr.placeholder(%[[VAL_4]]) ins(%[[VAL_5]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_4:.*]] = hipsr.cast(%[[VAL_4]]) ins(%[[VAL_5]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_5]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_6:.*]] = hipsr.placeholder(%[[VAL_4]]) ins(%[[PLACEHOLDER_4]], %[[PLACEHOLDER_5]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[ADD_1:.*]] = hipsr.add(%[[VAL_4]]) ins(%[[CAST_3]], %[[CAST_4]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_6]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[ADD_1]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>> {domain_id = 2 : i64}
// CHECK-NEXT:      return %[[POOL_DOMAIN_2]] : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }

func.func @cascaded_diamonds(
    %ctx: !hipsr.context, %input: tensor<4xf32, #hipsr.mem<device>>) -> tensor<4xf16, #hipsr.mem<device>> {
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<4xf32, #hipsr.mem<device>>)
      outs(%root_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %upper_lhs_init = hipsr.placeholder(%ctx)
      ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %upper_lhs = hipsr.cast(%ctx) ins(%root : tensor<4xf16, #hipsr.mem<device>>)
      outs(%upper_lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %upper_rhs_init = hipsr.placeholder(%ctx)
      ins(%root : tensor<4xf16, #hipsr.mem<device>>)
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
      ins(%upper_join : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lower_lhs = hipsr.cast(%ctx) ins(%upper_join : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lower_lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>

  %lower_rhs_init = hipsr.placeholder(%ctx)
      ins(%upper_join : tensor<4xf16, #hipsr.mem<device>>)
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
// CHECK-LABEL:   func.func @multi_result_boundaries(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xf16, #hipsr.mem<device>>) -> (tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      %[[POOL_DOMAIN_0:.*]]:2 = hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_0:.*]]:2 = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[COMPUTE_0:.*]]:2 = hipsr.compute(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_0]]#0, %[[PLACEHOLDER_0]]#1 : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:        ^bb0(%[[VAL_2:.*]]: !hipsr.context, %[[VAL_3:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_4:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_5:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:          hipsr.compute_yield %[[VAL_4]], %[[VAL_5]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        } : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
// CHECK-NEXT:      %[[POOL_DOMAIN_1:.*]]:2 = hipsr.pool_domain(%[[ARG0]], %[[VAL_7:.*]]#0, %[[VAL_7]]#1 : !hipsr.context, tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_8:.*]]: !hipsr.context, %[[VAL_9:.*]]: tensor<4xf16, #hipsr.mem<device>>, %[[VAL_10:.*]]: tensor<4xf16, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_1:.*]] = hipsr.placeholder(%[[VAL_8]]) ins(%[[VAL_9]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_0:.*]] = hipsr.cast(%[[VAL_8]]) ins(%[[VAL_9]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_1]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[PLACEHOLDER_2:.*]] = hipsr.placeholder(%[[VAL_8]]) ins(%[[VAL_10]] : tensor<4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_1:.*]] = hipsr.cast(%[[VAL_8]]) ins(%[[VAL_10]] : tensor<4xf16, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_2]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        hipsr.pool_domain_yield %[[CAST_0]], %[[CAST_1]] : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } -> tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>> {domain_id = 1 : i64}
// CHECK-NEXT:      return %[[VAL_11:.*]]#0, %[[VAL_11]]#1 : tensor<4xf16, #hipsr.mem<device>>, tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    }

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
      ins(%root#0 : tensor<4xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<4xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root#0 : tensor<4xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root#1 : tensor<4xf16, #hipsr.mem<device>>)
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
// CHECK-LABEL:   func.func @no_result_domain(
// CHECK-SAME:      %[[ARG0:.*]]: !hipsr.context,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      hipsr.pool_domain(%[[ARG0]], %[[ARG1]] : !hipsr.context, tensor<4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:      ^bb0(%[[VAL_0:.*]]: !hipsr.context, %[[VAL_1:.*]]: tensor<4xf32, #hipsr.mem<device>>):
// CHECK-NEXT:        %[[PLACEHOLDER_0:.*]] = hipsr.placeholder(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:        %[[CAST_0:.*]] = hipsr.cast(%[[VAL_0]]) ins(%[[VAL_1]] : tensor<4xf32, #hipsr.mem<device>>) outs(%[[PLACEHOLDER_0]] : tensor<4xf16, #hipsr.mem<device>>) : tensor<4xf16, #hipsr.mem<device>>
// CHECK-NEXT:      } {domain_id = 0 : i64}
// CHECK-NEXT:      return
// CHECK-NEXT:    }

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
// CHECK-LABEL:   func.func @empty(
// CHECK-SAME:                     %[[ARG0:.*]]: !hipsr.context) {
// CHECK-NEXT:      return
// CHECK-NEXT:    }

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
// CHECK-LABEL:   func.func private @declaration(i32) -> i32

func.func private @declaration(i32) -> i32
