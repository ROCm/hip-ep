// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file --verify-diagnostics \
// RUN:   -hipsr-partition-pool-domains='emit-analysis-report=true' \
// RUN:   -o %t

// Parallel barriers over one producer share the same next domain.
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0,2->1,3->1,4->1,5->1,6->1,7->1]}}
func.func @parallel_barriers(
    %ctx: !hipsr.context, %input: tensor<?x8xf16, #hipsr.mem<device>>) -> tensor<?x8xf16, #hipsr.mem<device>> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast] results [0#0,1#0]}}
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%root_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 1 ops [2=hipsr.placeholder,3=hipsr.cast,4=hipsr.placeholder,5=hipsr.cast,6=hipsr.placeholder,7=hipsr.add] results [7#0]}}
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %rhs = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%rhs_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %sum_init = hipsr.placeholder(%ctx)
      ins(%lhs_init, %rhs_init : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %sum = hipsr.add(%ctx)
      ins(%lhs, %rhs : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%sum_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
  return %sum : tensor<?x8xf16, #hipsr.mem<device>>
}

// -----

// A barrier with only block-argument dependencies remains in domain zero.
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0]}}
func.func @root_barrier(
    %ctx: !hipsr.context, %input: tensor<?x8xf16, #hipsr.mem<device>>) -> tensor<?x8xf16, #hipsr.mem<device>> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast] results [1#0]}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
  return %result : tensor<?x8xf16, #hipsr.mem<device>>
}

// -----

// Normal placeholders do not move work beyond the barrier's domain.
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0,2->1,3->1,4->1,5->1,6->1,7->1]}}
func.func @normal_after_barrier(
    %ctx: !hipsr.context, %input: tensor<?x8xf16, #hipsr.mem<device>>) -> tensor<?x8xf16, #hipsr.mem<device>> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast] results [0#0,1#0]}}
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%root_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 1 ops [2=hipsr.placeholder,3=hipsr.cast,4=hipsr.placeholder,5=hipsr.cast,6=hipsr.placeholder,7=hipsr.cast] results [7#0]}}
  %barrier_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %barrier = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%barrier_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %normal_init = hipsr.placeholder(%ctx)
      ins(%barrier_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %normal = hipsr.cast(%ctx) ins(%barrier : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%normal_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %next_init = hipsr.placeholder(%ctx)
      ins(%normal_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %next = hipsr.cast(%ctx) ins(%normal : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%next_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
  return %next : tensor<?x8xf16, #hipsr.mem<device>>
}

// -----

// Later branches can return to earlier domains. A normal join uses its deepest
// input domain, while a barrier join starts the next domain.
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0,2->1,3->1,4->2,5->2,6->1,7->1,8->0,9->0,10->1,11->1,12->3,13->3]}}
func.func @mixed_depth_branches(
    %ctx: !hipsr.context, %input: tensor<?x8xf16, #hipsr.mem<device>>) -> tensor<?x8xf16, #hipsr.mem<device>> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast,8=hipsr.placeholder,9=hipsr.cast] results [0#0,1#0,8#0,9#0]}}
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %root = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%root_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 1 ops [2=hipsr.placeholder,3=hipsr.cast,6=hipsr.placeholder,7=hipsr.cast,10=hipsr.placeholder,11=hipsr.add] results [2#0,3#0,10#0,11#0]}}
  %deep1_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %deep1 = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%deep1_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 2 ops [4=hipsr.placeholder,5=hipsr.cast] results [4#0,5#0]}}
  %deep2_init = hipsr.placeholder(%ctx)
      ins(%deep1_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %deep2 = hipsr.cast(%ctx) ins(%deep1 : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%deep2_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %middle_init = hipsr.placeholder(%ctx)
      ins(%deep1_init : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %middle = hipsr.cast(%ctx) ins(%deep1 : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%middle_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %shallow_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %shallow = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%shallow_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %middle_shallow_init = hipsr.placeholder(%ctx)
      ins(%middle_init, %shallow_init : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
  %middle_shallow = hipsr.add(%ctx)
      ins(%middle, %shallow : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%middle_shallow_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 3 ops [12=hipsr.placeholder,13=hipsr.add] results [13#0]}}
  %result_init = hipsr.placeholder(%ctx)
      ins(%deep2_init, %middle_shallow_init
          : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %result = hipsr.add(%ctx)
      ins(%deep2, %middle_shallow : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%result_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
  return %result : tensor<?x8xf16, #hipsr.mem<device>>
}

// -----

// Each result used in a later domain is collected independently.
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0,2->1,3->1,4->1,5->1]}}
func.func @multi_result_domain_results(
    %ctx: !hipsr.context, %input: tensor<?x8xf16, #hipsr.mem<device>>)
    -> (tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.compute] results [0#0,0#1,1#0,1#1]}}
  %root_inits:2 = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>
  %root:2 = hipsr.compute(%ctx)
      ins(%input : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%root_inits#0, %root_inits#1 : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>) {
  ^bb0(%body_ctx: !hipsr.context, %body_input: tensor<?x8xf16, #hipsr.mem<device>>,
       %lhs_dest: tensor<?x8xf16, #hipsr.mem<device>>, %rhs_dest: tensor<?x8xf16, #hipsr.mem<device>>):
    hipsr.compute_yield %lhs_dest, %rhs_dest
        : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>
  } : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 1 ops [2=hipsr.placeholder,3=hipsr.cast,4=hipsr.placeholder,5=hipsr.cast] results [3#0,5#0]}}
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_inits#0 : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %lhs = hipsr.cast(%ctx) ins(%root#0 : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%lhs_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>

  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_inits#1 : tensor<?x8xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16, #hipsr.mem<device>>
  %rhs = hipsr.cast(%ctx) ins(%root#1 : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%rhs_init : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
  return %lhs, %rhs : tensor<?x8xf16, #hipsr.mem<device>>, tensor<?x8xf16, #hipsr.mem<device>>
}

// -----

// An empty body has no assignments. Declarations are skipped.
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains []}}
func.func @empty(%ctx: !hipsr.context) {
  return
}

func.func private @declaration(i32) -> i32
