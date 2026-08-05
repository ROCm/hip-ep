// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file --verify-diagnostics \
// RUN:   -hipsr-partition-pool-domains='emit-analysis-report=true' \
// RUN:   | FileCheck %s --implicit-check-not=hipsr.pool_domain

// Parallel barriers over one producer share the same next domain.
// CHECK-LABEL: func.func @parallel_barriers
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0,2->1,3->1,4->1,5->1,6->1,7->1]}}
func.func @parallel_barriers(
    %ctx: !hipsr.context, %input: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast]}}
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %root = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16>)
      outs(%root_init : tensor<?x8xf16>) : tensor<?x8xf16>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 1 ops [2=hipsr.placeholder,3=hipsr.cast,4=hipsr.placeholder,5=hipsr.cast,6=hipsr.placeholder,7=hipsr.add]}}
  %lhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16>
  %lhs = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16>)
      outs(%lhs_init : tensor<?x8xf16>) : tensor<?x8xf16>

  %rhs_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16>
  %rhs = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16>)
      outs(%rhs_init : tensor<?x8xf16>) : tensor<?x8xf16>

  %sum_init = hipsr.placeholder(%ctx)
      ins(%lhs_init, %rhs_init : tensor<?x8xf16>, tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %sum = hipsr.add(%ctx)
      ins(%lhs, %rhs : tensor<?x8xf16>, tensor<?x8xf16>)
      outs(%sum_init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %sum : tensor<?x8xf16>
}

// -----

// A barrier with only block-argument dependencies remains in domain zero.
// CHECK-LABEL: func.func @root_barrier
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0]}}
func.func @root_barrier(
    %ctx: !hipsr.context, %input: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast]}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16>)
      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// Normal placeholders do not move work beyond the barrier's domain.
// CHECK-LABEL: func.func @normal_after_barrier
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0,2->1,3->1,4->1,5->1,6->1,7->1]}}
func.func @normal_after_barrier(
    %ctx: !hipsr.context, %input: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast]}}
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %root = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16>)
      outs(%root_init : tensor<?x8xf16>) : tensor<?x8xf16>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 1 ops [2=hipsr.placeholder,3=hipsr.cast,4=hipsr.placeholder,5=hipsr.cast,6=hipsr.placeholder,7=hipsr.cast]}}
  %barrier_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16>
  %barrier = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16>)
      outs(%barrier_init : tensor<?x8xf16>) : tensor<?x8xf16>

  %normal_init = hipsr.placeholder(%ctx)
      ins(%barrier_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %normal = hipsr.cast(%ctx) ins(%barrier : tensor<?x8xf16>)
      outs(%normal_init : tensor<?x8xf16>) : tensor<?x8xf16>

  %next_init = hipsr.placeholder(%ctx)
      ins(%normal_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %next = hipsr.cast(%ctx) ins(%normal : tensor<?x8xf16>)
      outs(%next_init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %next : tensor<?x8xf16>
}

// -----

// Later branches can return to earlier domains. A normal join uses its deepest
// input domain, while a barrier join starts the next domain.
// CHECK-LABEL: func.func @mixed_depth_branches
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains [0->0,1->0,2->1,3->1,4->2,5->2,6->1,7->1,8->0,9->0,10->1,11->1,12->3,13->3]}}
func.func @mixed_depth_branches(
    %ctx: !hipsr.context, %input: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 0 ops [0=hipsr.placeholder,1=hipsr.cast,8=hipsr.placeholder,9=hipsr.cast]}}
  %root_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %root = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16>)
      outs(%root_init : tensor<?x8xf16>) : tensor<?x8xf16>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 1 ops [2=hipsr.placeholder,3=hipsr.cast,6=hipsr.placeholder,7=hipsr.cast,10=hipsr.placeholder,11=hipsr.add]}}
  %deep1_init = hipsr.placeholder(%ctx)
      ins(%root_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16>
  %deep1 = hipsr.cast(%ctx) ins(%root : tensor<?x8xf16>)
      outs(%deep1_init : tensor<?x8xf16>) : tensor<?x8xf16>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 2 ops [4=hipsr.placeholder,5=hipsr.cast]}}
  %deep2_init = hipsr.placeholder(%ctx)
      ins(%deep1_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16>
  %deep2 = hipsr.cast(%ctx) ins(%deep1 : tensor<?x8xf16>)
      outs(%deep2_init : tensor<?x8xf16>) : tensor<?x8xf16>

  %middle_init = hipsr.placeholder(%ctx)
      ins(%deep1_init : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %middle = hipsr.cast(%ctx) ins(%deep1 : tensor<?x8xf16>)
      outs(%middle_init : tensor<?x8xf16>) : tensor<?x8xf16>

  %shallow_init = hipsr.placeholder(%ctx)
      ins(%input : tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %shallow = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16>)
      outs(%shallow_init : tensor<?x8xf16>) : tensor<?x8xf16>

  %middle_shallow_init = hipsr.placeholder(%ctx)
      ins(%middle_init, %shallow_init : tensor<?x8xf16>, tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
  %middle_shallow = hipsr.add(%ctx)
      ins(%middle, %shallow : tensor<?x8xf16>, tensor<?x8xf16>)
      outs(%middle_shallow_init : tensor<?x8xf16>) : tensor<?x8xf16>

  // expected-remark@+1 {{hipsr-partition-pool-domains: domain 3 ops [12=hipsr.placeholder,13=hipsr.add]}}
  %result_init = hipsr.placeholder(%ctx)
      ins(%deep2_init, %middle_shallow_init
          : tensor<?x8xf16>, tensor<?x8xf16>)
      {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<?x8xf16>
  %result = hipsr.add(%ctx)
      ins(%deep2, %middle_shallow : tensor<?x8xf16>, tensor<?x8xf16>)
      outs(%result_init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %result : tensor<?x8xf16>
}

// -----

// An empty body has no assignments. Declarations are skipped.
// CHECK-LABEL: func.func @empty()
// expected-remark@+1 {{hipsr-partition-pool-domains: operation domains []}}
func.func @empty() {
  return
}

// CHECK-LABEL: func.func private @declaration
func.func private @declaration(i32) -> i32
