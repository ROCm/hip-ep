// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics \
// RUN:   --hipsr-pool-alloc='emit-pool-report=true'
// RUN: hip-mlir-opt %s -split-input-file --hipsr-pool-alloc | FileCheck %s

// The report is opt-in and advisory: RUN line 1 asserts the remark wording,
// RUN line 2 asserts the pass emits the same IR without the option. Byte totals
// appear only when every extent is a constant; a domain with a dynamic extent
// reports counts alone because the saving would be a symbolic ratio.

// a1 (16384 B) and a2 (8192 B) have disjoint lifetimes, so one group sized by
// the larger member replaces both: naive is both aligned separately, and slack
// is what a2 leaves unused while it borrows a1-sized space.
// CHECK-LABEL: func.func @report_static
// CHECK: hipsr.get_pool
// CHECK-NOT: hipsr.get_pool
func.func @report_static(%ctx: !hipsr.context,
                         %in: memref<8x1024xf16, #hipsr.mem<device>>,
                         %in2: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: domain 0 pooled 2 allocs into 1 groups (reused 1); pool 16384/24576 B (saved 33%), slack 8192 B}}
  hipsr.pool_domain(%ctx, %in, %in2 :
      !hipsr.context,
      memref<8x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %din: memref<8x1024xf16, #hipsr.mem<device>>,
       %din2: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%din : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din2, %din2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// One dynamic member is enough to drop the byte totals, even though the other
// member is static.
// CHECK-LABEL: func.func @report_dynamic
// CHECK: hipsr.get_pool
// CHECK: memref.view
// CHECK: memref.view
func.func @report_dynamic(%ctx: !hipsr.context,
                          %in: memref<?x512xf16, #hipsr.mem<device>>,
                          %sin: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: domain 0 pooled 2 allocs into 2 groups (reused 0)}}
  hipsr.pool_domain(%ctx, %in, %sin :
      !hipsr.context,
      memref<?x512xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %din: memref<?x512xf16, #hipsr.mem<device>>,
       %dsin: memref<4x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %a_dyn = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    %a_stat = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a_dyn : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%dsin, %dsin : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a_stat : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_dyn, %a_dyn : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%din : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_stat, %a_stat : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%dsin : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// Two domains in one module: the id is a per-run counter, not a per-function one.
// CHECK-LABEL: func.func @report_first_domain
// CHECK: hipsr.get_pool
func.func @report_first_domain(%ctx: !hipsr.context,
                               %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: domain 0 pooled 1 allocs into 1 groups (reused 0); pool 8192/8192 B (saved 0%), slack 0 B}}
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// CHECK-LABEL: func.func @report_second_domain
// CHECK: hipsr.get_pool
func.func @report_second_domain(%ctx: !hipsr.context,
                                %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: domain 1 pooled 1 allocs into 1 groups (reused 0); pool 8192/8192 B (saved 0%), slack 0 B}}
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// A tensor-mode domain is still counted and reported, so an id gap never means
// a domain was missed.
// CHECK-LABEL: func.func @report_empty
// CHECK-NOT: hipsr.get_pool
func.func @report_empty(%ctx: !hipsr.context,
                        %in: tensor<3x4xf32>) -> tensor<2x4xi64> {
  // expected-remark@+1 {{hipsr-pool-alloc: domain 0 has no poolable allocations}}
  %0 = hipsr.pool_domain(%ctx, %in : !hipsr.context, tensor<3x4xf32>) {
  ^bb0(%dctx: !hipsr.context, %din: tensor<3x4xf32>):
    %buf = tensor.empty() : tensor<2x4xi64>
    hipsr.pool_domain_yield %buf : tensor<2x4xi64>
  } -> tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}
