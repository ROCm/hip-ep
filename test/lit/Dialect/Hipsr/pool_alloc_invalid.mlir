// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-pool-alloc

func.func @empty_domain() {
  // expected-error@+1 {{hipsr-pool-alloc: pool_domain has no poolable allocation}}
  hipsr.pool_domain() {
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

func.func @alloc_without_dps_write(%ctx: !hipsr.context,
                                   %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{hipsr-pool-alloc: pool_domain has no poolable allocation}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %unwritten = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%unwritten, %unwritten
                             : memref<4x1024xf16, #hipsr.mem<device>>,
                               memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

func.func @no_context(%in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-error@+1 {{hipsr-pool-alloc: pool_domain has no context}}
  hipsr.pool_domain(%in : memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%din: memref<4x1024xf16, #hipsr.mem<device>>):
    %cst = arith.constant 0.0 : f16
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    linalg.fill ins(%cst : f16)
           outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    linalg.copy ins(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
           outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}
