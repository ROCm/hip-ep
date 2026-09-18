// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-pool-alloc

func.func @alloc_without_dps_write(%ctx: !hipsr.context,
                                   %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    // expected-error@+1 {{buffer used before written}}
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

// The pool is the domain's only device allocation, so an allocation the pass
// cannot pool must not survive it. Nothing reads this one, so it gets no live
// range and stays a memref.alloc.
func.func @dead_device_alloc(%ctx: !hipsr.context) {
  hipsr.pool_domain(%ctx : !hipsr.context) {
  ^bb0(%dctx: !hipsr.context):
    // expected-warning@+2 {{allocation has no users}}
    // expected-error@+1 {{device allocation is not backed by the pool}}
    %dead = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}
