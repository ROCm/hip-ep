// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file --verify-diagnostics \
// RUN:   -hipsr-pool-alloc='emit-pool-report=true' | FileCheck %s

// CHECK-LABEL: func.func @hoisted_allocs
func.func @hoisted_allocs(%ctx: !hipsr.context,
                          %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: insertion point after op 2}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [3,4] group 0 size 8192}}
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [4,5] group 1 size 8192}}
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [5,6] group 0 size 8192}}
    %a3 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>,
                                    memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>,
                                    memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a3 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<4x1024xf16, #hipsr.mem<device>>,
                                    memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @nested_region_user
func.func @nested_region_user(%ctx: !hipsr.context,
                              %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: insertion point after op 0}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [1,2] group 0 size 8192}}
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    scf.execute_region {
      hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
                 outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
      scf.yield
    }
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @alloc_inside_region
func.func @alloc_inside_region(%ctx: !hipsr.context,
                               %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: insertion point after op 0}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [1,2] group 0 size 8192}}
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    scf.execute_region {
      %inner = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
      hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
                 outs(%inner : memref<4x1024xf16, #hipsr.mem<device>>)
      hipsr.add(%dctx) ins(%inner, %inner : memref<4x1024xf16, #hipsr.mem<device>>,
                                            memref<4x1024xf16, #hipsr.mem<device>>)
                 outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
      scf.yield
    }
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @mixed_dtypes
func.func @mixed_dtypes(%ctx: !hipsr.context,
                        %inf32: memref<4x256xf32, #hipsr.mem<device>>,
                        %inf16: memref<?x512xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: insertion point after op 3}}
  hipsr.pool_domain(%ctx, %inf32, %inf16 :
      !hipsr.context,
      memref<4x256xf32, #hipsr.mem<device>>,
      memref<?x512xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %sf32: memref<4x256xf32, #hipsr.mem<device>>,
       %sf16: memref<?x512xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %sf16, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [4,4] group 0 size 4096}}
    %a1 = memref.alloc() : memref<4x256xf32, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [5,5] group 0 size 1024 x 1 dyn}}
    %a2 = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%sf32, %sf32 : memref<4x256xf32, #hipsr.mem<device>>, memref<4x256xf32, #hipsr.mem<device>>)
               outs(%a1 : memref<4x256xf32, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%sf16, %sf16 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @diamond
func.func @diamond(%ctx: !hipsr.context, %in: memref<?x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+1 {{hipsr-pool-alloc: insertion point after op 5}}
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [6,8] group 0 size 2048 x 1 dyn}}
    %a = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [7,9] group 1 size 1024 x 1 dyn}}
    %b = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [8,9] group 2 size 512 x 1 dyn}}
    %c = memref.alloc(%d) : memref<?x256xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [9,10] group 0 size 4096 x 1 dyn}}
    %e = memref.alloc(%d) : memref<?x2048xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a, %a : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%b : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a, %a : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%c : memref<?x256xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%b, %c : memref<?x512xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%e : memref<?x2048xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%e, %e : memref<?x2048xf16, #hipsr.mem<device>>, memref<?x2048xf16, #hipsr.mem<device>>) outs(%din : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}
