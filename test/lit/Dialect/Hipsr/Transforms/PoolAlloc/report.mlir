// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file --verify-diagnostics \
// RUN:   -hipsr-pool-alloc='emit-pool-report=true' | FileCheck %s

// CHECK-LABEL: func.func @static_line
func.func @static_line(%ctx: !hipsr.context,
                       %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+4 {{hipsr-pool-alloc: domain 0: 5 allocs in 2 groups, 3 saved (60%)}}
  // expected-remark@+3 {{hipsr-pool-alloc: domain 0: before vs after: 20992 bytes vs 12288 bytes, 41% saved}}
  // expected-remark@+2 {{hipsr-pool-alloc: domain 0: group 0: 3 allocs, max 8192 bytes, 39% avg unused}}
  // expected-remark@+1 {{hipsr-pool-alloc: domain 0: group 1: 2 allocs, max 4096 bytes, 25% avg unused}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a0 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a1 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<3x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<1x1024xf16, #hipsr.mem<device>>
    %a4 = memref.alloc() : memref<256xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a0 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a0, %a0 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<3x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<3x1024xf16, #hipsr.mem<device>>, memref<3x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<1x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<1x1024xf16, #hipsr.mem<device>>, memref<1x1024xf16, #hipsr.mem<device>>) outs(%a4 : memref<256xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a4, %a4 : memref<256xf16, #hipsr.mem<device>>, memref<256xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @dyn_common_factor
func.func @dyn_common_factor(%ctx: !hipsr.context,
                             %in: memref<?x1536xf16, #hipsr.mem<device>>) {
  // expected-remark@+4 {{hipsr-pool-alloc: domain 1: 4 allocs in 2 groups, 2 saved (50%)}}
  // expected-remark@+3 {{hipsr-pool-alloc: domain 1: before vs after: 28% saved}}
  // expected-remark@+2 {{hipsr-pool-alloc: domain 1: group 0: 2 allocs, 33% avg unused}}
  // expected-remark@+1 {{hipsr-pool-alloc: domain 1: group 1: 2 allocs, 25% avg unused}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<?x1536xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x1536xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x1536xf16, #hipsr.mem<device>>
    %a0 = memref.alloc(%d) : memref<?x1536xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    %a3 = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x1536xf16, #hipsr.mem<device>>, memref<?x1536xf16, #hipsr.mem<device>>) outs(%a0 : memref<?x1536xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a0, %a0 : memref<?x1536xf16, #hipsr.mem<device>>, memref<?x1536xf16, #hipsr.mem<device>>) outs(%a1 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a3 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%din : memref<?x1536xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 1 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @dyn_mixed_extents
func.func @dyn_mixed_extents(%ctx: !hipsr.context,
                             %ina: memref<?x512xf16, #hipsr.mem<device>>,
                             %inb: memref<?x256xf16, #hipsr.mem<device>>) {
  // expected-remark@+3 {{hipsr-pool-alloc: domain 2: 2 allocs in 1 groups, 1 saved (50%)}}
  // expected-remark@+2 {{hipsr-pool-alloc: domain 2: before vs after: not comparable (mixed dynamic extents)}}
  // expected-remark@+1 {{hipsr-pool-alloc: domain 2: group 0: 2 allocs, not comparable (mixed dynamic extents)}}
  hipsr.pool_domain(%ctx, %ina, %inb :
      !hipsr.context,
      memref<?x512xf16, #hipsr.mem<device>>,
      memref<?x256xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %da: memref<?x512xf16, #hipsr.mem<device>>,
       %db: memref<?x256xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d0 = memref.dim %da, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %d1 = memref.dim %db, %c0 : memref<?x256xf16, #hipsr.mem<device>>
    %a0 = memref.alloc(%d0) : memref<?x512xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d1) : memref<?x256xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%da, %da : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a0 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a0, %a0 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%da : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%db, %db : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%a1 : memref<?x256xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<?x256xf16, #hipsr.mem<device>>, memref<?x256xf16, #hipsr.mem<device>>) outs(%db : memref<?x256xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 2 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @per_group_extents
func.func @per_group_extents(%ctx: !hipsr.context,
                             %ina: memref<?x1536xf16, #hipsr.mem<device>>,
                             %inb: memref<?x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+4 {{hipsr-pool-alloc: domain 3: 4 allocs in 2 groups, 2 saved (50%)}}
  // expected-remark@+3 {{hipsr-pool-alloc: domain 3: before vs after: not comparable (mixed dynamic extents)}}
  // expected-remark@+2 {{hipsr-pool-alloc: domain 3: group 0: 2 allocs, 33% avg unused}}
  // expected-remark@+1 {{hipsr-pool-alloc: domain 3: group 1: 2 allocs, 25% avg unused}}
  hipsr.pool_domain(%ctx, %ina, %inb :
      !hipsr.context,
      memref<?x1536xf16, #hipsr.mem<device>>,
      memref<?x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %da: memref<?x1536xf16, #hipsr.mem<device>>,
       %db: memref<?x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d0 = memref.dim %da, %c0 : memref<?x1536xf16, #hipsr.mem<device>>
    %d1 = memref.dim %db, %c0 : memref<?x1024xf16, #hipsr.mem<device>>
    %a0 = memref.alloc(%d0) : memref<?x1536xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d1) : memref<?x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc(%d0) : memref<?x512xf16, #hipsr.mem<device>>
    %a3 = memref.alloc(%d1) : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%da, %da : memref<?x1536xf16, #hipsr.mem<device>>, memref<?x1536xf16, #hipsr.mem<device>>) outs(%a0 : memref<?x1536xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a0, %a0 : memref<?x1536xf16, #hipsr.mem<device>>, memref<?x1536xf16, #hipsr.mem<device>>) outs(%a1 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a3 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%da : memref<?x1536xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 3 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @nested_region_user
func.func @nested_region_user(%ctx: !hipsr.context,
                              %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+4 {{hipsr-pool-alloc: domain 4: 2 allocs in 2 groups, 0 saved (0%)}}
  // expected-remark@+3 {{hipsr-pool-alloc: domain 4: before vs after: 12288 bytes vs 12288 bytes, 0% saved}}
  // expected-remark@+2 {{hipsr-pool-alloc: domain 4: group 0: 1 allocs, max 8192 bytes, 0% avg unused}}
  // expected-remark@+1 {{hipsr-pool-alloc: domain 4: group 1: 1 allocs, max 4096 bytes, 0% avg unused}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<2x1024xf16, #hipsr.mem<device>>,
                                    memref<2x1024xf16, #hipsr.mem<device>>)
               outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    scf.execute_region {
      hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
                 outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
      scf.yield
    }
    hipsr.pool_domain_yield
  } {domain_id = 4 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @alloc_inside_region
// CHECK: memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
func.func @alloc_inside_region(%ctx: !hipsr.context,
                               %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  // expected-remark@+3 {{hipsr-pool-alloc: domain 5: 1 allocs in 1 groups, 0 saved (0%)}}
  // expected-remark@+2 {{hipsr-pool-alloc: domain 5: before vs after: 8192 bytes vs 8192 bytes, 0% saved}}
  // expected-remark@+1 {{hipsr-pool-alloc: domain 5: group 0: 1 allocs, max 8192 bytes, 0% avg unused}}
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
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
  } {domain_id = 5 : i64}
  return
}
