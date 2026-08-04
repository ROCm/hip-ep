// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file --verify-diagnostics \
// RUN:   -hipsr-pool-alloc='emit-pool-report=true' \
// RUN:   | FileCheck %s --implicit-check-not=hipsr.get_pool \
// RUN:       --implicit-check-not=memref.view

// CHECK-LABEL: func.func @interleaved_allocs
func.func @interleaved_allocs(%ctx: !hipsr.context,
                              %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in
      : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [1,3] group 0 size 8192}}
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>,
                                      memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [3,5] group 1 size 8192}}
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%a1, %a1 : memref<4x1024xf16, #hipsr.mem<device>>,
                                    memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [5,7] group 0 size 8192}}
    %a3 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>,
                                    memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a3 : memref<4x1024xf16, #hipsr.mem<device>>)
    // No DPS write, so no live range: a remark here would be unexpected and
    // fail the run.
    %unwritten = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%a3, %a3 : memref<4x1024xf16, #hipsr.mem<device>>,
                                    memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// Hoisted allocs are the form materialize-init-tensors produces. Ranges start
// at the DPS write, so a1 and a3 stay disjoint; indexing the allocs themselves
// would overlap all three and defeat reuse.
// CHECK-LABEL: func.func @hoisted_allocs
func.func @hoisted_allocs(%ctx: !hipsr.context,
                          %in: memref<4x1024xf16, #hipsr.mem<device>>) {
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
  }
  return
}

// -----

// A user inside a nested region extends the range to the enclosing
// block-level op (index 2), not to the nested op's own index.
// CHECK-LABEL: func.func @nested_region_user
func.func @nested_region_user(%ctx: !hipsr.context,
                              %in: memref<4x1024xf16, #hipsr.mem<device>>) {
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
  }
  return
}

// -----

// CHECK-LABEL: func.func @coalesce_static
func.func @coalesce_static(%ctx: !hipsr.context,
                               %in8: memref<8x1024xf16, #hipsr.mem<device>>,
                               %in4a: memref<4x1024xf16, #hipsr.mem<device>>,
                               %in2: memref<2x1024xf16, #hipsr.mem<device>>,
                               %in4b: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in8, %in4a, %in2, %in4b :
      !hipsr.context,
      memref<8x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>,
      memref<2x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %d8: memref<8x1024xf16, #hipsr.mem<device>>,
       %d4a: memref<4x1024xf16, #hipsr.mem<device>>,
       %d2: memref<2x1024xf16, #hipsr.mem<device>>,
       %d4b: memref<4x1024xf16, #hipsr.mem<device>>):
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [4,5] group 0 size 16384}}
    %a1 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [6,7] group 0 size 8192}}
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [8,9] group 0 size 4096}}
    %a3 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [10,11] group 0 size 8192}}
    %a4 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d8, %d8 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a1 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>) outs(%d8 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d4a, %d4a : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%d4a : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d2, %d2 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a3 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>) outs(%d2 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d4b, %d4b : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a4 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a4, %a4 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%d4b : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @split_three_groups
func.func @split_three_groups(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [3,8] group 0 size 8192}}
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [4,7] group 1 size 8192}}
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [5,8] group 2 size 8192}}
    %a3 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %a3 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %a3 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @dynamic_size
func.func @dynamic_size(%ctx: !hipsr.context,
                        %in: memref<?x512xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x512xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x512xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [3,3] group 0 size 1024 x 1 dyn}}
    %a1 = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @coalesce_mixed
func.func @coalesce_mixed(%ctx: !hipsr.context,
                          %inf32: memref<4x256xf32, #hipsr.mem<device>>,
                          %inf16: memref<?x512xf16, #hipsr.mem<device>>) {
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
  }
  return
}

// -----

// CHECK-LABEL: func.func @coalesce_dynamic
func.func @coalesce_dynamic(%ctx: !hipsr.context, %in: memref<?x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [4,5] group 0 size 2048 x 1 dyn}}
    %a1 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [6,7] group 0 size 2048 x 1 dyn}}
    %a2 = memref.alloc(%d) : memref<?x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%din : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %din : memref<?x1024xf16, #hipsr.mem<device>>, memref<?x1024xf16, #hipsr.mem<device>>) outs(%din : memref<?x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @split_two_groups_dynamic
func.func @split_two_groups_dynamic(%ctx: !hipsr.context,
                                    %inf16: memref<?x512xf16, #hipsr.mem<device>>,
                                    %sin: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %inf16, %sin :
      !hipsr.context,
      memref<?x512xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %din: memref<?x512xf16, #hipsr.mem<device>>,
       %dsin: memref<4x1024xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [4,6] group 0 size 1024 x 1 dyn}}
    %a_dyn = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    // expected-remark@+1 {{hipsr-pool-alloc: lifetime [5,7] group 1 size 8192}}
    %a_static = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a_dyn : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%dsin, %dsin : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a_static : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_dyn, %a_dyn : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%din : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_static, %a_static : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%dsin : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}
