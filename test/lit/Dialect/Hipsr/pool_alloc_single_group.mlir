// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-pool-alloc | FileCheck %s

// Three sequential allocs of different sizes with disjoint lifetimes collapse
// into one hipsr.get_pool; each alloc becomes a memref.view at offset 0 with
// pool size = alignUp(max size, 256).
// CHECK-LABEL: func.func @pool_diff_sizes
// CHECK: %[[POOL:.+]] = hipsr.get_pool
// CHECK: %[[OFF:.+]] = arith.constant 0 : index
// CHECK: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<8x1024xf16, #hipsr.mem<device>>
// CHECK: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<2x1024xf16, #hipsr.mem<device>>
// CHECK-NOT: memref.alloc
func.func @pool_diff_sizes(
    %ctx: !hipsr.context,
    %in8: memref<8x1024xf16, #hipsr.mem<device>>,
    %in4: memref<4x1024xf16, #hipsr.mem<device>>,
    %in2: memref<2x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in8, %in4, %in2 :
      !hipsr.context,
      memref<8x1024xf16, #hipsr.mem<device>>,
      memref<4x1024xf16, #hipsr.mem<device>>,
      memref<2x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context,
       %d8: memref<8x1024xf16, #hipsr.mem<device>>,
       %d4: memref<4x1024xf16, #hipsr.mem<device>>,
       %d2: memref<2x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d8, %d8 : memref<8x1024xf16, #hipsr.mem<device>>, memref<8x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<8x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d4, %d4 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d2, %d2 : memref<2x1024xf16, #hipsr.mem<device>>, memref<2x1024xf16, #hipsr.mem<device>>)
               outs(%a3 : memref<2x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// Homogeneous sequential allocs also collapse into one pool.
// CHECK-LABEL: func.func @pool_homogeneous
// CHECK: %[[POOL:.+]] = hipsr.get_pool
// CHECK: %[[OFF:.+]] = arith.constant 0 : index
// CHECK: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK: memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NOT: memref.alloc
func.func @pool_homogeneous(
    %ctx: !hipsr.context,
    %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in :
      !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %d: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d, %d : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%d, %d : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}
