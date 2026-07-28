// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-pool-alloc | FileCheck %s

// a1 is read while writing a2, so their lifetimes overlap and land in separate
// groups: two aligned sizes, poolSize = g0 + g1, and off1 = off0 + g0.
// CHECK-LABEL: func.func @two_groups
// CHECK: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[C8192A]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192B]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.addi %[[OFF0]], %[[G0]] : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @two_groups(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// a1 and a2 have disjoint lifetimes so they reuse one group (shared off0), while
// a3 overlaps both and takes its own group at off1: intra-group offset reuse
// plus cross-group separation in a single domain.
// CHECK-LABEL: func.func @group_reuse
// CHECK: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[MAX:.+]] = arith.maxui %[[C8192A]], %[[C8192B]] : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[MAX]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192C:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192C]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.addi %[[OFF0]], %[[G0]] : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V2:.+]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V2]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V2]], %[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @group_reuse(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a3 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a3, %a2 : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}

// -----

// a1, a2, a3 overlap pairwise, forcing three distinct groups: poolSize chains two
// arith.addi (g0+g1+g2) and offsets chain off0 / off1 / off2.
// CHECK-LABEL: func.func @three_groups
// CHECK: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[C8192A]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192B]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[C8192C:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256C:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255C:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N2:.+]] = arith.addi %[[C8192C]], %[[C255C]] : index
// CHECK-NEXT: %[[D2:.+]] = arith.divui %[[N2]], %[[C256C]] : index
// CHECK-NEXT: %[[G2:.+]] = arith.muli %[[D2]], %[[C256C]] : index
// CHECK-NEXT: %[[SUM0:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[SUM0]], %[[G2]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFF1:.+]] = arith.addi %[[OFF0]], %[[G0]] : index
// CHECK-NEXT: %[[OFF2:.+]] = arith.addi %[[OFF1]], %[[G1]] : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V2:.+]] = memref.view %[[POOL]][%[[OFF2]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V2]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %[[V2]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %[[V2]] : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @three_groups(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
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

// Two same-size disjoint allocs coalesce into one group (no size-bucketing):
// poolSize is the lone aligned size, off is a single constant 0, and both views
// share it. The CHECK-NEXT chain (muli feeding get_pool, constant-0 offset
// immediately followed by the views) proves there is no second group summation.
// CHECK-LABEL: func.func @no_bucketing
// CHECK: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[MAX:.+]] = arith.maxui %[[C8192A]], %[[C8192B]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[MAX]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[SZ:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: %[[POOL:.+]] = hipsr.get_pool(%{{.+}}, %[[SZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: %[[OFF:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V0:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[V1:.+]] = memref.view %[[POOL]][%[[OFF]]][] : memref<?xi8, #hipsr.mem<device>> to memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V0]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V0]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[V1]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%[[V1]], %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: memref.alloc
func.func @no_bucketing(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a2, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%din : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  }
  return
}
