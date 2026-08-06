// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -hipsr-pool-alloc \
// RUN:   | FileCheck %s --implicit-check-not=memref.view

// Cases are ordered by lifetime-interval topology (interference-graph chromatic
// number chi = group count), ascending:
//   chi=1  single point : align_up_rounding, dynamic_size, dead_alloc_skipped
//   chi=1  independent  : coalesce_static
//   chi=2  staggered    : split_two_groups, split_two_groups_dynamic

// 3xf16 = 6 B is not a multiple of 256, so the alignUp chain must round up.
// CHECK-LABEL: func.func @align_up_rounding
// CHECK: %[[C6:.+]] = arith.constant 6 : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[C6]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[G0]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NOT: arith.maxui
func.func @align_up_rounding(%ctx: !hipsr.context,
                        %in: memref<3xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<3xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %d: memref<3xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<3xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%d, %d : memref<3xf16, #hipsr.mem<device>>, memref<3xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<3xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// The dynamic factor is the alloc's %dim operand, not a shape query on the
// buffer itself, which would be cyclic SSA.
// CHECK-LABEL: func.func @dynamic_size
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.+]] = memref.dim %{{.+}}, %[[C0]] : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: memref.alloc(%[[DIM]]) : memref<?x512xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[BYTES:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[BYTES]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[G0]]) : memref<?xi8, #hipsr.mem<device>>
func.func @dynamic_size(%ctx: !hipsr.context,
                   %in: memref<?x512xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<?x512xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<?x512xf16, #hipsr.mem<device>>):
    %c0 = arith.constant 0 : index
    %d = memref.dim %din, %c0 : memref<?x512xf16, #hipsr.mem<device>>
    %a1 = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// An alloc with no DPS write has no interval to place, so it joins no group and
// the size chain covers the live alloc alone (8192, no maxui).
// CHECK-LABEL: func.func @dead_alloc_skipped
// CHECK: memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[LIVE:.+]] = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[C8192:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[C8192]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[G0]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.add(%{{.+}}) ins(%{{.+}}, %{{.+}} : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%[[LIVE]] : memref<4x1024xf16, #hipsr.mem<device>>)
// CHECK-NOT: arith.maxui
func.func @dead_alloc_skipped(%ctx: !hipsr.context,
                        %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %dead = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>)
               outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// Four disjoint allocs of differing sizes coalesce into one group: three
// arith.maxui fold the sizes and the align chain runs once for the whole group
// (the trailing CHECK-NOT rules out a second group).
// CHECK-LABEL: func.func @coalesce_static
// CHECK: %[[C16384:.+]] = arith.constant 16384 : index
// CHECK-NEXT: %[[C8192A:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-NEXT: %[[C8192B:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[M0:.+]] = arith.maxui %[[C16384]], %[[C8192A]] : index
// CHECK-NEXT: %[[M1:.+]] = arith.maxui %[[M0]], %[[C4096]] : index
// CHECK-NEXT: %[[M2:.+]] = arith.maxui %[[M1]], %[[C8192B]] : index
// CHECK-NEXT: %[[C256:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[NUM:.+]] = arith.addi %[[M2]], %[[C255]] : index
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[NUM]], %[[C256]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[DIV]], %[[C256]] : index
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[G0]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NOT: arith.divui
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
    %a1 = memref.alloc() : memref<8x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a3 = memref.alloc() : memref<2x1024xf16, #hipsr.mem<device>>
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
  } {domain_id = 0 : i64}
  return
}

// -----

// a1 is read while writing a2, so their lifetimes overlap and land in separate
// groups: two independent size chains, each aligned on its own.
// CHECK-LABEL: func.func @split_two_groups
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
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NOT: arith.maxui
func.func @split_two_groups(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a1, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @split_two_groups_dynamic
// CHECK: %[[DIM:.+]] = memref.dim %{{.+}} : memref<?x512xf16, #hipsr.mem<device>>
// CHECK: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-NEXT: %[[BYTES:.+]] = arith.muli %[[C1024]], %[[DIM]] : index
// CHECK-NEXT: %[[C256A:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255A:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N0:.+]] = arith.addi %[[BYTES]], %[[C255A]] : index
// CHECK-NEXT: %[[D0:.+]] = arith.divui %[[N0]], %[[C256A]] : index
// CHECK-NEXT: %[[G0:.+]] = arith.muli %[[D0]], %[[C256A]] : index
// CHECK-NEXT: %[[C8192:.+]] = arith.constant 8192 : index
// CHECK-NEXT: %[[C256B:.+]] = arith.constant 256 : index
// CHECK-NEXT: %[[C255B:.+]] = arith.constant 255 : index
// CHECK-NEXT: %[[N1:.+]] = arith.addi %[[C8192]], %[[C255B]] : index
// CHECK-NEXT: %[[D1:.+]] = arith.divui %[[N1]], %[[C256B]] : index
// CHECK-NEXT: %[[G1:.+]] = arith.muli %[[D1]], %[[C256B]] : index
// CHECK-NEXT: %[[POOLSZ:.+]] = arith.addi %[[G0]], %[[G1]] : index
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[POOLSZ]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK-NOT: arith.maxui
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
    %a_dyn = memref.alloc(%d) : memref<?x512xf16, #hipsr.mem<device>>
    %a_static = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%a_dyn : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%dsin, %dsin : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a_static : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_dyn, %a_dyn : memref<?x512xf16, #hipsr.mem<device>>, memref<?x512xf16, #hipsr.mem<device>>) outs(%din : memref<?x512xf16, #hipsr.mem<device>>)
    hipsr.add(%dctx) ins(%a_static, %a_static : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%dsin : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @two_domains
// CHECK: %[[G0:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[G0]]) : memref<?xi8, #hipsr.mem<device>>
// CHECK: %[[G1:.+]] = arith.muli %{{.+}}, %{{.+}} : index
// CHECK-NEXT: hipsr.get_pool(%{{.+}}, %[[G1]]) {domain_id = 7 : i64} : memref<?xi8, #hipsr.mem<device>>
func.func @two_domains(%ctx: !hipsr.context, %in: memref<4x1024xf16, #hipsr.mem<device>>) {
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a1 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a1 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 0 : i64}
  hipsr.pool_domain(%ctx, %in : !hipsr.context, memref<4x1024xf16, #hipsr.mem<device>>) {
  ^bb0(%dctx: !hipsr.context, %din: memref<4x1024xf16, #hipsr.mem<device>>):
    %a2 = memref.alloc() : memref<4x1024xf16, #hipsr.mem<device>>
    hipsr.add(%dctx) ins(%din, %din : memref<4x1024xf16, #hipsr.mem<device>>, memref<4x1024xf16, #hipsr.mem<device>>) outs(%a2 : memref<4x1024xf16, #hipsr.mem<device>>)
    hipsr.pool_domain_yield
  } {domain_id = 7 : i64}
  return
}
