// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for the default (lifetime-only) dynamic packing of
// --hip-pool-allocs: every runtime-sized alloc is its own group, and any two
// lifetime-disjoint allocs share one pool slab (width = maxui of footprints)
// regardless of their dynamic dimensions.
//
// Three behaviors, one per function:
//   1. disjoint allocs with DIFFERENT dynamic dims share a slab (maxui);
//   2. OVERLAPPING allocs never share (stacked, sum, no maxui) -- the safety
//      guardrail;
//   3. a small (non-alignment-multiple) alloc rides the same path and shares an
//      aligned alloc's slab, with the slab width rounded up (divui) for it.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s

//===----------------------------------------------------------------------===//
// 1. Different dynamic dims (%d0 vs %d1 -> different F), disjoint lifetimes.
// One pool, sized to the larger footprint (maxui, not sum); both views share
// one base offset because the allocs are never live together.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @disjoint_share_maxui
// CHECK: arith.maxui
// CHECK: %[[POOL:[0-9a-z_]+]] = hip.get_pool
// CHECK: memref.view %[[POOL]][%[[OFF:[0-9a-z_]+]]]
// CHECK: memref.view %[[POOL]][%[[OFF]]]
// CHECK-NOT: hip.get_pool
func.func @disjoint_share_maxui(%ctx: !hip.context,
                                %arg0: memref<?xf32>,
                                %arg1: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg0, %c0 : memref<?xf32>
  %d1 = memref.dim %arg1, %c0 : memref<?xf32>

  %a = memref.alloc(%d0) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  // Allocated after %a is dead -> disjoint lifetime, different dynamic dim.
  %b = memref.alloc(%d1) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)

  return %arg0 : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// 2. Overlapping lifetimes -> separate slabs. The pool is the SUM of the two
// footprints (addi) and never a maxui: overlapping allocs must not alias.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlapping_stacked
// CHECK-NOT: arith.maxui
// CHECK: hip.get_pool
// CHECK: memref.view
// CHECK: memref.view
func.func @overlapping_stacked(%ctx: !hip.context,
                               %arg0: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg0, %c0 : memref<?xf32>

  %a = memref.alloc(%d0) : memref<?x8192xf32>
  %b = memref.alloc(%d0) : memref<?x4096xf32>
  // %b lives, then %a is used again -> the two lifetimes overlap.
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  return %arg0 : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// 3. A small (non-alignment-multiple) alloc is packed through the same lifetime
// path -- there is no separate small-bucket scheme in the default. Disjoint
// from an aligned alloc, it shares the slab (maxui), and the slab width is
// rounded up (alignUp -> divui) so the pool stays aligned for the small member.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @small_alloc_shares_slab
// CHECK: arith.maxui
// CHECK: arith.divui
// CHECK: %[[POOL:[0-9a-z_]+]] = hip.get_pool
// CHECK: memref.view %[[POOL]][%[[OFF:[0-9a-z_]+]]]
// CHECK: memref.view %[[POOL]][%[[OFF]]]
// CHECK-NOT: hip.get_pool
func.func @small_alloc_shares_slab(%ctx: !hip.context,
                                   %arg0: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg0, %c0 : memref<?xf32>

  // Aligned: staticFactor 32768 (multiple of 256).
  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  // Small: staticFactor 8 (not a multiple of 256). Disjoint from %a.
  %s = memref.alloc(%d) : memref<?x2xf32>
  hip.miopen.softmax(%ctx) ins(%s : memref<?x2xf32>)
                           outs(%s : memref<?x2xf32>)

  return %arg0 : memref<?xf32>
}
