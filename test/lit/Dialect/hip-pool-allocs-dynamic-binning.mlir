// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for the fallback dynamic packing of --hip-pool-allocs
// (lifetime-only=false).
//
// The grouped mode best-fit-packs allocs sharing a dynamic factor before
// applying cross-group lifetime coloring. Besides the basic two-width sharing
// case, the second test covers the gap-packing advantage retained by this mode:
// two overlapping small allocs can occupy opposite halves of a larger,
// lifetime-disjoint alloc's reservation.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs='lifetime-only=false' %s | FileCheck %s

// CHECK-LABEL: func.func @cross_width_share
// One pool (single domain), sized to the LARGER width, and both views share the
// same base offset -> they reuse one slab.
// CHECK: %[[POOL:[0-9a-z_]+]] = hip.get_pool
// CHECK: memref.view %[[POOL]][%[[OFF:[0-9a-z_]+]]]
// CHECK: memref.view %[[POOL]][%[[OFF]]]
// CHECK-NOT: hip.get_pool

func.func @cross_width_share(%ctx: !hip.context,
                             %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  // A: [?, 8192]f32 -> staticFactor 32768. Live only across its softmax.
  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  // B: [?, 4096]f32 -> staticFactor 16384. Allocated after A is dead ->
  // disjoint lifetime -> may reuse A's space.
  %b = memref.alloc(%d) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)

  return %arg : memref<?xf32>
}

// CHECK-LABEL: func.func @same_factor_gap_packing
// A, B, and C share %d. A is disjoint from both smaller allocs, while B and C
// overlap each other. Best-fit places B and C in opposite halves of A's
// reservation, keeping the span at 32768 units instead of 49152.
// CHECK-NOT: arith.constant 49152 : index
// CHECK: %[[SPAN:[0-9a-z_]+]] = arith.constant 32768 : index
// CHECK: %[[SIZE:[0-9a-z_]+]] = arith.muli %{{.*}}, %[[SPAN]] : index
// CHECK: %[[POOL:[0-9a-z_]+]] = hip.get_pool({{.*}}%[[SIZE]])
// CHECK: memref.view %[[POOL]][%[[OFF:[0-9a-z_]+]]]
// CHECK: memref.view %[[POOL]][%[[OFF]]]
// CHECK: memref.view %[[POOL]]
func.func @same_factor_gap_packing(
    %ctx: !hip.context, %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  // A: 32768 units, dead before B and C are allocated.
  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  // B and C: 16384 units each, with overlapping lifetimes.
  %b = memref.alloc(%d) : memref<?x4096xf32>
  %c = memref.alloc(%d) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)
  hip.miopen.softmax(%ctx) ins(%c : memref<?x4096xf32>)
                           outs(%c : memref<?x4096xf32>)

  return %arg : memref<?xf32>
}
