// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for the fallback dynamic packing of --hip-pool-allocs
// (lifetime-only=false).
//
// Two dynamic allocs with the SAME dynamic dim (%d) but DIFFERENT channel width
// (8192 vs 4096 -> different, both 256-aligned, staticFactor) and DISJOINT
// lifetimes. The fallback groups by dynamic dim and best-fit-packs the integer
// staticFactors within the group, so the two allocs share one slab (both at
// offset 0), sized to the larger -- exercising the greedy best-fit path rather
// than the cross-group lifetime coloring that the default relies on.
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
