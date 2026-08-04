// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for grouped dynamic packing in --hip-pool-allocs
// (lifetime-only=false).
//
// Grouped mode best-fit-packs alignment-multiple allocations with
// SSA-identical ordered dynamic-size operands. The second case verifies that
// two overlapping smaller allocations occupy distinct ranges within storage
// previously used by a larger, lifetime-disjoint allocation.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs='lifetime-only=false' %s | FileCheck %s

// CHECK-LABEL: func.func @cross_width_share
// The larger coefficient sizes the group, and both views use its base.
// CHECK: %[[DIM:.*]] = memref.dim
// CHECK: %[[COEFF:.*]] = arith.constant 32768 : index
// CHECK: %[[SIZE:.*]] = arith.muli %[[DIM]], %[[COEFF]] : index
// CHECK: %[[POOL:.*]] = hip.get_pool(%{{.*}}, %[[SIZE]])
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[DIM]]]{{.*}}memref<?x8192xf32>
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[DIM]]]{{.*}}memref<?x4096xf32>
// CHECK-NOT: hip.get_pool

func.func @cross_width_share(%ctx: !hip.context,
                             %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  // Byte coefficient 32768; live only across this operation.
  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  // Byte coefficient 16384; allocated after %a is dead.
  %b = memref.alloc(%d) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)

  return %arg : memref<?xf32>
}

// CHECK-LABEL: func.func @same_factor_gap_packing
// A, B, and C share %d. A is disjoint from both smaller allocs, while B and C
// overlap each other. Best-fit places B and C in opposite halves of A's
// reservation, keeping the span coefficient at 32768 instead of 49152.
// CHECK: %[[DIM:.*]] = memref.dim
// CHECK: %[[SPAN:.*]] = arith.constant 32768 : index
// CHECK: %[[SIZE:.*]] = arith.muli %[[DIM]], %[[SPAN]] : index
// CHECK: %[[POOL:[0-9a-z_]+]] = hip.get_pool({{.*}}%[[SIZE]])
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[HALF_COEFF:.*]] = arith.constant 16384 : index
// CHECK: %[[HALF:.*]] = arith.muli %[[DIM]], %[[HALF_COEFF]] : index
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[DIM]]]{{.*}}memref<?x8192xf32>
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[DIM]]]{{.*}}memref<?x4096xf32>
// CHECK: memref.view %[[POOL]][%[[HALF]]][%[[DIM]]]{{.*}}memref<?x4096xf32>
func.func @same_factor_gap_packing(
    %ctx: !hip.context, %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  // Byte coefficient 32768; dead before %b and %c are allocated.
  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  // Byte coefficient 16384 for each allocation; their lifetimes overlap.
  %b = memref.alloc(%d) : memref<?x4096xf32>
  %c = memref.alloc(%d) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)
  hip.miopen.softmax(%ctx) ins(%c : memref<?x4096xf32>)
                           outs(%c : memref<?x4096xf32>)

  return %arg : memref<?xf32>
}
