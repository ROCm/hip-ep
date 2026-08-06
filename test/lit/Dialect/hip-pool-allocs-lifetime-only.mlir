// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for the default lifetime-only dynamic packing of
// --hip-pool-allocs. Every runtime-sized allocation forms a group; groups with
// disjoint member lifetimes may share a slab across different dynamic-size
// operands. Grouped common-factor packing remains available with
// lifetime-only=false.
//
// Three behaviors, one per function:
//   1. disjoint allocations with different dynamic sizes share a slab;
//   2. overlapping allocations occupy separate slabs;
//   3. an unaligned member shares a common-factor slab whose effective width is
//      already alignment-multiple.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s

//===----------------------------------------------------------------------===//
// 1. Allocations using %d0 and %d1 have disjoint lifetimes. One slab is sized
// to their runtime maximum, and both views use its base.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @disjoint_share_maxui
// CHECK: %[[D0:.*]] = memref.dim
// CHECK: %[[D1:.*]] = memref.dim
// CHECK: %[[A_COEFF:.*]] = arith.constant 32768 : index
// CHECK: %[[A_SIZE:.*]] = arith.muli %[[D0]], %[[A_COEFF]] : index
// CHECK: %[[B_COEFF:.*]] = arith.constant 16384 : index
// CHECK: %[[B_SIZE:.*]] = arith.muli %[[D1]], %[[B_COEFF]] : index
// CHECK: %[[WIDTH:.*]] = arith.maxui %[[A_SIZE]], %[[B_SIZE]] : index
// CHECK: %[[POOL:.*]] = hip.get_pool(%{{.*}}, %[[WIDTH]])
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[D0]]]
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[D1]]]
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

  // %a is dead before %b is allocated.
  %b = memref.alloc(%d1) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)

  return %arg0 : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// 2. Overlapping lifetimes require separate slabs. The pool contribution is the
// sum of both footprints, and the views use distinct offsets.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlapping_stacked
// CHECK: %[[DIM:.*]] = memref.dim
// CHECK: %[[A_COEFF:.*]] = arith.constant 32768 : index
// CHECK: %[[A_SIZE:.*]] = arith.muli %[[DIM]], %[[A_COEFF]] : index
// CHECK: %[[B_COEFF:.*]] = arith.constant 16384 : index
// CHECK: %[[B_SIZE:.*]] = arith.muli %[[DIM]], %[[B_COEFF]] : index
// CHECK: %[[TOTAL:.*]] = arith.addi %[[A_SIZE]], %[[B_SIZE]] : index
// CHECK: %[[POOL:.*]] = hip.get_pool(%{{.*}}, %[[TOTAL]])
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[DIM]]]
// CHECK: memref.view %[[POOL]][%[[A_SIZE]]][%[[DIM]]]
// CHECK-NOT: hip.get_pool
func.func @overlapping_stacked(%ctx: !hip.context,
                               %arg0: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg0, %c0 : memref<?xf32>

  %a = memref.alloc(%d0) : memref<?x8192xf32>
  %b = memref.alloc(%d0) : memref<?x4096xf32>
  // The use of %a after %b is allocated makes their lifetimes overlap.
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  return %arg0 : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// 3. A non-alignment-multiple member shares a slab with a larger,
// alignment-multiple member using the same F. The effective slab coefficient is
// the maximum (32768), so no runtime alignment is required.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @unaligned_member_aligned_effective_width
// CHECK: %[[DIM:.*]] = memref.dim
// CHECK: %[[LARGE_COEFF:.*]] = arith.constant 32768 : index
// CHECK: %[[LARGE_SIZE:.*]] = arith.muli %[[DIM]], %[[LARGE_COEFF]] : index
// CHECK: %[[SMALL_COEFF:.*]] = arith.constant 8 : index
// CHECK: %[[SMALL_SIZE:.*]] = arith.muli %[[DIM]], %[[SMALL_COEFF]] : index
// CHECK: %[[WIDTH:.*]] = arith.maxui %[[LARGE_SIZE]], %[[SMALL_SIZE]] : index
// CHECK-NOT: arith.divui
// CHECK: %[[POOL:.*]] = hip.get_pool(%{{.*}}, %[[WIDTH]])
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[DIM]]]{{.*}}memref<?x8192xf32>
// CHECK: memref.view %[[POOL]][%[[ZERO]]][%[[DIM]]]{{.*}}memref<?x2xf32>
// CHECK-NOT: hip.get_pool
func.func @unaligned_member_aligned_effective_width(
    %ctx: !hip.context, %arg0: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg0, %c0 : memref<?xf32>

  // Alignment-multiple coefficient: 32768.
  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  // Non-alignment-multiple coefficient: 8. Its lifetime is disjoint from %a.
  %s = memref.alloc(%d) : memref<?x2xf32>
  hip.miopen.softmax(%ctx) ins(%s : memref<?x2xf32>)
                           outs(%s : memref<?x2xf32>)

  return %arg0 : memref<?xf32>
}
