// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Verify the final pool-quality sequence:
//   resolve memref dims -> CSE -> canonicalize -> hoist sizes -> pool allocs.
//
// Repeated dim queries on a late, opaque call result survive memref-dim
// resolution and cannot move above the call. The first two runs isolate CSE
// and canonicalization from hoisting. The last run composes the full sequence.

// RUN: hip-mlir-opt --hip-resolve-memref-dims \
// RUN:   --hip-pool-allocs %s \
// RUN:   | FileCheck %s --check-prefix=NO-CSE
// RUN: hip-mlir-opt --hip-resolve-memref-dims --cse \
// RUN:   --hip-pool-allocs %s \
// RUN:   | FileCheck %s --check-prefix=WITH-CSE
// RUN: hip-mlir-opt --hip-resolve-memref-dims --cse --canonicalize \
// RUN:   --hip-hoist-alloc-size-arith --hip-pool-allocs --verify-each %s \
// RUN:   | FileCheck %s --check-prefix=NORMALIZED

// NO-CSE-LABEL: func.func @dedup_late_dims
// NO-CSE-COUNT-3: hip.get_pool
// NO-CSE-NOT: hip.get_pool

// WITH-CSE-LABEL: func.func @dedup_late_dims
// WITH-CSE-COUNT-2: hip.get_pool
// WITH-CSE-NOT: hip.get_pool

// NORMALIZED-LABEL: func.func @dedup_late_dims
// NORMALIZED-COUNT-2: hip.get_pool
// NORMALIZED-NOT: hip.get_pool

func.func private @late_source(memref<?xf32>) -> memref<?x?xf32>

func.func @dedup_late_dims(
    %ctx: !hip.context, %input: memref<?xf32>, %n: index) -> memref<?xf32> {
  %c0 = arith.constant 0 : index

  %seed = memref.alloc(%n) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%seed : memref<?xf32>)
  %late = func.call @late_source(%seed)
      : (memref<?xf32>) -> memref<?x?xf32>

  %dim0 = memref.dim %late, %c0 : memref<?x?xf32>
  %a = memref.alloc(%dim0) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%a : memref<?xf32>)

  %dim1 = memref.dim %late, %c0 : memref<?x?xf32>
  %b = memref.alloc(%dim1) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?xf32>)
                           outs(%b : memref<?xf32>)
  return %b : memref<?xf32>
}

// CSE makes the broadcast select's arms identical, but does not fold it.
// Without hoisting, its late definition opens a third domain even though
// %a and %b have the same extent and disjoint lifetimes.
// Folding before pooling lets them reuse the same view in the second domain.

// NO-CSE-LABEL: func.func @fold_late_broadcast
// NO-CSE-COUNT-3: hip.get_pool
// NO-CSE-NOT: hip.get_pool

// WITH-CSE-LABEL: func.func @fold_late_broadcast
// WITH-CSE-COUNT-3: hip.get_pool
// WITH-CSE-NOT: hip.get_pool

// NORMALIZED-LABEL: func.func @fold_late_broadcast
// NORMALIZED: hip.get_pool
// NORMALIZED: %[[POOL:.*]] = hip.get_pool
// NORMALIZED: %[[A:.*]] = memref.view %[[POOL]][%[[OFFSET:.*]]][%[[DIM:.*]]] : memref<?xi8> to memref<?xf32>
// NORMALIZED: hip.miopen.softmax
// NORMALIZED: hip.miopen.softmax
// NORMALIZED-NOT: hip.get_pool
// NORMALIZED-NOT: arith.select
// NORMALIZED: %[[B:.*]] = memref.view %[[POOL]][%[[OFFSET]]][%[[DIM]]] : memref<?xi8> to memref<?xf32>
// NORMALIZED: hip.miopen.softmax
// NORMALIZED: return %[[B]]

func.func @fold_late_broadcast(
    %ctx: !hip.context, %input: memref<?xf32>, %n: index) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %seed = memref.alloc(%n) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%seed : memref<?xf32>)
  %late = func.call @late_source(%seed)
      : (memref<?xf32>) -> memref<?x?xf32>
  %dim0 = memref.dim %late, %c0 : memref<?x?xf32>
  %a = memref.alloc(%dim0) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%a : memref<?xf32>)
  hip.miopen.softmax(%ctx) ins(%a : memref<?xf32>)
                           outs(%seed : memref<?xf32>)
  %dim1 = memref.dim %late, %c0 : memref<?x?xf32>
  %is_one = arith.cmpi eq, %dim0, %c1 : index
  %extent = arith.select %is_one, %dim1, %dim0 : index
  %b = memref.alloc(%extent) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%seed : memref<?xf32>)
                           outs(%b : memref<?xf32>)
  return %b : memref<?xf32>
}

// Unequal dynamic extents still require the exact broadcast select. In
// particular, select(lhs == 1, rhs, lhs) must retain the zero/one case.

// NO-CSE-LABEL: func.func @retain_dynamic_broadcast
// NO-CSE-COUNT-3: hip.get_pool
// NO-CSE-NOT: hip.get_pool

// WITH-CSE-LABEL: func.func @retain_dynamic_broadcast
// WITH-CSE-COUNT-3: hip.get_pool
// WITH-CSE-NOT: hip.get_pool

// NORMALIZED-LABEL: func.func @retain_dynamic_broadcast
// NORMALIZED: %[[LHS:.*]] = memref.dim {{.*}}, %c0
// NORMALIZED: %[[RHS:.*]] = memref.dim {{.*}}, %c1
// NORMALIZED: %[[IS_ONE:.*]] = arith.cmpi eq, %[[LHS]], %c1 : index
// NORMALIZED: %[[EXTENT:.*]] = arith.select %[[IS_ONE]], %[[RHS]], %[[LHS]] : index
// NORMALIZED: hip.get_pool{{.*}}domain_id = 1
// NORMALIZED: memref.view {{.*}}[%[[EXTENT]]] : memref<?xi8> to memref<?xf32>

func.func @retain_dynamic_broadcast(
    %ctx: !hip.context, %input: memref<?xf32>, %n: index) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %seed = memref.alloc(%n) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%seed : memref<?xf32>)
  %late = func.call @late_source(%seed)
      : (memref<?xf32>) -> memref<?x?xf32>
  %lhs = memref.dim %late, %c0 : memref<?x?xf32>
  %a = memref.alloc(%lhs) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?xf32>)
                           outs(%a : memref<?xf32>)
  %rhs = memref.dim %late, %c1 : memref<?x?xf32>
  %is_one = arith.cmpi eq, %lhs, %c1 : index
  %extent = arith.select %is_one, %rhs, %lhs : index
  %b = memref.alloc(%extent) : memref<?xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?xf32>)
                           outs(%b : memref<?xf32>)
  return %b : memref<?xf32>
}
