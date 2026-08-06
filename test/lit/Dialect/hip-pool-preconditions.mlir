// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Verify the final pool-quality sequence:
//   resolve memref dims -> CSE -> hoist alloc sizes -> pool allocs.
//
// Repeated dim queries on a late, opaque call result survive memref-dim
// resolution and cannot move above the call. Without CSE, each query appears
// after the preceding allocation and opens another domain. CSE makes both later
// allocations use the first query, so they share one late domain.

// RUN: hip-mlir-opt --hip-resolve-memref-dims \
// RUN:   --hip-hoist-alloc-size-arith --hip-pool-allocs %s \
// RUN:   | FileCheck %s --check-prefix=NO-CSE
// RUN: hip-mlir-opt --hip-resolve-memref-dims --cse \
// RUN:   --hip-hoist-alloc-size-arith --hip-pool-allocs %s \
// RUN:   | FileCheck %s --check-prefix=WITH-CSE

// NO-CSE-LABEL: func.func @dedup_late_dims
// NO-CSE-COUNT-3: hip.get_pool
// NO-CSE-NOT: hip.get_pool

// WITH-CSE-LABEL: func.func @dedup_late_dims
// WITH-CSE-COUNT-2: hip.get_pool
// WITH-CSE-NOT: hip.get_pool

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
