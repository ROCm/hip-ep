// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hip-pool-allocs --verify-each | FileCheck %s --check-prefix=BEFORE
// RUN: hip-mlir-opt %s --hip-hoist-alloc-size-arith --hip-pool-allocs --canonicalize --verify-each | FileCheck %s --check-prefix=AFTER
//
// The late read cannot cross the seed allocation. Its pure division can cross
// the working allocations, making their domain available to the final buffers.
// Both working buffers are larger than the final buffers for positive n.

// BEFORE-LABEL: func.func @late_size_reuse
// BEFORE-COUNT-3: hip.get_pool
// BEFORE-NOT: hip.get_pool
// AFTER-LABEL: func.func @late_size_reuse
// AFTER: hip.get_pool
// AFTER: %[[N:.*]] = call @late_extent
// AFTER: %[[QUARTER:.*]] = arith.divsi %[[N]], %{{.*}}
// AFTER: %[[POOL:.*]] = hip.get_pool
// AFTER: memref.view %[[POOL]][%[[FIRST:.*]]][%[[N]]] : memref<?xi8> to memref<?x4304xf16>
// AFTER: memref.view %[[POOL]][%[[SECOND:.*]]][%[[N]]] : memref<?xi8> to memref<?x4304xf16>
// AFTER: memref.view %[[POOL]][%[[FIRST]]][%[[QUARTER]]] : memref<?xi8> to memref<?x4608xf16>
// AFTER: memref.view %[[POOL]][%[[SECOND]]][%[[QUARTER]]] : memref<?xi8> to memref<?x4608xf16>
// AFTER-NOT: hip.get_pool
// AFTER: return

func.func private @late_extent(memref<?xf16>) -> index

func.func @late_size_reuse(
    %ctx: !hip.context, %n: index, %seed_input: memref<?xf16>,
    %input: memref<?x4304xf16>, %output: memref<?x4304xf16>,
    %tail_input: memref<?x4608xf16>, %tail_output: memref<?x4608xf16>) {
  %c4 = arith.constant 4 : index
  %seed = memref.alloc(%n) : memref<?xf16>
  hip.miopen.softmax(%ctx) ins(%seed_input : memref<?xf16>) outs(%seed : memref<?xf16>)
  %late = call @late_extent(%seed) : (memref<?xf16>) -> index
  %a = memref.alloc(%late) : memref<?x4304xf16>
  %b = memref.alloc(%late) : memref<?x4304xf16>
  hip.miopen.softmax(%ctx) ins(%input : memref<?x4304xf16>) outs(%a : memref<?x4304xf16>)
  hip.miopen.softmax(%ctx) ins(%a : memref<?x4304xf16>) outs(%b : memref<?x4304xf16>)
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4304xf16>) outs(%output : memref<?x4304xf16>)
  %quarter = arith.divsi %late, %c4 : index
  %c = memref.alloc(%quarter) : memref<?x4608xf16>
  %d = memref.alloc(%quarter) : memref<?x4608xf16>
  hip.miopen.softmax(%ctx) ins(%tail_input : memref<?x4608xf16>) outs(%c : memref<?x4608xf16>)
  hip.miopen.softmax(%ctx) ins(%c : memref<?x4608xf16>) outs(%d : memref<?x4608xf16>)
  hip.miopen.softmax(%ctx) ins(%d : memref<?x4608xf16>) outs(%tail_output : memref<?x4608xf16>)
  return
}
