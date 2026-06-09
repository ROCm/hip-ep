// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Lock the multi-domain module-attribute layout emitted by
// --hip-pool-allocs. This case lives in its own file because module
// attributes are PER-MODULE: when multiple functions share a file, only
// the last function's run leaves attributes on the module.
//
// Contract checked here:
//   * `hipdnn.pool_size` / `hipdnn.buffer_count` / `hipdnn.buffer_offsets`
//     are still emitted (legacy ABI, consumed by GenerateInterface today).
//   * `hipdnn.pool_size` equals `hipdnn.pool_sizes[0]` for backward compat.
//   * `hipdnn.domain_count`, `hipdnn.pool_sizes`, `hipdnn.buffer_domains`
//     are emitted only when `domain_count > 1` (the multi-domain runtime ABI).
//   * `hipdnn.buffer_domains` is parallel to `hipdnn.buffer_offsets` and
//     records which pool each pooled alloc came from, in textual order.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s

// CHECK: hipdnn.buffer_count = 2
// CHECK-SAME: hipdnn.buffer_domains = [0, 1]
// CHECK-SAME: hipdnn.buffer_offsets = [0, 0]
// CHECK-SAME: hipdnn.domain_count = 2
// CHECK-SAME: hipdnn.pool_size = 0
// CHECK-SAME: hipdnn.pool_sizes = [0, 0]

func.func @multi_domain_metadata(
    %ctx: !hip.context,
    %x: memref<?x16xf32>,
    %scratch: memref<2xindex>) -> memref<?x16xf32> {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %x, %c0 : memref<?x16xf32>
  %alloc0 = memref.alloc(%d0) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%x : memref<?x16xf32>) outs(%alloc0 : memref<?x16xf32>)
  %d1 = memref.load %scratch[%c0] : memref<2xindex>
  %alloc1 = memref.alloc(%d1) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x16xf32>) outs(%alloc1 : memref<?x16xf32>)
  return %alloc1 : memref<?x16xf32>
}
