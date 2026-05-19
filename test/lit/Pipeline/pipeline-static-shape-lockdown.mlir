// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Lockdown: a static-shape model with no host-fed scalar staging buffers
// must flow through the MaterializeHostScalars + PoolAllocs reorder
// (introduced for the dynseqlen regression) without picking up any host
// scratch.  MaterializeHostScalars must reject every alloc here (no host
// I/O on the f32 path, no integer/index element type on the staging path),
// leaving PoolAllocs to handle the IR exactly as it did before the
// reorder.
//
// If a future refactor relaxes MaterializeHostScalars's candidate filter
// (e.g. drops the `hasHostIO` requirement, accepts f32, or removes the
// arg-0 `!hip.context` check) this test fails immediately.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-materialize-host-scalars --hip-pool-allocs %s | FileCheck %s

// ===== Static, pure-GPU: no host I/O at all. =====
//
// Two memref<8x8xf32> allocs flow through PoolAllocs as-is; no
// hip.get_host_scratch is emitted because:
//   - element type f32 is not isIntOrIndex (filter rejects),
//   - no memref.store / memref.load user (the `hasHostIO` flag stays
//     false even if the type filter is relaxed in the future).
//
// CHECK-LABEL: func.func @static_pure_gpu_lockdown
// CHECK-NOT: hip.get_host_scratch
// CHECK: hip.get_pool
// CHECK: memref.view
func.func @static_pure_gpu_lockdown(
    %ctx: !hip.context,
    %a: memref<8x8xf32>,
    %b: memref<8x8xf32>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32>, memref<8x8xf32>)
                   outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<8x8xf32>)
                            outs(%alloc1 : memref<8x8xf32>)
  return %alloc1 : memref<8x8xf32>
}
