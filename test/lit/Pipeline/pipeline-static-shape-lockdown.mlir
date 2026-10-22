// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Lock down the pre-pooling pass order. Host-fed scalars must move to host
// scratch before OptimizeMemRefs considers ordinary allocations for reuse,
// while a static pure-GPU function must not pick up host scratch.
//
// If a future refactor relaxes MaterializeHostScalars's candidate filter
// (e.g. drops the `hasHostIO` requirement or removes the arg-0
// `!hip.context` check) this test fails immediately.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-materialize-host-scalars --hip-optimize-memrefs --hip-promote-strided-operands --hip-pool-allocs %s | FileCheck %s

// ===== Static, pure-GPU: no host I/O at all. =====
//
// Two memref<8x8xf32> allocs flow through PoolAllocs as-is. No
// hip.get_host_scratch is emitted because there is no memref.store /
// memref.load user, so the `hasHostIO` requirement rejects both allocations.
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

// The host-written scalar is replaced before OptimizeMemRefs runs, so it is a
// view of host scratch rather than a reusable memref.alloc slot previously
// occupied by an asynchronous HIP operation.
//
// CHECK-LABEL: func.func @host_scalar_excluded_from_reuse
// CHECK: %[[SCRATCH:.*]] = hip.get_host_scratch
// CHECK: %[[SHAPE:.*]] = memref.view %[[SCRATCH]]
// CHECK: memref.store %{{.*}}, %[[SHAPE]][]
// CHECK: hip.add{{.*}}ins(%[[SHAPE]]
func.func @host_scalar_excluded_from_reuse(
    %ctx: !hip.context, %a: memref<i64>, %b: memref<i64>,
    %host_value: i64) -> memref<i64> {
  %gpu_tmp = memref.alloc() : memref<i64>
  hip.add(%ctx) ins(%a, %b : memref<i64>, memref<i64>)
                outs(%gpu_tmp : memref<i64>)
  %host_scalar = memref.alloc() : memref<i64>
  memref.store %host_value, %host_scalar[] : memref<i64>
  %out = memref.alloc() : memref<i64>
  hip.add(%ctx) ins(%host_scalar, %a : memref<i64>, memref<i64>)
                outs(%out : memref<i64>)
  return %out : memref<i64>
}
