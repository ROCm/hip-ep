// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test for the debug-only fragmentation probe of --hip-pool-allocs
// (emit-fragmentation-report option).
//
// The probe compares each part of a domain's packed footprint to its max-load
// lower bound (the peak sum of concurrently-live sizes) and reports the surplus
// as fragmentation. It emits NO IR and changes no offsets: RUN line 1 asserts
// the report; RUN line 2 asserts codegen is a single pool either way.
//
// Both functions below are shaped so best-fit reaches the floor (0 frag),
// which is the expected verdict on well-formed graphs. A positive-frag case
// would require a pathological many-interval input that defeats best-fit and is
// not worth encoding here; the max-load helper's arithmetic is exercised by the
// exact numbers below.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs='lifetime-only=true emit-fragmentation-report=true' \
// RUN:   --verify-diagnostics %s
// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Static allocs: A (512 B) spans the whole function; B and C (256 B each) have
// disjoint lifetimes, so C reuses B's slot. Peak concurrent = A+B = 768 B, and
// best-fit also lands at 768 B -> 0 fragmentation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_static
// CHECK: hip.get_pool
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 768/768 B (0 frag); dyn-pool 0/0 units (0 frag); small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_static(%ctx: !hip.context,
                              %arg: memref<1x128xf32>) -> memref<1x128xf32> {
  %a = memref.alloc() : memref<1x128xf32>
  %b = memref.alloc() : memref<1x64xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<1x128xf32>)
                           outs(%a : memref<1x128xf32>)
  hip.miopen.softmax(%ctx) ins(%b : memref<1x64xf32>)
                           outs(%b : memref<1x64xf32>)
  // B's last use -> B dead before C is allocated.
  hip.miopen.softmax(%ctx) ins(%b : memref<1x64xf32>)
                           outs(%b : memref<1x64xf32>)

  %c = memref.alloc() : memref<1x64xf32>
  hip.miopen.softmax(%ctx) ins(%c : memref<1x64xf32>)
                           outs(%c : memref<1x64xf32>)
  // A's last use -> A spans B's and C's lifetimes.
  hip.miopen.softmax(%ctx) ins(%a : memref<1x128xf32>)
                           outs(%a : memref<1x128xf32>)
  return %arg : memref<1x128xf32>
}

//===----------------------------------------------------------------------===//
// Two dynamic allocs, same dynamic dim %d, different static width (staticFactor
// 32768 vs 16384) with disjoint lifetimes. The two allocs share ONE slab sized
// to the max, so the emitted dyn-pool coefficient is max(32768, 16384) = 32768
// units (x F). The max-load lower bound over the disjoint pair is also 32768,
// so best-fit reaches the floor -> 0 fragmentation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_dyn
// CHECK: hip.get_pool
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 0/0 B (0 frag); dyn-pool 32768/32768 units (0 frag); small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_dyn(%ctx: !hip.context,
                           %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  %b = memref.alloc(%d) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)
  return %arg : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// A non-alignment-multiple dynamic slab is rounded up in byte space at runtime.
// Its footprint cannot be represented as a staticFactor coefficient times F,
// so the report must not claim zero coefficient fragmentation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_runtime_aligned
// CHECK: hip.get_pool
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 0/0 B (0 frag); dyn-pool runtime-aligned; small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_runtime_aligned(
    %ctx: !hip.context, %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  // staticFactor = 8, so the emitted slab is alignUp(8 * %d, 256).
  %small = memref.alloc(%d) : memref<?x2xf32>
  hip.miopen.softmax(%ctx) ins(%small : memref<?x2xf32>)
                           outs(%small : memref<?x2xf32>)
  return %arg : memref<?xf32>
}
