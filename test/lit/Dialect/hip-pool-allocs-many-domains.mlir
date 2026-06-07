// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for the unbounded multi-domain path of --hip-pool-allocs.
//
// There is NO compile-time cap on the dominance-domain count: the runtime
// grows its per-domain pool arrays on demand, so a function that partitions
// into many domains must compile successfully and emit one hip.get_pool per
// domain plus the multi-domain metadata. Above an advisory threshold (8) the
// pass emits a NON-FATAL remark suggesting that upstream hoisting may be
// missing — compilation continues regardless.
//
// In practice a well-canonicalised post-hoist graph collapses to 1 or 2
// domains; the cascading chains below deliberately defeat merging to exercise
// the high-domain-count path.
//===----------------------------------------------------------------------===//

// Functional check: the pass succeeds and tags the module with a 9-domain
// metadata blob (one get_pool anchor per domain).
// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s

// Diagnostic check: the advisory remark fires (and the pass still succeeds).
// RUN: hip-mlir-opt --hip-pool-allocs --verify-diagnostics %s

// Module attributes (incl. the multi-domain metadata) print before the body.
// CHECK: hipdnn.domain_count = 9
// CHECK-LABEL: func.func @many_domains
// CHECK-COUNT-9: hip.get_pool

// 9 cascading load+alloc chains -> 9 domains. Each new load is non-
// speculatable and is positioned below all previous allocs, so partition
// cannot merge it into any existing domain. The 9th chain pushes the domain
// count past the advisory threshold and triggers the remark below.
// expected-remark@+1 {{hip-pool-allocs: dominance partition produced 9 domains}}
func.func @many_domains(
    %ctx: !hip.context,
    %x: memref<?x16xf32>,
    %scratch: memref<16xindex>) -> memref<?x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %d0 = memref.dim %x, %c0 : memref<?x16xf32>
  %a0 = memref.alloc(%d0) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%x : memref<?x16xf32>) outs(%a0 : memref<?x16xf32>)
  %d1 = memref.load %scratch[%c0] : memref<16xindex>
  %a1 = memref.alloc(%d1) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a0 : memref<?x16xf32>) outs(%a1 : memref<?x16xf32>)
  %d2 = memref.load %scratch[%c1] : memref<16xindex>
  %a2 = memref.alloc(%d2) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a1 : memref<?x16xf32>) outs(%a2 : memref<?x16xf32>)
  %d3 = memref.load %scratch[%c2] : memref<16xindex>
  %a3 = memref.alloc(%d3) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a2 : memref<?x16xf32>) outs(%a3 : memref<?x16xf32>)
  %d4 = memref.load %scratch[%c3] : memref<16xindex>
  %a4 = memref.alloc(%d4) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a3 : memref<?x16xf32>) outs(%a4 : memref<?x16xf32>)
  %d5 = memref.load %scratch[%c4] : memref<16xindex>
  %a5 = memref.alloc(%d5) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a4 : memref<?x16xf32>) outs(%a5 : memref<?x16xf32>)
  %d6 = memref.load %scratch[%c5] : memref<16xindex>
  %a6 = memref.alloc(%d6) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a5 : memref<?x16xf32>) outs(%a6 : memref<?x16xf32>)
  %d7 = memref.load %scratch[%c6] : memref<16xindex>
  %a7 = memref.alloc(%d7) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a6 : memref<?x16xf32>) outs(%a7 : memref<?x16xf32>)
  %d8 = memref.load %scratch[%c7] : memref<16xindex>
  %a8 = memref.alloc(%d8) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%a7 : memref<?x16xf32>) outs(%a8 : memref<?x16xf32>)
  return %a8 : memref<?x16xf32>
}
