// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-pool-allocs (memory pooling into i8 byte buffer).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s
// RUN: hip-mlir-opt --hip-pool-allocs='alignment=64' %s | FileCheck %s --check-prefix=ALIGN64
// RUN: not hip-mlir-opt --hip-pool-allocs='alignment=0' %s 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
// RUN: not hip-mlir-opt --hip-pool-allocs='alignment=3' %s 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
// RUN: not hip-mlir-opt --hip-pool-allocs='alignment=-1' %s 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
// BAD-ALIGN: alignment must be a positive power of 2
//
// The pass uses dominator-aware emission: per-bucket size arithmetic and
// the pool acquisition are placed at the latest legal point (after every
// dyn-operand def, before every pooled alloc). No code motion. The cases
// near the end of this file lock down each pattern that the prior
// hoist-based design needed dedicated helpers for (foldDimOfReshape,
// resolveDimAtSource cast/view/subview/reshape extensions, and the
// host-scratch scalar-load chain).

// ===== Static pooling: two non-overlapping f32 allocs =====
//
// Two memref<8x8xf32> (256 bytes each). Both are live at different times.
// With 256-byte alignment, they pack at offsets 0 and 256 -> pool needs 512.
//
// CHECK-LABEL: func.func @static_two_allocs
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         %[[SIZE:.*]] = arith.constant 512 : index
// CHECK:         %[[POOL:.*]] = hip.get_pool(%[[CTX]], %[[SIZE]]) : memref<?xi8>
// CHECK-DAG:     %[[OFF0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[OFF1:.*]] = arith.constant 256 : index
// CHECK:         %[[V0:.*]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8> to memref<8x8xf32>
// CHECK:         hip.matmul{{.*}}outs(%[[V0]] :
// CHECK:         %[[V1:.*]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8> to memref<8x8xf32>
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[V1]] :
// CHECK:         return %[[V1]]
func.func @static_two_allocs(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  return %alloc1 : memref<8x8xf32>
}

// ===== Static pooling: three allocs with overlapping lifetimes =====
//
// alloc0 is used by matmul, alloc1 by softmax (alloc0 as input means alloc0 overlaps alloc1).
// alloc2 reads alloc1 -> alloc0 is dead by then and can share space at a different offset.
// Pool size depends on alignment and overlap analysis.
//
// CHECK-LABEL: func.func @static_three_allocs_overlap
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}) : memref<?xi8>
// CHECK-COUNT-3: memref.view %[[POOL]]
// CHECK:         return
func.func @static_three_allocs_overlap(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  %alloc2 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc1 : memref<8x8xf32>) outs(%alloc2 : memref<8x8xf32>)
  return %alloc2 : memref<8x8xf32>
}

// ===== CSE hazard: simultaneously-live same-typed inits stay DISTINCT =====
//
// Mirrors the pre-bufferize CSE hazard that Pipelines.cpp deliberately avoids
// (it runs `--hip-dedup-dps-inits`, not `-cse`): two ops with IDENTICAL
// inputs+type whose results are BOTH live at a later use. A stock `-cse` would
// fold the two `tensor.empty` inits to one SSA value and after bufferize the two
// ops would clobber one shared buffer. pool-allocs runs AFTER bufferize and
// assigns offsets by LIVENESS, so the three overlapping buffers (two producers +
// the consumer's out, all live at the final matmul) each get a distinct slot ->
// pool = 3 x 256 = 768. This is the liveness half of the invariant: the
// liveness-safe merge that recovers cross-layer memory (which `-cse` does
// unsafely) is `--hip-dedup-dps-inits`, guarded separately in
// `test/lit/Dialect/hip-dedup-dps-inits.mlir`.
//
// CHECK-LABEL: func.func @cse_hazard_live_distinct
// CHECK:         %[[SIZE:.*]] = arith.constant 768 : index
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}%[[SIZE]]) : memref<?xi8>
// CHECK-DAG:     %[[O0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[O1:.*]] = arith.constant 256 : index
// CHECK-DAG:     %[[O2:.*]] = arith.constant 512 : index
// CHECK-DAG:     memref.view %[[POOL]][%[[O0]]][] : memref<?xi8> to memref<8x8xf32>
// CHECK-DAG:     memref.view %[[POOL]][%[[O1]]][] : memref<?xi8> to memref<8x8xf32>
// CHECK-DAG:     memref.view %[[POOL]][%[[O2]]][] : memref<?xi8> to memref<8x8xf32>
// CHECK:         return
func.func @cse_hazard_live_distinct(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc1 : memref<8x8xf32>)
  %alloc2 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%alloc0, %alloc1 : memref<8x8xf32>, memref<8x8xf32>) outs(%alloc2 : memref<8x8xf32>)
  return %alloc2 : memref<8x8xf32>
}

// ===== Mixed element types: f32 and f16 in same pool =====
//
// memref<8x8xf32> = 256 bytes, memref<8x8xf16> = 128 bytes.
// Both go into the same i8 pool (type-agnostic via memref.view).
//
// CHECK-LABEL: func.func @mixed_element_types
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<8x8xf32>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<8x8xf16>
// CHECK:         return
func.func @mixed_element_types(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %c: memref<8x8xf16>) -> memref<8x8xf16> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf16>
  hip.miopen.softmax(%ctx) ins(%c : memref<8x8xf16>) outs(%alloc1 : memref<8x8xf16>)
  return %alloc1 : memref<8x8xf16>
}

// ===== Single alloc: still pooled (a lone hip.alloc would otherwise lower to
// the undefined hip_device_malloc — every transient must be pooled or written
// through to an out-param) =====
//
// CHECK-LABEL: func.func @single_alloc_pooled
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<8x8xf32>
// CHECK:         return
func.func @single_alloc_pooled(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  return %alloc0 : memref<8x8xf32>
}

// ===== No allocs: pass is a no-op =====
//
// CHECK-LABEL: func.func @no_allocs_noop
// CHECK-NOT:     hip.get_pool
// CHECK:         hip.miopen.softmax
// CHECK:         return
func.func @no_allocs_noop(
    %ctx: !hip.context,
    %in: memref<8x8xf32>,
    %out: memref<8x8xf32>) {
  hip.miopen.softmax(%ctx) ins(%in : memref<8x8xf32>) outs(%out : memref<8x8xf32>)
  return
}

// ===== Dynamic pooling: two dynamic allocs with same size SSA value =====
//
// Both memref<?x8xf32> allocs use %n. Pool comes from hip.get_pool.
//
// CHECK-LABEL: func.func @dynamic_two_allocs_same_size
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         return
func.func @dynamic_two_allocs_same_size(
    %ctx: !hip.context,
    %a: memref<?x8xf32>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x8xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<?x8xf32>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x8xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}

// ===== Mixed static + dynamic: f32 static + dynamic allocs in same pool =====
//
// One static memref<8x8xf32> (256 bytes) and one dynamic memref<?x8xf32>.
// Pool comes from hip.get_pool.
//
// CHECK-LABEL: func.func @mixed_static_dynamic
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<8x8xf32>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         return
func.func @mixed_static_dynamic(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %c: memref<?x8xf32>,
    %n: index) -> memref<?x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x8xf32>
  hip.miopen.softmax(%ctx) ins(%c : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}

// ===== Alignment: offsets are 256-byte aligned =====
//
// memref<1xf32> = 4 bytes, but should be rounded up to 256-byte alignment.
// Two allocs -> pool needs at least 512 bytes (256 + 256).
//
// CHECK-LABEL: func.func @alignment_256
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         %[[SIZE:.*]] = arith.constant 512 : index
// CHECK:         %[[POOL:.*]] = hip.get_pool(%[[CTX]], %[[SIZE]]) : memref<?xi8>
// CHECK-DAG:     arith.constant 0 : index
// CHECK-DAG:     arith.constant 256 : index
// CHECK:         memref.view %[[POOL]]
// CHECK:         memref.view %[[POOL]]
// CHECK:         return
//
// With --alignment=64, each 4-byte alloc rounds to 64 bytes -> pool = 128xi8.
// ALIGN64-LABEL: func.func @alignment_256
// ALIGN64:         arith.constant 128 : index
// ALIGN64:         hip.get_pool({{.*}}) : memref<?xi8>
func.func @alignment_256(
    %ctx: !hip.context,
    %in: memref<1xf32>) -> memref<1xf32> {
  %alloc0 = memref.alloc() : memref<1xf32>
  hip.miopen.softmax(%ctx) ins(%in : memref<1xf32>) outs(%alloc0 : memref<1xf32>)
  %alloc1 = memref.alloc() : memref<1xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<1xf32>) outs(%alloc1 : memref<1xf32>)
  return %alloc1 : memref<1xf32>
}

// ===== Dynamic: two buckets with different SSA sizes =====
//
// Two dynamic allocs with different dim values (%n vs %m) go into separate
// buckets.  Pool = bucket0_aligned_size + bucket1_aligned_size.
//
// CHECK-LABEL: func.func @dynamic_two_buckets
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x4xf32>
// CHECK:         return
func.func @dynamic_two_buckets(
    %ctx: !hip.context,
    %a: memref<?x8xf32>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %c: memref<?x4xf32>,
    %n: index, %m: index) -> memref<?x4xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<?x8xf32>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x8xf32>)
  %alloc1 = memref.alloc(%m) : memref<?x4xf32>
  hip.miopen.softmax(%ctx) ins(%c : memref<?x4xf32>) outs(%alloc1 : memref<?x4xf32>)
  return %alloc1 : memref<?x4xf32>
}

// ===== Metadata: attributes are no longer emitted =====
//
// Pool metadata (hipdnn.pool_size, hipdnn.buffer_offsets) is removed;
// pool sizing is handled at runtime via hip.get_pool.
//
// CHECK-LABEL: func.func @metadata_attributes
// CHECK-NOT:     hipdnn.pool_size
// CHECK-NOT:     hipdnn.buffer_offsets
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
func.func @metadata_attributes(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  return %alloc1 : memref<8x8xf32>
}

// ===== Pre-existing dealloc: erased (pool is runtime-owned) =====
//
// When input has memref.dealloc ops, they should be erased after pooling
// (since views into the pool can't be individually deallocated).
// No pool dealloc is inserted — the pool is owned by the runtime.
//
// CHECK-LABEL: func.func @preexisting_deallocs
// CHECK:         %[[POOL:.*]] = hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]
// CHECK:         memref.view %[[POOL]]
// CHECK-NOT:     memref.dealloc
// CHECK:         return
func.func @preexisting_deallocs(
    %ctx: !hip.context,
    %in: memref<8x8xf32>,
    %out: memref<8x8xf32>) {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%in : memref<8x8xf32>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  memref.dealloc %alloc0 : memref<8x8xf32>
  memref.dealloc %alloc1 : memref<8x8xf32>
  return
}

// ===== Dynamic metadata: attributes no longer emitted =====
//
// CHECK-LABEL: func.func @dynamic_metadata
// CHECK-NOT:     hipdnn.buffer_offsets
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
func.func @dynamic_metadata(
    %ctx: !hip.context,
    %a: memref<?x8xf32>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x8xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<?x8xf32>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x8xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}

// ===== Retired-hack matrix (one test per pattern that the prior =========
// hoist design needed a dedicated helper for) ============================
//
// Each test below pins one IR shape that, in the prior hoist-based
// design, was (or in PR #259's proposed extension would have been)
// handled by a dedicated helper: `foldDimOfReshape`, the four
// `resolveDimAtSource` extensions (cast / view / subview / reshape), or
// `foldScalarMemrefArith`. The dominator-emit path needs none of them —
// `findLatestLegalInsertionPoint` selects an insertion point downstream
// of the dyn-operand def and the bucket size SSA arithmetic emits there,
// with no code motion.
//
// Every test uses the "natural-position" layout: every dyn-operand def
// precedes every pooled alloc. Real-model post-pool IR audits (e.g.
// Llama-8B-asym dyn-shape compile #0) show every dyn-size def chain
// living entirely in the `memref.dim`-of-function-arg prefix at block
// start, so this layout is the production case.

// ===== Retired hack 1: foldDimOfReshape via memref.collapse_shape =====
//
// `memref.dim %collapsed, %c0` where %collapsed is a memref.collapse_shape
// of a function arg. The dim op stays at its natural position; the
// bucket's size SSA simply emits AFTER the dim op, BEFORE the first
// alloc.
//
// CHECK-LABEL: func.func @dim_of_collapse
// CHECK:         memref.collapse_shape
// CHECK:         memref.dim
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK-COUNT-2: memref.view %{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         return
func.func @dim_of_collapse(
    %ctx: !hip.context,
    %a: memref<?x4x8xf16>,
    %x: memref<?x8xf32>) -> memref<?x8xf32> {
  %c0 = arith.constant 0 : index
  %collapsed = memref.collapse_shape %a [[0], [1, 2]]
      : memref<?x4x8xf16> into memref<?x32xf16>
  %dim = memref.dim %collapsed, %c0 : memref<?x32xf16>
  %alloc0 = memref.alloc(%dim) : memref<?x8xf32>
  hip.miopen.softmax(%ctx) ins(%x : memref<?x8xf32>) outs(%alloc0 : memref<?x8xf32>)
  %alloc1 = memref.alloc(%dim) : memref<?x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}

// ===== Retired hack 2: resolveDimAtSource cast extension =====
//
// `memref.dim %casted, %c0` where %casted is a memref.cast of a
// function arg with a non-identity layout.
//
// CHECK-LABEL: func.func @dim_of_cast
// CHECK:         memref.cast
// CHECK:         memref.dim
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view %{{.*}} : memref<?xi8> to memref<?x16xf32>
// CHECK:         return
func.func @dim_of_cast(
    %ctx: !hip.context,
    %y: memref<?x16xf32, strided<[?, ?], offset: ?>>,
    %z: memref<?x16xf32>) -> memref<?x16xf32> {
  %c0 = arith.constant 0 : index
  %casted = memref.cast %y
      : memref<?x16xf32, strided<[?, ?], offset: ?>> to memref<?x16xf32>
  %dim = memref.dim %casted, %c0 : memref<?x16xf32>
  %alloc0 = memref.alloc(%dim) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%z : memref<?x16xf32>) outs(%alloc0 : memref<?x16xf32>)
  %alloc1 = memref.alloc(%dim) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x16xf32>) outs(%alloc1 : memref<?x16xf32>)
  return %alloc1 : memref<?x16xf32>
}

// ===== Retired hack 3: resolveDimAtSource view extension =====
//
// `memref.dim` of a typed memref.view of a pre-existing buffer feeds
// the next bucket's size.
//
// CHECK-LABEL: func.func @dim_of_view
// CHECK:         memref.view
// CHECK:         memref.dim
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         return
func.func @dim_of_view(
    %ctx: !hip.context,
    %raw: memref<?xi8>,
    %x: memref<?x16xf32>,
    %n: index) -> memref<?x16xf32> {
  %c0 = arith.constant 0 : index
  %off = arith.constant 0 : index
  %v = memref.view %raw[%off][%n] : memref<?xi8> to memref<?x16xf32>
  %dim = memref.dim %v, %c0 : memref<?x16xf32>
  %alloc0 = memref.alloc(%dim) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%x : memref<?x16xf32>) outs(%alloc0 : memref<?x16xf32>)
  %alloc1 = memref.alloc(%dim) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x16xf32>) outs(%alloc1 : memref<?x16xf32>)
  return %alloc1 : memref<?x16xf32>
}

// ===== Retired hack 4: resolveDimAtSource subview extension (rank-preserving) =====
//
// `memref.dim` of a rank-preserving memref.subview, modeling a packed-QKV
// split where the parent is `memref<?x?x8192xf16>` and the child slice is
// `memref<?x?x2048xf16, strided<[?, 8192, 1], offset: 2048>>`.
//
// CHECK-LABEL: func.func @dim_of_subview
// CHECK:         memref.subview
// CHECK:         memref.dim
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         return
func.func @dim_of_subview(
    %ctx: !hip.context,
    %parent: memref<?x?x8192xf16>,
    %xin: memref<?x?x2048xf16>) -> memref<?x?x2048xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_b = memref.dim %parent, %c0 : memref<?x?x8192xf16>
  %dim_s = memref.dim %parent, %c1 : memref<?x?x8192xf16>
  %sl = memref.subview %parent[0, 0, 2048][%dim_b, %dim_s, 2048][1, 1, 1]
      : memref<?x?x8192xf16>
        to memref<?x?x2048xf16, strided<[?, 8192, 1], offset: 2048>>
  %dim = memref.dim %sl, %c1
      : memref<?x?x2048xf16, strided<[?, 8192, 1], offset: 2048>>
  %alloc0 = memref.alloc(%dim_b, %dim) : memref<?x?x2048xf16>
  hip.miopen.softmax(%ctx) ins(%xin : memref<?x?x2048xf16>) outs(%alloc0 : memref<?x?x2048xf16>)
  %alloc1 = memref.alloc(%dim_b, %dim) : memref<?x?x2048xf16>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x?x2048xf16>) outs(%alloc1 : memref<?x?x2048xf16>)
  return %alloc1 : memref<?x?x2048xf16>
}

// ===== Retired hack 5: resolveDimAtSource reshape extension =====
//
// `memref.dim %r, %c1` where `%r = memref.reshape %src(%shape)` with
// multi-dyn input and partially-static output.
//
// CHECK-LABEL: func.func @dim_of_reshape
// CHECK:         memref.reshape
// CHECK:         memref.dim
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         return
func.func @dim_of_reshape(
    %ctx: !hip.context,
    %src: memref<?x?x?xf16>,
    %shape: memref<4xindex>,
    %x: memref<1x?x16x72xf16>) -> memref<1x?x16x72xf16> {
  %c1 = arith.constant 1 : index
  %r = memref.reshape %src(%shape)
      : (memref<?x?x?xf16>, memref<4xindex>) -> memref<1x?x16x72xf16>
  %dim = memref.dim %r, %c1 : memref<1x?x16x72xf16>
  %alloc0 = memref.alloc(%dim) : memref<1x?x16x72xf16>
  hip.miopen.softmax(%ctx) ins(%x : memref<1x?x16x72xf16>) outs(%alloc0 : memref<1x?x16x72xf16>)
  %alloc1 = memref.alloc(%dim) : memref<1x?x16x72xf16>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<1x?x16x72xf16>) outs(%alloc1 : memref<1x?x16x72xf16>)
  return %alloc1 : memref<1x?x16x72xf16>
}

// ===== Retired hack 6: foldScalarMemrefArith / strict-SSA via host-scratch chain =====
//
// A small host-side scratch buffer holds a scalar dyn dim (mirroring the
// `tensor.from_elements` lowering pattern), and a `memref.load` reads it
// back into an SSA value that feeds two pooled allocs.
//
// The scratch buffer is modeled here as a function argument
// (`%scratch: memref<2xindex>`) to match the IR shape that
// `hip-materialize-host-scalars` produces in production: that pass
// rewrites the original `memref.alloc + memref.store` into a
// `memref.view` of a `hip.get_host_scratch` buffer BEFORE
// hip-pool-allocs runs, so the scratch is no longer a pool candidate
// from pool-allocs's point of view.
//
// CHECK-LABEL: func.func @dim_from_host_scratch
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         return
func.func @dim_from_host_scratch(
    %ctx: !hip.context,
    %scratch: memref<2xindex>,
    %x: memref<?x16xf32>,
    %seed: index) -> memref<?x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  memref.store %seed, %scratch[%c0] : memref<2xindex>
  memref.store %seed, %scratch[%c1] : memref<2xindex>
  %dim = memref.load %scratch[%c0] : memref<2xindex>
  %alloc0 = memref.alloc(%dim) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%x : memref<?x16xf32>) outs(%alloc0 : memref<?x16xf32>)
  %alloc1 = memref.alloc(%dim) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x16xf32>) outs(%alloc1 : memref<?x16xf32>)
  return %alloc1 : memref<?x16xf32>
}

// ===== Multi-domain: two pools (host-load splits the dyn-def graph) =====
//
// `%alloc0` consumes a dim derived from a function arg (hoist-feasible above
// `%alloc0` itself). A `memref.load` then produces a NEW dim BELOW `%alloc0`,
// which `%alloc1` consumes. The load is non-speculatable so it cannot be
// hoisted above `%alloc0`; therefore `%alloc1`'s pool anchor must come AFTER
// the load, which puts it below `%alloc0`. The two allocs end up in
// different dominance domains and emit two `hip.get_pool` calls — one
// anchored above `%alloc0`, one between the load and `%alloc1`.
//
// Domain 0 prints with no attr-dict (default-elided). Domain 1 prints with
// `{domain_id = 1 : i64}` so the runtime selects the right pool slot.
//
// CHECK-LABEL: func.func @multi_domain_two_pools
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         memref.dim %{{.*}}, %{{.*}}
// CHECK:         %[[POOL0:.*]] = hip.get_pool(%[[CTX]], %{{.*}}) : memref<?xi8>
// CHECK-NOT:     domain_id
// CHECK:         %[[V0:.*]] = memref.view %[[POOL0]]{{.*}} to memref<?x16xf32>
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[V0]] :
// CHECK:         memref.load
// CHECK:         %[[POOL1:.*]] = hip.get_pool(%[[CTX]], %{{.*}}) {domain_id = 1 : i64} : memref<?xi8>
// CHECK:         %[[V1:.*]] = memref.view %[[POOL1]]{{.*}} to memref<?x16xf32>
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[V1]] :
// CHECK:         return %[[V1]]
func.func @multi_domain_two_pools(
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

// ===== Multi-domain: three pools (cascading load+alloc chain) =====
//
// Three back-to-back load+alloc chains: each load produces a fresh dyn dim
// BELOW the previous alloc, so each alloc opens a new domain.
//
// CHECK-LABEL: func.func @multi_domain_three_pools
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK-NOT:     domain_id
// CHECK:         memref.load
// CHECK:         hip.get_pool({{.*}}) {domain_id = 1 : i64} : memref<?xi8>
// CHECK:         memref.load
// CHECK:         hip.get_pool({{.*}}) {domain_id = 2 : i64} : memref<?xi8>
// CHECK:         return
func.func @multi_domain_three_pools(
    %ctx: !hip.context,
    %x: memref<?x16xf32>,
    %scratch: memref<3xindex>) -> memref<?x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %x, %c0 : memref<?x16xf32>
  %alloc0 = memref.alloc(%d0) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%x : memref<?x16xf32>) outs(%alloc0 : memref<?x16xf32>)
  %d1 = memref.load %scratch[%c0] : memref<3xindex>
  %alloc1 = memref.alloc(%d1) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x16xf32>) outs(%alloc1 : memref<?x16xf32>)
  %d2 = memref.load %scratch[%c1] : memref<3xindex>
  %alloc2 = memref.alloc(%d2) : memref<?x16xf32>
  hip.miopen.softmax(%ctx) ins(%alloc1 : memref<?x16xf32>) outs(%alloc2 : memref<?x16xf32>)
  return %alloc2 : memref<?x16xf32>
}
