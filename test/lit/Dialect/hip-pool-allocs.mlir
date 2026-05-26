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

// ===== Single alloc: also pooled =====
//
// The pass previously short-circuited at fewer than 2 allocs, but that left
// the single alloc to flow through `LowerAllocs` -> `hip.alloc` ->
// `MemoryLowering` which lowers to a `hip_device_malloc` symbol the runtime
// does not export. We now pool every alloc, even a single one, so the
// dyn-pool path is the only allocation path the compiled model uses.
//
// CHECK-LABEL: func.func @single_alloc_noop
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view {{.*}} : memref<?xi8> to memref<8x8xf32>
// CHECK-NOT:     memref.alloc() : memref<8x8xf32>
// CHECK:         return
func.func @single_alloc_noop(
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

// ===== Dynamic: dim-op hoisted above firstPooledAlloc =====
//
// Bufferization places `memref.dim` right next to its consumer alloc.
// PoolAllocs must hoist that `memref.dim` so the dyn byte-size computation
// (and the `hip.get_pool` that consumes it) dominates the first pooled
// `memref.view`. Verifies the hoisting fix that unlocks fully-symbolic
// embedding models (qwen3-35b-a3b).
//
// CHECK-LABEL: func.func @dynamic_dim_hoisted
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[A:.*]]: memref<8x8xf32{{.*}}>, %[[B:.*]]: memref<?x8xf32>)
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = memref.dim %[[B]], %[[C0]]
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK:         memref.view {{.*}} : memref<?xi8> to memref<8x8xf32>
// CHECK:         hip.matmul
// CHECK:         memref.view {{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         return
func.func @dynamic_dim_hoisted(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<?x8xf32>) -> memref<?x8xf32> {
  // static alloc first -> firstPooledAlloc is here
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %a : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  // memref.dim is created AFTER the static alloc -- the pass must hoist it.
  %c0 = arith.constant 0 : index
  %dim = memref.dim %b, %c0 : memref<?x8xf32>
  %alloc1 = memref.alloc(%dim) : memref<?x8xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}

// ===== Dynamic: non-hoistable load leaves alloc unpooled =====
//
// When a dyn-size operand transitively depends on a non-pure op (here a
// `memref.load` of a shape buffer that an earlier `hip.shape` writes to),
// PoolAllocs cannot hoist the chain above `firstPooledAlloc` without
// reordering the load past the store, which would change semantics. The
// pass leaves the alloc as `memref.alloc` and `LowerAllocs` +
// `MemoryLowering` route it through `hipdnn_ep_state_dyn_pool_alloc`.
//
// The pooled allocs (the static shape buffer + the static input matmul
// output) still go through the static pool; the load-dependent alloc
// remains a `memref.alloc`.
//
// CHECK-LABEL: func.func @dynamic_load_dependent_unpooled
// CHECK:         hip.get_pool({{.*}}) : memref<?xi8>
// CHECK-DAG:     memref.view {{.*}} : memref<?xi8> to memref<3xi64>
// CHECK-DAG:     memref.view {{.*}} : memref<?xi8> to memref<8x8xf32>
// CHECK:         hip.shape{{.*}}outs({{.*}} : memref<3xi64>)
// CHECK:         memref.load
// CHECK:         arith.index_cast
// CHECK:         memref.alloc({{.*}}) {{.*}}: memref<?xi8>
// CHECK:         return
func.func @dynamic_load_dependent_unpooled(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %src: memref<2x3x4xf32>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %shape = memref.alloc() : memref<3xi64>
  hip.shape(%ctx) ins(%src : memref<2x3x4xf32>) outs(%shape : memref<3xi64>) {element_dim_specs = [[array<i64: 0, 2, 0, 0, 0, -1, -1, -1>], [array<i64: 0, 3, 0, 0, 0, -1, -1, -1>], [array<i64: 0, 4, 0, 0, 0, -1, -1, -1>]]}
  %c0 = arith.constant 0 : index
  %d0_i64 = memref.load %shape[%c0] : memref<3xi64>
  %d0 = arith.index_cast %d0_i64 : i64 to index
  %unpooled = memref.alloc(%d0) : memref<?xi8>
  return %alloc0 : memref<8x8xf32>
}
