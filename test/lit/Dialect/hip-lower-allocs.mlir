// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-lower-allocs (memref.alloc -> hip.alloc/free).
//
// Ownership convention verified by these tests:
//   - Every memref.alloc becomes hip.alloc (device memory via hipMalloc).
//   - Returned buffers are caller-owned: no hip.free emitted.
//   - Non-returned buffers get hip.free (using the HIP context from arg 0).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-lower-allocs %s 2>&1 | FileCheck %s

// Two static allocs: alloc0 is not returned -> hip.free.
// alloc1 is returned -> no hip.free (caller-owned).
// CHECK-LABEL: func.func @static_lower
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         %[[A:.*]] = hip.alloc(%[[CTX]]) : memref<2x64x64xf32>
// CHECK:         hip.matmul
// CHECK:         %[[B:.*]] = hip.alloc(%[[CTX]]) : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[CTX]], %[[A]]) : memref<2x64x64xf32>
// CHECK:         return %[[B]]
func.func @static_lower(
    %ctx: !hip.context,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<2x64x64xf32>) outs(%alloc1 : memref<2x64x64xf32>)
  return %alloc1 : memref<2x64x64xf32>
}

// One dynamic memref<?x64xf32> alloc sized by %n: the dynamic size is forwarded
// to hip.alloc. Since the buffer is returned, no hip.free is inserted.
// CHECK-LABEL: func.func @dynamic_lower
// CHECK-SAME:    (%[[CTX2:.*]]: !hip.context,
// CHECK:         hip.alloc(%[[CTX2]], %{{.*}}) : memref<?x64xf32>
// CHECK:         hip.matmul
// CHECK-NOT:     hip.free
// CHECK:         return
func.func @dynamic_lower(
    %ctx: !hip.context,
    %a: memref<?x64xf32>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x64xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x64xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<?x64xf32>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x64xf32>)
  return %alloc0 : memref<?x64xf32>
}

// No memref.alloc in function: pass is a no-op.
// CHECK-LABEL: func.func @no_allocs_noop
// CHECK-NOT:     hip.alloc
// CHECK:         return
func.func @no_allocs_noop(%ctx: !hip.context, %a: memref<2x64x64xf32>) -> memref<2x64x64xf32> {
  return %a : memref<2x64x64xf32>
}

// Three memref<2x64x64xf32> allocs: alloc0 and alloc1 are not returned, so
// both get hip.free after their last use. alloc2 is returned.
// CHECK-LABEL: func.func @multiple_frees
// CHECK-SAME:    (%[[CTX3:.*]]: !hip.context,
// CHECK:         %[[A:.*]] = hip.alloc(%[[CTX3]]) : memref<2x64x64xf32>
// CHECK:         hip.matmul
// CHECK:         %[[B:.*]] = hip.alloc(%[[CTX3]]) : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[CTX3]], %[[A]]) : memref<2x64x64xf32>
// CHECK:         %[[C:.*]] = hip.alloc(%[[CTX3]]) : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[CTX3]], %[[B]]) : memref<2x64x64xf32>
// CHECK:         return %[[C]]
func.func @multiple_frees(
    %ctx: !hip.context,
    %input: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %w: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.matmul(%ctx) ins(%input, %w : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<2x64x64xf32>) outs(%alloc1 : memref<2x64x64xf32>)
  %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%ctx) ins(%alloc1 : memref<2x64x64xf32>) outs(%alloc2 : memref<2x64x64xf32>)
  return %alloc2 : memref<2x64x64xf32>
}

// memref.dealloc of a hip.alloc result is converted to hip.free and the
// memref.dealloc is erased.  The pass should NOT insert an extra hip.free
// for alloc0 since it was already explicitly deallocated.
// CHECK-LABEL: func.func @dealloc_conversion
// CHECK-SAME:    (%[[CTX4:.*]]: !hip.context,
// CHECK:         %[[ALLOC:.*]] = hip.alloc(%[[CTX4]]) : memref<8x8xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[CTX4]], %[[ALLOC]]) : memref<8x8xf32>
// CHECK-NOT:     memref.dealloc
// CHECK-NOT:     hip.free
// CHECK:         return
func.func @dealloc_conversion(
    %ctx: !hip.context,
    %in: memref<8x8xf32>,
    %out: memref<8x8xf32>) {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%in : memref<8x8xf32>) outs(%alloc0 : memref<8x8xf32>)
  memref.dealloc %alloc0 : memref<8x8xf32>
  return
}

// Two allocs where both are used by multiple ops: verify free placement
// accounts for all direct users.  alloc0 is used by both matmul (as output)
// and softmax (as input).  Free goes after softmax (the later user).
// CHECK-LABEL: func.func @free_after_all_direct_users
// CHECK-SAME:    (%[[CTX5:.*]]: !hip.context,
// CHECK:         %[[A:.*]] = hip.alloc(%[[CTX5]]) : memref<8x8xf32>
// CHECK:         hip.matmul{{.*}}outs(%[[A]] :
// CHECK:         %[[B:.*]] = hip.alloc(%[[CTX5]]) : memref<8x8xf32>
// CHECK:         hip.miopen.softmax{{.*}}ins(%[[A]]{{.*}}outs(%[[B]] :
// CHECK:         hip.free(%[[CTX5]], %[[A]]) : memref<8x8xf32>
// CHECK:         return %[[B]]
func.func @free_after_all_direct_users(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  return %alloc1 : memref<8x8xf32>
}

// Alias chain: alloc0 -> subview -> cast.  BufferViewFlowAnalysis tracks all
// downstream aliases via resolve(), so hip.free is placed after the LAST
// transitive consumer (the matmul that reads the cast), not just after the
// subview.  The alloc is not returned (return %out), so it must be freed.
// CHECK-LABEL: func.func @view_alias_chain_free_placement
// CHECK-SAME:    (%[[CTX6:.*]]: !hip.context,
// CHECK:         %[[ALLOC:.*]] = hip.alloc(%[[CTX6]]) : memref<2x64x64xf32>
// CHECK:         memref.subview
// CHECK:         memref.cast
// CHECK:         hip.matmul
// CHECK:         hip.free(%[[CTX6]], %[[ALLOC]]) : memref<2x64x64xf32>
// CHECK:         return
func.func @view_alias_chain_free_placement(
    %ctx: !hip.context,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %out: memref<1x64x64xf32>) -> memref<1x64x64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  %sv = memref.subview %alloc0[0, 0, 0][1, 64, 64][1, 1, 1]
      : memref<2x64x64xf32> to memref<1x64x64xf32, strided<[4096, 64, 1]>>
  %cast = memref.cast %sv
      : memref<1x64x64xf32, strided<[4096, 64, 1]>>
        to memref<1x64x64xf32, strided<[?, ?, ?], offset: ?>>
  hip.matmul(%ctx) ins(%cast, %b : memref<1x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%out : memref<1x64x64xf32>)
  return %out : memref<1x64x64xf32>
}

// Pool pattern: after hip-pool-allocs, the function has a single memref.alloc
// (the pool) with memref.view aliases, one of which is returned.
// BufferViewFlowAnalysis::resolve(pool) must return {pool, view0, view1},
// so isAliasInSet detects that view1 (returned) aliases the pool.
// No hip.free should be emitted for the pool.
// CHECK-LABEL: func.func @pool_view_returned_no_free
// CHECK-SAME:    (%[[CTX7:.*]]: !hip.context,
// CHECK:         %[[POOL:.*]] = hip.alloc(%[[CTX7]]) : memref<512xi8>
// CHECK:         memref.view %[[POOL]]
// CHECK:         memref.view %[[POOL]]
// CHECK-NOT:     hip.free
// CHECK:         return
func.func @pool_view_returned_no_free(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %pool = memref.alloc() : memref<512xi8>
  %view0 = memref.view %pool[%c0][] : memref<512xi8> to memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%view0 : memref<8x8xf32>)
  %view1 = memref.view %pool[%c256][] : memref<512xi8> to memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%view0 : memref<8x8xf32>) outs(%view1 : memref<8x8xf32>)
  return %view1 : memref<8x8xf32>
}

// Allocs inside an scf.while (onnx.Loop scan lowering). Each is freed in its
// own block: an iteration-local buffer after its last use, a carried value
// after its last use in the block that receives it, and a buffer only one
// scf.if branch yields when the scf.if did not return it. Buffers passed on
// by scf.condition / scf.yield are freed by the next holder.
// CHECK-LABEL: func.func @while_carried
// CHECK-SAME:    (%[[CTX8:[^:]*]]: !hip.context, %[[INIT:[^:]*]]: memref<?xf32>, %[[SRC:[^:]*]]: memref<4xf32>
// CHECK:         %[[LOOP:.*]] = scf.while (%[[A:.*]] = %[[INIT]])
// CHECK:           %[[COPY:.*]] = hip.alloc(%[[CTX8]], %{{.*}}) : memref<?xf32>
// CHECK:           memref.copy %[[A]], %[[COPY]]
// CHECK-NEXT:      hip.free(%[[CTX8]], %[[A]]) : memref<?xf32>
// CHECK:           scf.condition(%{{.*}}) %[[COPY]] : memref<?xf32>
// CHECK:         ^bb0(%[[B:.*]]: memref<?xf32>):
// CHECK:           %[[TMP:.*]] = hip.alloc(%[[CTX8]]) : memref<4xf32>
// CHECK:           %[[FRESH:.*]] = hip.alloc(%[[CTX8]]) : memref<4xf32>
// CHECK:           memref.copy %[[TMP]], %[[FRESH]]
// CHECK-NEXT:      hip.free(%[[CTX8]], %[[TMP]]) : memref<4xf32>
// CHECK:           %[[NEXT:.*]] = scf.if
// CHECK:             %[[GROW:.*]] = hip.alloc
// CHECK-NOT:         hip.free
// CHECK:             scf.yield %[[GROW]]
// CHECK:           hip.free(%[[CTX8]], %[[B]]) : memref<?xf32>
// CHECK:           %[[P0:.*]] = memref.extract_aligned_pointer_as_index %[[FRESH]]
// CHECK:           %[[P1:.*]] = memref.extract_aligned_pointer_as_index %[[NEXT]]
// CHECK:           arith.cmpi eq, %[[P0]], %[[P1]]
// CHECK:           scf.if
// CHECK-NEXT:        hip.free(%[[CTX8]], %[[FRESH]]) : memref<4xf32>
// CHECK:           scf.yield %[[NEXT]]
// CHECK:         memref.copy
// CHECK-NEXT:    hip.free(%[[CTX8]], %[[LOOP]]) : memref<?xf32>
// CHECK-NEXT:    return
func.func @while_carried(%ctx: !hip.context, %init: memref<?xf32>,
                         %src: memref<4xf32>, %n: index, %flag: i1) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %r = scf.while (%a = %init) : (memref<?xf32>) -> memref<?xf32> {
    %d = memref.dim %a, %c0 : memref<?xf32>
    %copy = memref.alloc(%d) : memref<?xf32>
    memref.copy %a, %copy : memref<?xf32> to memref<?xf32>
    %go = arith.cmpi slt, %d, %n : index
    scf.condition(%go) %copy : memref<?xf32>
  } do {
  ^bb0(%b: memref<?xf32>):
    %tmp = memref.alloc() : memref<4xf32>
    memref.copy %src, %tmp : memref<4xf32> to memref<4xf32>
    %fresh = memref.alloc() : memref<4xf32>
    %cast = memref.cast %fresh : memref<4xf32> to memref<?xf32>
    memref.copy %tmp, %fresh : memref<4xf32> to memref<4xf32>
    %next = scf.if %flag -> (memref<?xf32>) {
      %d2 = memref.dim %b, %c0 : memref<?xf32>
      %size = arith.addi %d2, %c4 : index
      %grow = memref.alloc(%size) : memref<?xf32>
      %head = memref.subview %grow[0] [%d2] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      memref.copy %b, %head : memref<?xf32> to memref<?xf32, strided<[1]>>
      scf.yield %grow : memref<?xf32>
    } else {
      scf.yield %cast : memref<?xf32>
    }
    scf.yield %next : memref<?xf32>
  }
  %head = memref.subview %r[0] [4] [1] : memref<?xf32> to memref<4xf32, strided<[1]>>
  memref.copy %head, %src : memref<4xf32, strided<[1]>> to memref<4xf32>
  return
}
