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

// RUN: %hip-mlir-opt --hip-lower-allocs %s 2>&1 | %FileCheck %s

// Two static allocs: alloc0 is not returned -> hip.free.
// alloc1 is returned -> no hip.free (caller-owned).
// CHECK-LABEL: func.func @static_lower
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         %[[A:.*]] = hip.alloc(%[[CTX]]) : memref<2x64x64xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         %[[B:.*]] = hip.alloc(%[[CTX]]) : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[CTX]], %[[A]]) : memref<2x64x64xf32>
// CHECK:         return %[[B]]
func.func @static_lower(
    %ctx: !hip.context,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%ctx) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<2x64x64xf32>) outs(%alloc1 : memref<2x64x64xf32>)
  return %alloc1 : memref<2x64x64xf32>
}

// One dynamic memref<?x64xf32> alloc sized by %n: the dynamic size is forwarded
// to hip.alloc. Since the buffer is returned, no hip.free is inserted.
// CHECK-LABEL: func.func @dynamic_lower
// CHECK-SAME:    (%[[CTX2:.*]]: !hip.context,
// CHECK:         hip.alloc(%[[CTX2]], %{{.*}}) : memref<?x64xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK-NOT:     hip.free
// CHECK:         return
func.func @dynamic_lower(
    %ctx: !hip.context,
    %a: memref<?x64xf32>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x64xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x64xf32>
  hip.hipblaslt.matmul(%ctx) ins(%a, %b : memref<?x64xf32>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x64xf32>)
  return %alloc0 : memref<?x64xf32>
}

// No !hip.context arg: pass is a no-op, memref.alloc is preserved.
// CHECK-LABEL: func.func @no_context_noop
// CHECK:         memref.alloc()
// CHECK-NOT:     hip.alloc
// CHECK:         return
func.func @no_context_noop(%a: memref<2x64x64xf32>) -> memref<2x64x64xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  memref.copy %a, %alloc : memref<2x64x64xf32> to memref<2x64x64xf32>
  return %alloc : memref<2x64x64xf32>
}

// Three memref<2x64x64xf32> allocs: alloc0 and alloc1 are not returned, so
// both get hip.free after their last use. alloc2 is returned.
// CHECK-LABEL: func.func @multiple_frees
// CHECK-SAME:    (%[[CTX3:.*]]: !hip.context,
// CHECK:         %[[A:.*]] = hip.alloc(%[[CTX3]]) : memref<2x64x64xf32>
// CHECK:         hip.hipblaslt.matmul
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
  hip.hipblaslt.matmul(%ctx) ins(%input, %w : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<2x64x64xf32>) outs(%alloc1 : memref<2x64x64xf32>)
  %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%ctx) ins(%alloc1 : memref<2x64x64xf32>) outs(%alloc2 : memref<2x64x64xf32>)
  return %alloc2 : memref<2x64x64xf32>
}
