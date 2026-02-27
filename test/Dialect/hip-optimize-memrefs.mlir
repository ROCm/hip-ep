// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-optimize-memrefs (liveness-based buffer reuse).
//===----------------------------------------------------------------------===//

// RUN: %hip-mlir-opt --hip-optimize-memrefs %s | %FileCheck %s

// Three memref<2x64x64xf32> buffers: alloc0 becomes dead after matmul writes
// alloc1, so alloc1 reuses alloc0. alloc2 (softmax output) is new.
//
// Intervals:
//   alloc0: [0, 0]   dead before index 1
//   alloc1: [1, 2]   reuses alloc0's slot (0 < 1)
//   alloc2: [2, ret]  new slot (returned)
//
// CHECK-LABEL: func.func @static_reuse_same_type
// CHECK:         %[[A:.*]] = memref.alloc()
// CHECK:         hip.hipblaslt.matmul
// CHECK-NOT:     memref.alloc()
// CHECK:         hip.hipblaslt.matmul{{.*}}outs(%[[A]] :
// CHECK:         %[[B:.*]] = memref.alloc()
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[B]] :
// CHECK:         return %[[B]]
func.func @static_reuse_same_type(
    %handle: !hip.handle,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %c: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %c : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc1 : memref<2x64x64xf32>)
  %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%handle) ins(%alloc1 : memref<2x64x64xf32>) outs(%alloc2 : memref<2x64x64xf32>)
  return %alloc2 : memref<2x64x64xf32>
}

// alloc0 (memref<2x64x64xf32>, 32768 bytes) becomes dead after mul reads it
// into alloc1. alloc2 (memref<64xf32>, 256 bytes) fits inside alloc0's
// byte-size, so alloc0 is reused via memref.reinterpret_cast.
//
// Intervals:
//   alloc0: [0, 1]   dead before index 2
//   alloc1: [1, 2]   new slot (alloc0 still live at 1)
//   alloc2: [2, ret]  reuses alloc0 via reinterpret_cast (256 <= 32768)
//
// CHECK-LABEL: func.func @bytesize_reuse_reinterpret_cast
// CHECK:         %[[BIG:.*]] = memref.alloc(){{.*}}: memref<2x64x64xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         %[[OTHER:.*]] = memref.alloc(){{.*}}: memref<2x64x64xf32>
// CHECK:         hip.miopen.mul
// CHECK-NOT:     memref.alloc()
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[BIG]]
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[CAST]] :
// CHECK:         return %[[CAST]]
func.func @bytesize_reuse_reinterpret_cast(
    %handle: !hip.handle,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %s: memref<f32, strided<[], offset: ?>>) -> memref<64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.mul(%handle) ins(%alloc0, %s : memref<2x64x64xf32>, memref<f32, strided<[], offset: ?>>) outs(%alloc1 : memref<2x64x64xf32>)
  %alloc2 = memref.alloc() : memref<64xf32>
  hip.miopen.softmax(%handle) ins(%alloc1 : memref<2x64x64xf32>) outs(%alloc2 : memref<64xf32>)
  return %alloc2 : memref<64xf32>
}

// alloc0 is read as BOTH inputs to the matmul that writes alloc1.  Since
// alloc0 is still live at the same op that defines alloc1, reuse is blocked.
//
// Intervals:
//   alloc0: [0, 1]   used at index 1 (as matmul input)
//   alloc1: [1, ret]  cannot reuse alloc0 (0's lastUse=1 >= 1's def=1)
//
// CHECK-LABEL: func.func @no_reuse_overlapping_lifetimes
// CHECK:         %[[A:.*]] = memref.alloc()
// CHECK:         hip.hipblaslt.matmul
// CHECK:         %[[B:.*]] = memref.alloc()
// CHECK:         hip.hipblaslt.matmul{{.*}}ins(%[[A]], %[[A]]{{.*}}outs(%[[B]] :
// CHECK:         return %[[B]]
func.func @no_reuse_overlapping_lifetimes(
    %handle: !hip.handle,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%alloc0, %alloc0 : memref<2x64x64xf32>, memref<2x64x64xf32>) outs(%alloc1 : memref<2x64x64xf32>)
  return %alloc1 : memref<2x64x64xf32>
}

// Three memref<?x64xf32> buffers all sized by the same SSA value %n.
// alloc0 becomes dead after softmax reads it, so alloc2 reuses alloc0.
//
// Intervals:
//   alloc0: [0, 1]   dead before index 2
//   alloc1: [1, 2]   new slot (alloc0 still live at 1)
//   alloc2: [2, ret]  reuses alloc0 (same type, same SSA dim %n)
//
// CHECK-LABEL: func.func @dynamic_same_dim_reuse
// CHECK:         %[[A:.*]] = memref.alloc(%arg3)
// CHECK:         hip.hipblaslt.matmul
// CHECK:         %[[B:.*]] = memref.alloc(%arg3)
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[B]] :
// CHECK-NOT:     memref.alloc
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[A]] :
// CHECK:         return %[[A]]
func.func @dynamic_same_dim_reuse(
    %handle: !hip.handle,
    %a: memref<?x64xf32>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x64xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<?x64xf32>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x64xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x64xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<?x64xf32>) outs(%alloc1 : memref<?x64xf32>)
  %alloc2 = memref.alloc(%n) : memref<?x64xf32>
  hip.miopen.softmax(%handle) ins(%alloc1 : memref<?x64xf32>) outs(%alloc2 : memref<?x64xf32>)
  return %alloc2 : memref<?x64xf32>
}

// Two memref<?x64xf32> buffers with different SSA dimension values (%n vs %m).
// Even though both are memref<?x64xf32>, the pass requires SSA identity of
// dynamic-size operands to guarantee equal runtime sizes.  Different SSA
// values means no reuse.
// CHECK-LABEL: func.func @no_reuse_different_dynamic
// CHECK:         memref.alloc(%arg3) : memref<?x64xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         memref.alloc(%arg4) : memref<?x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         return
func.func @no_reuse_different_dynamic(
    %handle: !hip.handle,
    %a: memref<?x64xf32>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %n: index, %m: index) -> memref<?x64xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<?x64xf32>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x64xf32>)
  %alloc1 = memref.alloc(%m) : memref<?x64xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<?x64xf32>) outs(%alloc1 : memref<?x64xf32>)
  return %alloc1 : memref<?x64xf32>
}

// alloc0 has a subview derived from it.  The transitive-use analysis follows
// the subview -> cast chain, extending alloc0's lifetime to the matmul that
// reads the cast.  Since alloc0 is still live when alloc1 is created, no
// reuse is possible.
//
// Intervals (transitive):
//   alloc0: [0, 3]   subview at 1, cast at 2, matmul reads cast at 3
//   alloc1: [2, ret]  cannot reuse alloc0 (0's lastUse=3 >= 2's def=2)
//
// CHECK-LABEL: func.func @subview_extends_lifetime
// CHECK:         %[[A:.*]] = memref.alloc()
// CHECK:         hip.hipblaslt.matmul
// CHECK:         memref.subview %[[A]]
// CHECK:         %[[B:.*]] = memref.alloc()
// CHECK:         memref.cast
// CHECK:         hip.hipblaslt.matmul
// CHECK:         return %[[B]]
func.func @subview_extends_lifetime(
    %handle: !hip.handle,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %sv = memref.subview %alloc0[0, 0, 0][1, 64, 64][1, 1, 1] : memref<2x64x64xf32> to memref<1x64x64xf32, strided<[4096, 64, 1]>>
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  %cast = memref.cast %sv : memref<1x64x64xf32, strided<[4096, 64, 1]>> to memref<1x64x64xf32, strided<[?, ?, ?], offset: ?>>
  hip.hipblaslt.matmul(%handle) ins(%cast, %b : memref<1x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc1 : memref<2x64x64xf32>)
  return %alloc1 : memref<2x64x64xf32>
}

// No allocs to optimize -- pass should be a no-op.
// CHECK-LABEL: func.func @no_allocs
// CHECK:         hip.miopen.softmax
// CHECK:         return
func.func @no_allocs(
    %handle: !hip.handle,
    %in: memref<2x64x64xf32>,
    %out: memref<2x64x64xf32>) {
  hip.miopen.softmax(%handle) ins(%in : memref<2x64x64xf32>) outs(%out : memref<2x64x64xf32>)
  return
}

// alloc0 (memref<64xf32>) and alloc1 (memref<64xf16>) have different element
// types.  canReuse requires matching element types, so reuse is blocked even
// though alloc0 is dead and the byte-sizes would fit (256 bytes >= 128 bytes).
// CHECK-LABEL: func.func @no_reuse_different_element_type
// CHECK:         memref.alloc(){{.*}}: memref<64xf32>
// CHECK:         memref.alloc(){{.*}}: memref<64xf16>
func.func @no_reuse_different_element_type(
    %handle: !hip.handle,
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %c: memref<64xf16>) -> memref<64xf16> {
  %alloc0 = memref.alloc() : memref<64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<64xf32>)
  %alloc1 = memref.alloc() : memref<64xf16>
  hip.miopen.softmax(%handle) ins(%c : memref<64xf16>) outs(%alloc1 : memref<64xf16>)
  return %alloc1 : memref<64xf16>
}
