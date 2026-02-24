// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-lower-allocs (memref.alloc -> hip.alloc/free).
//
// Ownership convention verified by these tests:
//   - Every memref.alloc becomes hip.alloc (device memory via hipMalloc).
//   - Returned buffers are caller-owned: no hip.free emitted.
//   - Non-returned buffers get hip.free before hip.destroy_handle (the handle
//     holds the HIP context needed by hipFree).
//===----------------------------------------------------------------------===//

// RUN: %hip-opt --hip-lower-allocs %s 2>&1 | %FileCheck %s

// Two static allocs: alloc0 is not returned -> hip.free before destroy_handle.
// alloc1 is returned -> no hip.free (caller-owned).
// CHECK-LABEL: func.func @static_lower
// CHECK:         %[[H:.*]] = hip.create_handle()
// CHECK:         %[[A:.*]] = hip.alloc(%[[H]]) : memref<2x64x64xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         %[[B:.*]] = hip.alloc(%[[H]]) : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[H]], %[[A]]) : memref<2x64x64xf32>
// CHECK:         hip.destroy_handle(%[[H]])
// CHECK:         return %[[B]]
func.func @static_lower(
    %a: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %handle = hip.create_handle() : !hip.handle
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<2x64x64xf32>) outs(%alloc1 : memref<2x64x64xf32>)
  hip.destroy_handle(%handle) : !hip.handle
  return %alloc1 : memref<2x64x64xf32>
}

// One dynamic memref<?x64xf32> alloc sized by %n: the dynamic size is forwarded
// to hip.alloc. Since the buffer is returned, no hip.free is inserted.
// CHECK-LABEL: func.func @dynamic_lower
// CHECK:         %[[H:.*]] = hip.create_handle()
// CHECK:         hip.alloc(%[[H]], %arg2) : memref<?x64xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         hip.destroy_handle(%[[H]])
// CHECK-NOT:     hip.free
// CHECK:         return
func.func @dynamic_lower(
    %a: memref<?x64xf32>,
    %b: memref<64x64xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x64xf32> {
  %handle = hip.create_handle() : !hip.handle
  %alloc0 = memref.alloc(%n) : memref<?x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<?x64xf32>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x64xf32>)
  hip.destroy_handle(%handle) : !hip.handle
  return %alloc0 : memref<?x64xf32>
}

// No hip.create_handle: pass is a no-op, memref.alloc is preserved.
// CHECK-LABEL: func.func @no_handle_noop
// CHECK:         memref.alloc()
// CHECK-NOT:     hip.alloc
// CHECK:         return
func.func @no_handle_noop(%a: memref<2x64x64xf32>) -> memref<2x64x64xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  memref.copy %a, %alloc : memref<2x64x64xf32> to memref<2x64x64xf32>
  return %alloc : memref<2x64x64xf32>
}

// Three memref<2x64x64xf32> allocs: alloc0 and alloc1 are not returned, so
// both get hip.free after their last use. alloc2 is returned.
// CHECK-LABEL: func.func @multiple_frees
// CHECK:         %[[H:.*]] = hip.create_handle()
// CHECK:         %[[A:.*]] = hip.alloc(%[[H]]) : memref<2x64x64xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         %[[B:.*]] = hip.alloc(%[[H]]) : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[H]], %[[A]]) : memref<2x64x64xf32>
// CHECK:         %[[C:.*]] = hip.alloc(%[[H]]) : memref<2x64x64xf32>
// CHECK:         hip.miopen.softmax
// CHECK:         hip.free(%[[H]], %[[B]]) : memref<2x64x64xf32>
// CHECK:         hip.destroy_handle(%[[H]])
// CHECK:         return %[[C]]
func.func @multiple_frees(
    %input: memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>,
    %w: memref<64x64xf32, strided<[?, ?], offset: ?>>) -> memref<2x64x64xf32> {
  %handle = hip.create_handle() : !hip.handle
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.hipblaslt.matmul(%handle) ins(%input, %w : memref<2x64x64xf32, strided<[?, ?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<2x64x64xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<2x64x64xf32>) outs(%alloc1 : memref<2x64x64xf32>)
  %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<2x64x64xf32>
  hip.miopen.softmax(%handle) ins(%alloc1 : memref<2x64x64xf32>) outs(%alloc2 : memref<2x64x64xf32>)
  hip.destroy_handle(%handle) : !hip.handle
  return %alloc2 : memref<2x64x64xf32>
}
