// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-pool-allocs (memory pooling into i8 byte buffer).
//===----------------------------------------------------------------------===//

// RUN: %hip-opt --hip-pool-allocs %s | %FileCheck %s

// ===== Static pooling: two non-overlapping f32 allocs =====
//
// Two memref<8x8xf32> (256 bytes each). Both are live at different times.
// With 256-byte alignment, they pack at offsets 0 and 256 -> pool = 512xi8.
//
// CHECK-LABEL: func.func @static_two_allocs
// CHECK:         %[[POOL:.*]] = memref.alloc() : memref<512xi8>
// CHECK-DAG:     %[[OFF0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[OFF1:.*]] = arith.constant 256 : index
// CHECK:         %[[V0:.*]] = memref.view %[[POOL]][%[[OFF0]]][] : memref<512xi8> to memref<8x8xf32>
// CHECK:         hip.hipblaslt.matmul{{.*}}outs(%[[V0]] :
// CHECK:         %[[V1:.*]] = memref.view %[[POOL]][%[[OFF1]]][] : memref<512xi8> to memref<8x8xf32>
// CHECK:         hip.miopen.softmax{{.*}}outs(%[[V1]] :
// CHECK:         return %[[V1]]
func.func @static_two_allocs(
    %handle: !hip.handle,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  return %alloc1 : memref<8x8xf32>
}

// ===== Static pooling: three allocs with overlapping lifetimes =====
//
// alloc0 is used by matmul, alloc1 by softmax (alloc0 as input means alloc0 overlaps alloc1).
// alloc2 reads alloc1 -> alloc0 is dead by then and can share space at a different offset.
// Pool size depends on alignment and overlap analysis.
//
// CHECK-LABEL: func.func @static_three_allocs_overlap
// CHECK:         %[[POOL:.*]] = memref.alloc() : memref<{{[0-9]+}}xi8>
// CHECK-COUNT-3: memref.view %[[POOL]]
// CHECK:         return
func.func @static_three_allocs_overlap(
    %handle: !hip.handle,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  %alloc2 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%handle) ins(%alloc1 : memref<8x8xf32>) outs(%alloc2 : memref<8x8xf32>)
  return %alloc2 : memref<8x8xf32>
}

// ===== Mixed element types: f32 and f16 in same pool =====
//
// memref<8x8xf32> = 256 bytes, memref<8x8xf16> = 128 bytes.
// Both go into the same i8 pool (type-agnostic via memref.view).
//
// CHECK-LABEL: func.func @mixed_element_types
// CHECK:         %[[POOL:.*]] = memref.alloc() : memref<{{[0-9]+}}xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<{{[0-9]+}}xi8> to memref<8x8xf32>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<{{[0-9]+}}xi8> to memref<8x8xf16>
// CHECK:         return
func.func @mixed_element_types(
    %handle: !hip.handle,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %c: memref<8x8xf16>) -> memref<8x8xf16> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf16>
  hip.miopen.softmax(%handle) ins(%c : memref<8x8xf16>) outs(%alloc1 : memref<8x8xf16>)
  return %alloc1 : memref<8x8xf16>
}

// ===== Single alloc: pass is a no-op (need >=2 allocs to pool) =====
//
// CHECK-LABEL: func.func @single_alloc_noop
// CHECK-NOT:     memref<{{[0-9]+}}xi8>
// CHECK:         memref.alloc() : memref<8x8xf32>
// CHECK:         return
func.func @single_alloc_noop(
    %handle: !hip.handle,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  return %alloc0 : memref<8x8xf32>
}

// ===== No allocs: pass is a no-op =====
//
// CHECK-LABEL: func.func @no_allocs_noop
// CHECK-NOT:     memref.alloc
// CHECK:         hip.miopen.softmax
// CHECK:         return
func.func @no_allocs_noop(
    %handle: !hip.handle,
    %in: memref<8x8xf32>,
    %out: memref<8x8xf32>) {
  hip.miopen.softmax(%handle) ins(%in : memref<8x8xf32>) outs(%out : memref<8x8xf32>)
  return
}

// ===== Dynamic pooling: two dynamic allocs with same size SSA value =====
//
// Both memref<?x8xf32> allocs use %n. Pool should be memref<?xi8>.
//
// CHECK-LABEL: func.func @dynamic_two_allocs_same_size
// CHECK:         %[[POOL:.*]] = memref.alloc({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         return
func.func @dynamic_two_allocs_same_size(
    %handle: !hip.handle,
    %a: memref<?x8xf32>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x8xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<?x8xf32>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x8xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x8xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}

// ===== Mixed static + dynamic: f32 static + dynamic allocs in same pool =====
//
// One static memref<8x8xf32> (256 bytes) and one dynamic memref<?x8xf32>.
// Pool is memref<?xi8>.
//
// CHECK-LABEL: func.func @mixed_static_dynamic
// CHECK:         %[[POOL:.*]] = memref.alloc({{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<8x8xf32>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         return
func.func @mixed_static_dynamic(
    %handle: !hip.handle,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %c: memref<?x8xf32>,
    %n: index) -> memref<?x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x8xf32>
  hip.miopen.softmax(%handle) ins(%c : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}

// ===== Alignment: offsets are 256-byte aligned =====
//
// memref<1xf32> = 4 bytes, but should be rounded up to 256-byte alignment.
// Two allocs -> pool should be at least 512xi8 (256 + 256).
//
// CHECK-LABEL: func.func @alignment_256
// CHECK:         %[[POOL:.*]] = memref.alloc() : memref<512xi8>
// CHECK-DAG:     arith.constant 0 : index
// CHECK-DAG:     arith.constant 256 : index
// CHECK:         memref.view %[[POOL]]
// CHECK:         memref.view %[[POOL]]
// CHECK:         return
func.func @alignment_256(
    %handle: !hip.handle,
    %in: memref<1xf32>) -> memref<1xf32> {
  %alloc0 = memref.alloc() : memref<1xf32>
  hip.miopen.softmax(%handle) ins(%in : memref<1xf32>) outs(%alloc0 : memref<1xf32>)
  %alloc1 = memref.alloc() : memref<1xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<1xf32>) outs(%alloc1 : memref<1xf32>)
  return %alloc1 : memref<1xf32>
}
