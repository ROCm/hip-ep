// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-resolve-extern-constants (standalone pass).
//
// Verifies:
//   - memref.get_global replaced with memref.view into %_constants arg
//   - Function signatures gain %_constants : memref<?xi8>
//   - Call sites forward the constants buffer
//   - Extern globals are erased
//   - Module hip.constants_file attribute is stripped
//   - No-op when no extern globals exist
//   - Multiple extern globals with distinct offsets
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-resolve-extern-constants %s | FileCheck %s

// ---- Module-level checks ----
// Extern globals should be erased.
// CHECK-NOT: memref.global{{.*}}hip.external_data
// hip.constants_file should be stripped.
// CHECK-NOT: hip.constants_file

// ===== Single function using one extern global =====
//
// CHECK-LABEL: func.func @single_func_one_global
// CHECK-SAME:    (%[[A:.*]]: memref<4x4xf32>, %[[CONST:.*]]: memref<?xi8>)
// CHECK:         %[[OFF:.*]] = arith.constant 0 : index
// CHECK:         %[[VIEW:.*]] = memref.view %[[CONST]][%[[OFF]]][]
// CHECK-SAME:      : memref<?xi8> to memref<4x4xf32>
// CHECK-NOT:     memref.get_global
// CHECK:         return %[[VIEW]]

// ===== Function with multiple extern globals at different offsets =====
//
// CHECK-LABEL: func.func @multi_global_offsets
// CHECK-SAME:    (%[[A2:.*]]: memref<4x4xf32>, %[[CONST2:.*]]: memref<?xi8>)
// CHECK:         memref.view %[[CONST2]]{{.*}} : memref<?xi8> to memref<4x4xf32>
// CHECK:         memref.view %[[CONST2]]{{.*}} : memref<?xi8> to memref<2x2xf32>
// CHECK-NOT:     memref.get_global
// CHECK:         return

// ===== Caller/callee: transitive propagation of constants arg =====
//
// CHECK-LABEL: func.func @callee_uses_global
// CHECK-SAME:    (%[[CA:.*]]: memref<4x4xf32>, %[[CC:.*]]: memref<?xi8>)
// CHECK:         memref.view %[[CC]]
// CHECK-NOT:     memref.get_global

// CHECK-LABEL: func.func @caller_passes_constants
// CHECK-SAME:    (%[[TA:.*]]: memref<4x4xf32>, %[[TC:.*]]: memref<?xi8>)
// CHECK:         call @callee_uses_global(%[[TA]], %[[TC]])

// ===== No-op: function without extern globals is unchanged =====
//
// CHECK-LABEL: func.func @no_globals_noop
// CHECK-SAME:    (%arg0: memref<4x4xf32>)
// CHECK-NOT:     memref<?xi8>
// CHECK:         return

module attributes {hip.constants_file = "model.constants.bin"} {

  memref.global "private" @weight_a : memref<4x4xf32> {
    alignment = 64 : i64,
    hip.external_data = {offset = 0 : i64, size = 64 : i64}
  }

  memref.global "private" @weight_b : memref<2x2xf32> {
    alignment = 64 : i64,
    hip.external_data = {offset = 64 : i64, size = 16 : i64}
  }

  func.func @single_func_one_global(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %w = memref.get_global @weight_a : memref<4x4xf32>
    return %w : memref<4x4xf32>
  }

  func.func @multi_global_offsets(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %wa = memref.get_global @weight_a : memref<4x4xf32>
    %wb = memref.get_global @weight_b : memref<2x2xf32>
    return %wa : memref<4x4xf32>
  }

  func.func @callee_uses_global(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %w = memref.get_global @weight_a : memref<4x4xf32>
    return %w : memref<4x4xf32>
  }

  func.func @caller_passes_constants(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %res = func.call @callee_uses_global(%arg0) : (memref<4x4xf32>) -> memref<4x4xf32>
    return %res : memref<4x4xf32>
  }

  func.func @no_globals_noop(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    return %arg0 : memref<4x4xf32>
  }
}
