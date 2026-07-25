// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-resolve-extern-constants (standalone pass).
//
// Verifies:
//   - memref.get_global replaced with hip.get_constant
//   - Function signatures are NOT modified (no %_constants arg)
//   - Extern globals are erased
//   - hip.constants_file is preserved for downstream metadata generation
//   - No-op when no extern globals exist
//   - Multiple extern globals with distinct indices
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-resolve-extern-constants %s | FileCheck %s

// ---- Module-level checks ----
// Extern globals should be erased.
// CHECK-NOT: memref.global{{.*}}hip.external_data
// hip.constants_file is preserved for downstream metadata generation.
// CHECK: hip.constants_file

// ===== Single function using one extern global =====
//
// CHECK-LABEL: func.func @single_func_one_global
// CHECK-SAME:    (%[[CTX1:.*]]: !hip.context, %[[A:.*]]: memref<4x4xf32>)
// CHECK:         %[[C0:.*]] = arith.constant 0 : i64
// CHECK:         hip.get_constant(%[[CTX1]], %[[C0]]) : memref<4x4xf32>
// CHECK-NOT:     memref.get_global
// CHECK-NOT:     memref.view

// ===== Function with multiple extern globals at different indices =====
//
// CHECK-LABEL: func.func @multi_global_offsets
// CHECK-SAME:    (%[[CTX2:.*]]: !hip.context, %[[A2:.*]]: memref<4x4xf32>)
// CHECK:         %[[IDX0:.*]] = arith.constant 0 : i64
// CHECK:         hip.get_constant(%[[CTX2]], %[[IDX0]]) : memref<4x4xf32>
// CHECK:         %[[IDX1:.*]] = arith.constant 1 : i64
// CHECK:         hip.get_constant(%[[CTX2]], %[[IDX1]]) : memref<2x2xf32>
// CHECK-NOT:     memref.get_global
// CHECK:         return

// ===== Caller/callee: no transitive argument propagation needed =====
//
// CHECK-LABEL: func.func @callee_uses_global
// CHECK-SAME:    (%[[CC:.*]]: !hip.context, %[[CA:.*]]: memref<4x4xf32>)
// CHECK:         %[[CI:.*]] = arith.constant 0 : i64
// CHECK:         hip.get_constant(%[[CC]], %[[CI]]) : memref<4x4xf32>
// CHECK-NOT:     memref.get_global

// CHECK-LABEL: func.func @caller_passes_constants
// CHECK-SAME:    (%[[TC:.*]]: !hip.context, %[[TA:.*]]: memref<4x4xf32>)
// CHECK:         call @callee_uses_global(%[[TC]], %[[TA]])

// ===== No-op: function without extern globals is unchanged =====
//
// CHECK-LABEL: func.func @no_globals_noop
// CHECK-SAME:    (%{{.*}}: !hip.context, %arg1: memref<4x4xf32>)
// CHECK:         return

module attributes {hip.constants_file = "model.constants.bin"} {

  memref.global "private" @weight_a : memref<4x4xf32> {
    alignment = 64 : i64,
    hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 64 : i64}
  }

  memref.global "private" @weight_b : memref<2x2xf32> {
    alignment = 64 : i64,
    hip.external_data = {index = 1 : i64, offset = 64 : i64, size = 16 : i64}
  }

  func.func @single_func_one_global(%ctx: !hip.context, %arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %w = memref.get_global @weight_a : memref<4x4xf32>
    return %w : memref<4x4xf32>
  }

  func.func @multi_global_offsets(%ctx: !hip.context, %arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %wa = memref.get_global @weight_a : memref<4x4xf32>
    %wb = memref.get_global @weight_b : memref<2x2xf32>
    return %wa : memref<4x4xf32>
  }

  func.func @callee_uses_global(%ctx: !hip.context, %arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %w = memref.get_global @weight_a : memref<4x4xf32>
    return %w : memref<4x4xf32>
  }

  func.func @caller_passes_constants(%ctx: !hip.context, %arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %res = func.call @callee_uses_global(%ctx, %arg0) : (!hip.context, memref<4x4xf32>) -> memref<4x4xf32>
    return %res : memref<4x4xf32>
  }

  func.func @no_globals_noop(%ctx: !hip.context, %arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    return %arg0 : memref<4x4xf32>
  }
}
