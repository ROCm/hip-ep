// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: hip-resolve-extern-constants with inter-function calls.
//
// Verifies that extern memref.global ops with hip.external_data are
// replaced by hip.get_constant ops using the !hip.context from arg 0.
// Function signatures should NOT be modified (no %_constants arg).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-resolve-extern-constants %s | FileCheck %s

// The extern globals should be erased.
// CHECK-NOT: memref.global{{.*}}hip.external_data

// callee uses hip.get_constant instead of memref.get_global.
// CHECK-LABEL: func.func @callee
// CHECK-SAME:  %[[CTX:.*]]: !hip.context, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>
// CHECK-NOT:   memref.get_global
// CHECK:       %[[IDX:.*]] = arith.constant 0 : i64
// CHECK:       hip.get_constant(%[[CTX]], %[[IDX]]) : memref<4x4xf32>
// CHECK-NOT:   memref.view

// caller signature is unchanged (no %_constants arg).
// CHECK-LABEL: func.func @caller
// CHECK-SAME:  %[[CTX2:.*]]: !hip.context, %arg1: memref<4x4xf32>
// CHECK:       call @callee

module attributes {hip.constants_file = "model.constants.bin"} {
  memref.global "private" @weight0 : memref<4x4xf32> {
    alignment = 64 : i64,
    hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 64 : i64}
  }

  func.func @callee(%ctx: !hip.context, %arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) -> memref<4x4xf32> {
    %w = memref.get_global @weight0 : memref<4x4xf32>
    return %w : memref<4x4xf32>
  }

  func.func @caller(%ctx: !hip.context, %arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %res = func.call @callee(%ctx, %arg0, %arg0) : (!hip.context, memref<4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32>
    return %res : memref<4x4xf32>
  }
}
