// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: hip-resolve-extern-constants with inter-function calls.
//
// Verifies that when a callee's signature gains %_constants, all call
// sites are rewritten to pass the argument through, and transitive
// callers also receive the new argument.
//===----------------------------------------------------------------------===//

// RUN: %hip-mlir-opt --hip-resolve-extern-constants %s | %FileCheck %s

// The extern globals should be erased.
// CHECK-NOT: memref.global{{.*}}hip.external_data

// callee gets %_constants (3rd arg) and uses memref.view instead of get_global.
// CHECK-LABEL: func.func @callee
// CHECK-SAME:  %arg0: memref<4x4xf32>
// CHECK-SAME:  %arg1: memref<4x4xf32>
// CHECK-SAME:  %arg2: memref<?xi8>
// CHECK:         memref.view %arg2
// CHECK-NOT:     memref.get_global

// caller gets %_constants (2nd arg) and passes it to callee.
// CHECK-LABEL: func.func @caller
// CHECK-SAME:  %arg0: memref<4x4xf32>
// CHECK-SAME:  %arg1: memref<?xi8>
// CHECK:         call @callee(%arg0, %arg0, %arg1)

module attributes {hip.constants_file = "model.constants.bin"} {
  memref.global "private" @weight0 : memref<4x4xf32> {
    alignment = 64 : i64,
    hip.external_data = {offset = 0 : i64, size = 64 : i64}
  }

  func.func @callee(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) -> memref<4x4xf32> {
    %w = memref.get_global @weight0 : memref<4x4xf32>
    return %w : memref<4x4xf32>
  }

  func.func @caller(%arg0: memref<4x4xf32>) -> memref<4x4xf32> {
    %res = func.call @callee(%arg0, %arg0) : (memref<4x4xf32>, memref<4x4xf32>) -> memref<4x4xf32>
    return %res : memref<4x4xf32>
  }
}
