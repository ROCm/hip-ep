// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// A module with no OpStateOpInterface ops must stay byte-identical to before
// the pass existed: no hipdnn.num_op_state_slots attribute is stamped, so
// --generate-op-state-init / generate-interface remain no-ops.
// See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots | FileCheck %s

// CHECK-NOT: hipdnn.num_op_state_slots
// CHECK-NOT: hip.op_state_slot
module {
  func.func @no_state(%a: memref<4xf32, 1>, %b: memref<4xf32, 1>) {
    return
  }
}
