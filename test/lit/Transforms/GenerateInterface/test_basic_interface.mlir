// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: GenerateInterface pass creates C-ABI wrapper functions.
//
// Input: Pre-lowered LLVM IR with @main_graph(ptr, ptr, ptr) -> i32 and
// all required hipdnn.* module attributes.  The earlier lowering passes in
// hip-to-llvm-pipeline are no-ops on already-lowered IR, so GenerateInterface
// runs directly on the input below.
//
// Verifies:
//   1. Four wrapper functions are generated.
//   2. inference_init calls hipdnn_ep_state_init_with_fs.
//   3. inference_compute calls tensor prepare / main_graph / finalize.
//   4. inference_cleanup calls hipdnn_ep_state_cleanup.
//   5. Metadata blob is embedded as a global constant.
// ============================================================================

// RUN: hip-mlir-opt --hip-to-llvm-pipeline %s 2>&1 | FileCheck %s

// --- Metadata blob embedded ---
// CHECK: llvm.mlir.global internal constant @__metadata_blob

// --- inference_init calls state init ---
// CHECK-LABEL: llvm.func @inference_init
// CHECK-SAME:  -> i32
// CHECK:   llvm.call @hipdnn_ep_state_init_with_fs

// --- inference_compute calls tensor helpers and main_graph ---
// CHECK-LABEL: llvm.func @inference_compute
// CHECK-SAME:  -> i32
// CHECK:   llvm.call @hipdnn_ep_tensor_prepare_input
// CHECK:   llvm.call @hipdnn_ep_tensor_prepare_output
// CHECK:   llvm.call @main_graph
// CHECK:   llvm.call @hipdnn_ep_tensor_finalize_output

// --- inference_cleanup calls state cleanup ---
// CHECK-LABEL: llvm.func @inference_cleanup
// CHECK-SAME:  -> i32
// CHECK:   llvm.call @hipdnn_ep_state_cleanup

// --- inference_get_metadata_json returns metadata pointer ---
// CHECK-LABEL: llvm.func @inference_get_metadata_json
// CHECK:   llvm.mlir.addressof @__metadata_json

module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: 8>],
  hipdnn.input_element_sizes = array<i64: 4>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 8>],
  hipdnn.output_element_sizes = array<i64: 4>,
  hipdnn.pool_size = 0 : i64,
  hipdnn.buffer_offsets = [],
  hipdnn.buffer_count = 0 : i64
} {
  llvm.func @main_graph(%ctx: !llvm.ptr, %inputs: !llvm.ptr,
                        %outputs: !llvm.ptr) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
}
