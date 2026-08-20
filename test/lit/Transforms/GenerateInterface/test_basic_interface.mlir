// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: GenerateInterface pass creates C-ABI wrapper functions.
//
// Input: HIP dialect IR with an output-allocator @main_graph using
// !hip.context and memref types. Outputs are allocated in-graph via
// hip.alloc_output (2-arg ABI); there are no output out-params.
// The hip-to-llvm-pipeline lowers this through:
//   1. convert-hip-to-llvm (func/memref → LLVM, transformMainFunction wraps
//      the expanded memref signature to the 2-pointer (ctx, inputs) C ABI)
//   2. generate-interface (creates ABI query + init/compute/cleanup wrappers)
//
// Verifies:
//   1. Five wrapper functions are generated.
//   2. inference_init calls hipdnn_ep_state_init_with_fs.
//   3. inference_compute stages inputs and calls main_graph (outputs
//      allocated in-graph).
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

// --- inference_compute stages inputs and calls main_graph (2-arg ABI) ---
// CHECK-LABEL: llvm.func @inference_compute
// CHECK-SAME:  -> i32
// CHECK:   %[[ZERO64:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:   %[[ONE64:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:   %[[PREPARED_COUNT:.*]] = llvm.alloca %[[ONE64]] x i64
// CHECK:   llvm.store %[[ZERO64]], %[[PREPARED_COUNT]]
// CHECK:   %[[BUFFER_WORDS:.*]] = llvm.call @hipdnn_ep_tensor_buffer_storage_words() : () -> i64
// CHECK:   llvm.alloca %[[BUFFER_WORDS]] x i64
// CHECK-COUNT-3: llvm.call @hipdnn_ep_tensor_buffer_construct
// CHECK:   llvm.call @hipdnn_ep_tensor_prepare_input
// CHECK:   %[[ERROR_COUNT:.*]] = llvm.load %[[PREPARED_COUNT]]
// CHECK:   llvm.call @hipdnn_ep_tensor_free_inputs({{.*}}, {{.*}}, %[[ERROR_COUNT]])
// CHECK:   %[[PREPARED1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:   llvm.store %[[PREPARED1]], %[[PREPARED_COUNT]]
// CHECK:   llvm.call @hipdnn_ep_tensor_prepare_input
// CHECK:   %[[PREPARED2:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:   llvm.store %[[PREPARED2]], %[[PREPARED_COUNT]]
// CHECK:   llvm.call @hipdnn_ep_tensor_prepare_input
// CHECK:   %[[PREPARED3:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK:   llvm.store %[[PREPARED3]], %[[PREPARED_COUNT]]
// CHECK:   llvm.call @main_graph
// CHECK:   llvm.call @hipdnn_ep_stream_sync
// CHECK:   llvm.call @hipdnn_ep_state_read_and_clear_error_flag
// CHECK:   llvm.call @hipdnn_ep_tensor_free_inputs

// --- inference_cleanup calls state cleanup ---
// CHECK-LABEL: llvm.func @inference_cleanup
// CHECK-SAME:  -> i32
// CHECK:   llvm.call @hipdnn_ep_state_cleanup

// --- inference_get_metadata_json returns metadata pointer ---
// CHECK-LABEL: llvm.func @inference_get_metadata_json
// CHECK:   llvm.mlir.addressof @__metadata_json

// --- loader validates this tagged token before binding any other function ---
// CHECK-LABEL: llvm.func @inference_get_artifact_abi() -> i64
// CHECK:   llvm.mlir.constant(5208782684133785602 : i64)
// CHECK:   llvm.return

module attributes {
  hipdnn.input_count = 3 : i64,
  hipdnn.input_shapes = [array<i64: 8>, array<i64: 8>, array<i64: 8>],
  hipdnn.input_element_sizes = array<i64: 4, 4, 4>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 8>],
  hipdnn.output_element_sizes = array<i64: 4>,
  hipdnn.pool_size = 0 : i64,
  hipdnn.buffer_offsets = [],
  hipdnn.buffer_count = 0 : i64
} {
  func.func @main_graph(%ctx: !hip.context,
                        %in0: memref<8xf32>,
                        %in1: memref<8xf32>,
                        %in2: memref<8xf32>) -> memref<8xf32> {
    %out = hip.alloc_output(%ctx) {out_idx = 0 : i64} : memref<8xf32>
    return %out : memref<8xf32>
  }
}
