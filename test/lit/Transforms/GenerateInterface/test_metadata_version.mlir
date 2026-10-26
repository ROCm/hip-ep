// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: HipModelMetaInfo.version follows the constants it describes.
//
// A row-padded constant (hip-pad-matmul-nbits-weights, streaming mode) has a
// ConstantInfo.size that is the padded footprint, not the source byte count,
// so its metadata is version RowPadded (2), which a runtime that does not know
// row_bytes / row_pitch rejects. Metadata without one stays version Dense (1).
// Checked on the JSON mirror, which is generated from the same native struct
// as __metadata_blob.
// ============================================================================

// RUN: hip-mlir-opt --hip-to-llvm-pipeline --split-input-file %s 2>&1 | FileCheck %s

// CHECK-LABEL: llvm.mlir.global internal constant @__metadata_json
// CHECK-NOT:   row_pitch
// CHECK-NOT:   version{{[^,]*}}2
// CHECK:       llvm.func @inference_get_metadata_json
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: 8>],
  hipdnn.input_element_sizes = array<i64: 4>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 8>],
  hipdnn.output_element_sizes = array<i64: 4>,
  hipdnn.pool_size = 0 : i64,
  hipdnn.buffer_offsets = [],
  hipdnn.buffer_count = 0 : i64,
  hipdnn.constant_sizes = array<i64: 4096>,
  hipdnn.constant_offsets = array<i64: 0>
} {
  func.func @main_graph(%ctx: !hip.context,
                        %in: memref<8xf32>) -> memref<8xf32> {
    %out = hip.alloc_output(%ctx) {out_idx = 0 : i64} : memref<8xf32>
    return %out : memref<8xf32>
  }
}

// -----

// CHECK-LABEL: llvm.mlir.global internal constant @__metadata_json
// CHECK-SAME:  version{{[^,]*}}2
// CHECK-SAME:  row_bytes{{[^,]*}}2048
// CHECK-SAME:  row_pitch{{[^,]*}}2304
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: 8>],
  hipdnn.input_element_sizes = array<i64: 4>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 8>],
  hipdnn.output_element_sizes = array<i64: 4>,
  hipdnn.pool_size = 0 : i64,
  hipdnn.buffer_offsets = [],
  hipdnn.buffer_count = 0 : i64,
  hipdnn.constant_sizes = array<i64: 9437184>,
  hipdnn.constant_offsets = array<i64: 0>,
  hipdnn.constant_row_bytes = array<i64: 2048>,
  hipdnn.constant_row_pitches = array<i64: 2304>
} {
  func.func @main_graph(%ctx: !hip.context,
                        %in: memref<8xf32>) -> memref<8xf32> {
    %out = hip.alloc_output(%ctx) {out_idx = 0 : i64} : memref<8xf32>
    return %out : memref<8xf32>
  }
}
