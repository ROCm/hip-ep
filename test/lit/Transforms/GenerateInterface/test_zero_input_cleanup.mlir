// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-to-llvm-pipeline %s | FileCheck %s

// A zero-input graph constructs and prepares no TensorBuffer records. The
// common cleanup call receives the zero prepared count and must not dereference
// its zero-length pointer array.
// CHECK-LABEL: llvm.func @inference_compute
// CHECK: %[[PREPARED_ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.store %[[PREPARED_ZERO]], %{{.*}}
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %{{.*}} = llvm.alloca %[[ZERO]] x !llvm.ptr
// CHECK-NOT: llvm.call @hipdnn_ep_tensor_buffer_construct
// CHECK-NOT: llvm.call @hipdnn_ep_tensor_prepare_input
// CHECK: llvm.call @main_graph
// CHECK: llvm.call @hipdnn_ep_tensor_free_inputs({{.*}}, {{.*}}, %[[ZERO]])

module attributes {
  hipdnn.input_count = 0 : i64,
  hipdnn.input_shapes = [],
  hipdnn.input_element_sizes = array<i64>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 1>],
  hipdnn.output_element_sizes = array<i64: 4>,
  hipdnn.pool_size = 0 : i64,
  hipdnn.buffer_offsets = [],
  hipdnn.buffer_count = 0 : i64
} {
  func.func @main_graph(%ctx: !hip.context) -> memref<1xf32> {
    %out = hip.alloc_output(%ctx) {out_idx = 0 : i64} : memref<1xf32>
    return %out : memref<1xf32>
  }
}
