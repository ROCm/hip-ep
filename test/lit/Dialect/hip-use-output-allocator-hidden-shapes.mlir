// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt --hip-use-output-allocator %s 2>&1 | FileCheck %s

// Bufferization can append scalar returns for dynamic result dimensions.
// hipdnn.output_shapes still describes only the real ONNX outputs, so the
// scalar carrier must remain a normal allocation for the later host-scalar
// materialization pass.
module attributes {
  hipdnn.output_shapes = [array<i64: -1, 128>]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: %[[OUT:.*]] = hip.alloc_output(%{{.*}}, %{{.*}}) {out_idx = 0 : i64} : memref<?x128xf32>
  // CHECK: %[[SHAPE:.*]] = memref.alloc() : memref<i64>
  // CHECK-NOT: {out_idx = 1 : i64}
  // CHECK: return %[[OUT]], %[[SHAPE]]
  func.func @main_graph(%ctx: !hip.context, %n: index)
      -> (memref<?x128xf32>, memref<i64>) {
    %out = memref.alloc(%n) : memref<?x128xf32>
    %shape = memref.alloc() : memref<i64>
    return %out, %shape : memref<?x128xf32>, memref<i64>
  }
}
