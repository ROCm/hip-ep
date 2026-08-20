// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: onnx.Conv -> hip.conv -> bufferize -> hip-pool-allocs
//
// Two back-to-back convolutions.  After the full pipeline:
//   1. convert-onnx-to-hip:  onnx.Conv -> tensor.empty + hip.conv
//   2. one-shot-bufferize:   tensor -> memref, tensor.empty -> memref.alloc
//   3. hip-pool-allocs:      2 memref.alloc -> hip.get_pool + memref.view
//
// conv1 output (1x16x8x8xf32 = 4096 bytes) is consumed by conv2, so
// lifetimes overlap and both buffers need separate offsets in the pool.
// Pool = 4096 + 4096 = 8192 bytes, offsets 0 and 4096.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" --hip-pool-allocs | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         %[[SIZE:.*]] = arith.constant 8192 : index
// CHECK:         %[[POOL:.*]] = hip.get_pool(%[[CTX]], %[[SIZE]]){{.*}} : memref<?xi8>
// CHECK-DAG:     %[[OFF0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[OFF1:.*]] = arith.constant 4096 : index
// CHECK:         memref.view %[[POOL]][%[[OFF0]]][] : memref<?xi8> to memref<1x16x8x8xf32>
// CHECK:         hip.conv
// CHECK:         memref.view %[[POOL]][%[[OFF1]]][] : memref<?xi8> to memref<1x16x8x8xf32>
// CHECK:         hip.conv
// CHECK-NOT:     memref.alloc
// CHECK:         return

module {
  func.func @main_graph(
      %input: tensor<1x3x8x8xf32>,
      %w1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>,
      %w2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>) -> tensor<1x16x8x8xf32> {
    %conv1 = "onnx.Conv"(%input, %w1, %b1) {
      kernel_shape = [3, 3], strides = [1, 1],
      pads = [1, 1, 1, 1], dilations = [1, 1], group = 1 : i64
    } : (tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x8x8xf32>
    %conv2 = "onnx.Conv"(%conv1, %w2, %b2) {
      kernel_shape = [3, 3], strides = [1, 1],
      pads = [1, 1, 1, 1], dilations = [1, 1], group = 1 : i64
    } : (tensor<1x16x8x8xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>) -> tensor<1x16x8x8xf32>
    return %conv2 : tensor<1x16x8x8xf32>
  }
}
