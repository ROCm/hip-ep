// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --canonicalize --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --hip-promote-strided-operands --hip-use-output-allocator --split-input-file %s | FileCheck %s

// What this file tests: the `hip.transpose` fold (lib/Dialect/IR/HipTransposeCanonicalize.cpp)
// through the buffer passes that follow it in buildOnnxToHipPipelineTail.
//
// Cancelling a pair hands the consumer the first Transpose's input, which was
// never a Transpose result: it can be a strided view or a graph input. Those
// need the same treatment any other such value gets, a contiguous copy for a
// HIP runtime consumer and a runtime-owned buffer for a graph output.

// A forwarded view must not reach a HIP runtime consumer as a strided pointer.
// CHECK-LABEL: func.func @forward_strided_view
// CHECK-NOT:     hip.transpose
// CHECK:         %[[VIEW:.*]] = memref.subview
// CHECK:         %[[COPY:.*]] = memref.alloc() : memref<1x4x2xf32>
// CHECK:         memref.copy %[[VIEW]], %[[COPY]]
// CHECK:         hip.sigmoid{{.*}}ins(%[[COPY]] : memref<1x4x2xf32>)
// CHECK-NOT:     hip.transpose
func.func @forward_strided_view(%ctx: !hip.context, %x: tensor<1x4x6xf32>) -> tensor<1x4x2xf32> {
  %slice = tensor.extract_slice %x[0, 0, 2] [1, 4, 2] [1, 1, 1] : tensor<1x4x6xf32> to tensor<1x4x2xf32>
  %e0 = tensor.empty() : tensor<1x2x4xf32>
  %0 = hip.transpose(%ctx) ins(%slice : tensor<1x4x2xf32>) outs(%e0 : tensor<1x2x4xf32>) {perm = [0, 2, 1]} : tensor<1x2x4xf32>
  %e1 = tensor.empty() : tensor<1x4x2xf32>
  %1 = hip.transpose(%ctx) ins(%0 : tensor<1x2x4xf32>) outs(%e1 : tensor<1x4x2xf32>) {perm = [0, 2, 1]} : tensor<1x4x2xf32>
  %e2 = tensor.empty() : tensor<1x4x2xf32>
  %out = hip.sigmoid(%ctx) ins(%1 : tensor<1x4x2xf32>) outs(%e2 : tensor<1x4x2xf32>) : tensor<1x4x2xf32>
  return %out : tensor<1x4x2xf32>
}

// -----

// A forwarded graph input still needs a runtime-owned output.
// CHECK-LABEL: func.func @forward_graph_input
// CHECK-SAME:    (%[[CTX:[^, :]+]]: !hip.context, %[[X:[^, :]+]]: memref<2x3xf32>)
// CHECK-NOT:     hip.transpose
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<2x3xf32>
// CHECK:         memref.copy %[[X]], %[[OUT]]
// CHECK:         return %[[OUT]]
func.func @forward_graph_input(%ctx: !hip.context, %x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %e0 = tensor.empty() : tensor<3x2xf32>
  %0 = hip.transpose(%ctx) ins(%x : tensor<2x3xf32>) outs(%e0 : tensor<3x2xf32>) {perm = [1, 0]} : tensor<3x2xf32>
  %e1 = tensor.empty() : tensor<2x3xf32>
  %1 = hip.transpose(%ctx) ins(%0 : tensor<3x2xf32>) outs(%e1 : tensor<2x3xf32>) {perm = [1, 0]} : tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// A composed chain writes straight into the second Transpose's destination,
// which becomes the runtime-owned output; the intermediate is never allocated.
// CHECK-LABEL: func.func @compose_into_graph_output
// CHECK-SAME:    (%[[CTX:[^, :]+]]: !hip.context, %[[X:[^, :]+]]: memref<2x3x4xf32>)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<3x4x2xf32>
// CHECK-NOT:     memref.alloc
// CHECK:         hip.transpose(%[[CTX]]) ins(%[[X]] : memref<2x3x4xf32>) outs(%[[OUT]] : memref<3x4x2xf32>) {perm = [1, 2, 0]}
// CHECK-NOT:     hip.transpose
// CHECK:         return %[[OUT]]
func.func @compose_into_graph_output(%ctx: !hip.context, %x: tensor<2x3x4xf32>) -> tensor<3x4x2xf32> {
  %e0 = tensor.empty() : tensor<3x2x4xf32>
  %0 = hip.transpose(%ctx) ins(%x : tensor<2x3x4xf32>) outs(%e0 : tensor<3x2x4xf32>) {perm = [1, 0, 2]} : tensor<3x2x4xf32>
  %e1 = tensor.empty() : tensor<3x4x2xf32>
  %1 = hip.transpose(%ctx) ins(%0 : tensor<3x2x4xf32>) outs(%e1 : tensor<3x4x2xf32>) {perm = [0, 2, 1]} : tensor<3x4x2xf32>
  return %1 : tensor<3x4x2xf32>
}
