// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: the smallest graph the hipsr pipeline has to handle.
//
// One onnx.MatMul over two graph inputs, every extent static, no constants.
// This is the baseline the pipeline is grown against: each pass added to
// buildHipsrPipeline extends the CHECK block below, so the expected output
// stays the whole function rather than a set of spot checks. Written with
// CHECK-NEXT throughout, so an unexpected operation breaks the chain instead
// of slipping through.
//
// The operands are deliberately rank 2. MatMul's shape recipe splits the batch
// dimensions off both operands and broadcasts them, so a rank-2 pair drives
// that path with empty batch shapes -- the cheapest form of the shape graph.
// Static extents keep every buffer size a constant, and leaving the weight as
// a graph input rather than an initializer keeps hipsr.constant out of the
// picture. Higher-rank, dynamic-extent, and constant-weight variants are
// separate cases to add once this one reaches the end of the pipeline.
//
// The pipeline currently inserts the runtime context argument and stops, so
// the ONNX operation is still expected in the output.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:      %[[A:.+]]: tensor<128x4096xf16> {onnx.name = "a"},
// CHECK-SAME:      %[[B:.+]]: tensor<4096x1024xf16> {onnx.name = "b"})
// CHECK-SAME:      -> (tensor<128x1024xf16> {onnx.name = "y"})
// CHECK-SAME:      attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:    %[[Y:.+]] = "onnx.MatMul"(%[[A]], %[[B]]) : (tensor<128x4096xf16>, tensor<4096x1024xf16>) -> tensor<128x1024xf16>
// CHECK-NEXT:    return %[[Y]] : tensor<128x1024xf16>
// CHECK-NEXT:  }
func.func @main_graph(%a: tensor<128x4096xf16> {onnx.name = "a"},
                      %b: tensor<4096x1024xf16> {onnx.name = "b"})
    -> (tensor<128x1024xf16> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<128x4096xf16>, tensor<4096x1024xf16>)
      -> tensor<128x1024xf16>
  return %0 : tensor<128x1024xf16>
}
