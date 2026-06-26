// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Aspect G: pipeline non-regression for the new memory-space operand
// constraints on hip.pad.
//
// The constraints accept ranked tensors (pre-bufferization) and memrefs that
// carry no hip memory space (post-bufferization, the form one-shot-bufferize
// produces today). This test drives onnx.Pad through
//   convert-onnx-to-hip -> one-shot-bufferize
// and confirms the op still verifies and reaches memref form — i.e. the
// constraint change did not break the existing lowering path.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s

// convert-onnx-to-hip requires a @main_graph for metadata generation; the
// pilot ops live in their own functions but are still converted + bufferized.
func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  return %arg0 : tensor<1xf32>
}

// CHECK-LABEL: func.func @pad_pipe
// CHECK:         hip.pad
// CHECK-SAME:    memref
func.func @pad_pipe(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"}
    : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
  return %r : tensor<5x6xf32>
}
