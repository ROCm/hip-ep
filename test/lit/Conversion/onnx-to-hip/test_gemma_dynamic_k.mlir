// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// Gemma attention-shaped contraction with two dynamic batch axes on both
// operands. Both runtime K extents are dynamic; the runtime validates
// that each operand contains either one matrix or every output-batch matrix.
module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.matmul
  // CHECK-SAME: tensor<?x?x256x?xf16>, tensor<?x?x?x256xf16>
  func.func @main_graph(
      %a: tensor<?x?x256x?xf16>,
      %b: tensor<?x?x?x256xf16>) -> tensor<?x?x256x256xf16> {
    %result = "onnx.MatMul"(%a, %b)
      : (tensor<?x?x256x?xf16>, tensor<?x?x?x256xf16>)
        -> tensor<?x?x256x256xf16>
    return %result : tensor<?x?x256x256xf16>
  }
}
