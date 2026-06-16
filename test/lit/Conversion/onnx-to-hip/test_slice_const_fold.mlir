// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a compile-time-constant onnx.Slice folds to a zero-copy
// tensor.extract_slice EVEN WHEN the slice's starts/ends/axes/steps come from
// onnx.Constant ops that constant externalization would strip the inline value
// from.
//
// SliceDecompose is registered in the PRE-lowering phase (before
// lowerOnnxConstants) precisely so it reads the inline onnx.Constant values.
// With externalize-min-num-elements=1 every constant -- including these tiny
// index tensors -- is externalized to a memref.global with no inline value;
// run AFTER externalization the fold could not read them and the slice would
// fall back to the runtime hip.slice D2H readback. Folding pre-externalization
// eliminates that readback at the compiler layer.
//
// Both RUN lines must produce the same extract_slice (no hip.slice / onnx.Slice):
//   1. default options
//   2. externalize-min-num-elements=1 (the case this fix targets; the dead
//      externalized globals are harmless and DCE away)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip=externalize-min-num-elements=1 | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Constant slice via onnx.Constant operands (the externalizable form).
  // Folds to tensor.extract_slice through the pre-lowering SliceDecompose.
  func.func @test_slice_const_fold(%input: tensor<4x6xf32>) -> tensor<2x6xf32> {
    // CHECK-LABEL: func.func @test_slice_const_fold
    %starts = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends   = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes   = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps  = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<4x6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<2x6xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: hip.slice
    // CHECK: tensor.extract_slice {{.*}}[1, 0] [2, 6] [1, 1]
    return %r : tensor<2x6xf32>
  }
}
