// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the convert-onnx-to-hip lowering for onnx.Shape:
//
//  * Static-shape inputs collapse to a rank-1 i64 `arith.constant` with
//    the input dim sizes. No HIP op, no runtime symbol.
//  * Dynamic-input Shape lowers to `hip.shape` whose `element_dim_specs`
//    attribute carries one DimSpec per output element. Static dims are
//    encoded as `Static(value)`; dims sourced from a function arg become
//    `InputDim(epIdx, dim)`; dims sourced from a Category-C producer
//    (e.g. NonZero) become `RuntimeSlot(id)`.
//  * The opset-15+ `start` / `end` attributes select a contiguous slice
//    of the shape vector. Negative values count from the back; out-of-
//    range values clamp to `[0, rank]`; `end < start` after clamping is
//    treated as an empty slice (length 0).

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: 3D static input, no slice -> dense<[2, 3, 4]>.
  func.func @test_shape_3d(%input: tensor<2x3x4xf32>) -> tensor<3xi64> {
    // CHECK-LABEL: func.func @test_shape_3d
    %r = "onnx.Shape"(%input) : (tensor<2x3x4xf32>) -> tensor<3xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<[2, 3, 4]> : tensor<3xi64>
    return %r : tensor<3xi64>
  }

  // Test 2: rank-1 input.
  func.func @test_shape_1d(%input: tensor<128xf16>) -> tensor<1xi64> {
    // CHECK-LABEL: func.func @test_shape_1d
    %r = "onnx.Shape"(%input) : (tensor<128xf16>) -> tensor<1xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<128> : tensor<1xi64>
    return %r : tensor<1xi64>
  }

  // Test 3: start / end slice (positive).
  func.func @test_shape_slice_pos(%input: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_slice_pos
    %r = "onnx.Shape"(%input) {start = 1 : si64, end = 3 : si64}
        : (tensor<2x3x4x5xf32>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<[3, 4]> : tensor<2xi64>
    return %r : tensor<2xi64>
  }

  // Test 4: negative end (Python-style trailing slice).
  func.func @test_shape_slice_neg_end(%input: tensor<2x3x4x5xf32>) -> tensor<3xi64> {
    // CHECK-LABEL: func.func @test_shape_slice_neg_end
    %r = "onnx.Shape"(%input) {start = 0 : si64, end = -1 : si64}
        : (tensor<2x3x4x5xf32>) -> tensor<3xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: arith.constant dense<[2, 3, 4]> : tensor<3xi64>
    return %r : tensor<3xi64>
  }

  // Test 5: dynamic input from a function arg lowers to hip.shape. The
  // dynamic dim (axis 0) becomes `InputDim(0, 0)` -- the EP-side input
  // index is 0 (the single tensor input after the !hip.context arg);
  // the static dim (axis 1, value 4) becomes `Static(4)`.
  func.func @test_shape_dyn(%input: tensor<?x4xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_dyn
    %r = "onnx.Shape"(%input) : (tensor<?x4xf32>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<2xi64>
    // CHECK: hip.shape(%{{.+}}) ins(%{{.+}} : tensor<?x4xf32>) outs(%[[INIT]] : tensor<2xi64>)
    // CHECK-SAME: element_dim_specs = [
    // InputDim leaf encoded as [kind=1, value=0, input_index=0, dim_index=0, ...]
    // CHECK-SAME{LITERAL}: [array<i64: 1, 0, 0, 0, 0, -1, -1, -1>],
    // Static(4) leaf encoded as [kind=0, value=4, ...]
    // CHECK-SAME{LITERAL}: [array<i64: 0, 4, 0, 0, 0, -1, -1, -1>]
    return %r : tensor<2xi64>
  }

  // Test 6: dynamic input with start / end slicing keeps only the
  // selected dims. start=1, end=3 drops dim 0 from the output.
  func.func @test_shape_dyn_sliced(%input: tensor<?x?x4x5xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_dyn_sliced
    %r = "onnx.Shape"(%input) {start = 1 : si64, end = 3 : si64}
        : (tensor<?x?x4x5xf32>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: hip.shape(%{{.+}})
    // CHECK-SAME: element_dim_specs = [
    // InputDim(0, 1) for the dynamic axis 1
    // CHECK-SAME{LITERAL}: [array<i64: 1, 0, 0, 1, 0, -1, -1, -1>],
    // Static(4) for axis 2
    // CHECK-SAME{LITERAL}: [array<i64: 0, 4, 0, 0, 0, -1, -1, -1>]
    return %r : tensor<2xi64>
  }
}
