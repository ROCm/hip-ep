// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the DimSpec builder registry (`shape_interface::populateBuiltin
// DimSpecBuilders`) lets `Shape(hip.transpose(...))` and
// `Shape(hip.<elementwise>(...))` resolve at conversion time without
// requiring every passthrough / elementwise op to manually attach an
// `output_dim_specs` attribute. Without the registry, ShapeToHip
// would notify-match-fail and `onnx.Shape` would survive into bufferize
// with `error: op was not bufferized`.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: rank-preserving permutation with all-static input. Sanity
  // check that the static fold path still applies: `Shape` sees an
  // all-static input (the post-transpose shape `[5,3]` is computable at
  // compile time without consulting the registry) so ShapeToConstant
  // wins. The transpose itself has no surviving users (Shape doesn't
  // read the data) and DCE drops it during the greedy run.
  func.func @test_shape_after_transpose_static(
      %input: tensor<3x5xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_after_transpose_static
    %t = "onnx.Transpose"(%input) {perm = [1, 0]}
        : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %r = "onnx.Shape"(%t) : (tensor<5x3xf32>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK-NOT: onnx.Transpose
    // CHECK: arith.constant dense<[5, 3]> : tensor<2xi64>
    return %r : tensor<2xi64>
  }

  // Test 2: rank-preserving permutation with a dynamic input dim. Here
  // the transpose builder MUST propagate the InputDim leaf for the
  // dynamic dim (without it ShapeToHip cannot resolve and the whole
  // graph fails to lower). Input is `tensor<?x4xf32>` -- transpose with
  // `perm=[1, 0]` produces `tensor<4x?xf32>`. Shape's output dim 0 is
  // Static(4); dim 1 is InputDim(0, 0) via the transpose's permutation
  // (output dim 1 -> input dim perm[1] = 0).
  func.func @test_shape_after_transpose_dyn(
      %input: tensor<?x4xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_after_transpose_dyn
    %t = "onnx.Transpose"(%input) {perm = [1, 0]}
        : (tensor<?x4xf32>) -> tensor<4x?xf32>
    %r = "onnx.Shape"(%t) : (tensor<4x?xf32>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: hip.transpose
    // CHECK: hip.shape(%{{.+}}) ins(%{{.+}} : tensor<4x?xf32>) outs(%{{.+}} : tensor<2xi64>)
    // CHECK-SAME: element_dim_specs = [
    // Static(4) for output dim 0
    // CHECK-SAME{LITERAL}: [array<i64: 0, 4, 0, 0, 0, -1, -1, -1>],
    // InputDim(0, 0) for output dim 1 -- arg 0 has the dynamic dim at index 0
    // CHECK-SAME{LITERAL}: [array<i64: 1, 0, 0, 0, 0, -1, -1, -1>]
    return %r : tensor<2xi64>
  }

  // Test 3: elementwise/broadcast builder. Add of two same-shape
  // dynamic tensors: the broadcast builder picks the first operand
  // whose dim resolves non-broadcast; since both operands have the
  // same `tensor<?xf32>` type, operand 0 (lhs) wins. Shape's only
  // output element is InputDim(0, 0).
  func.func @test_shape_after_add_dyn(
      %a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<1xi64> {
    // CHECK-LABEL: func.func @test_shape_after_add_dyn
    %s = "onnx.Add"(%a, %b) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %r = "onnx.Shape"(%s) : (tensor<?xf32>) -> tensor<1xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: hip.add
    // CHECK: hip.shape(%{{.+}}) ins(%{{.+}} : tensor<?xf32>) outs(%{{.+}} : tensor<1xi64>)
    // CHECK-SAME: element_dim_specs = [
    // CHECK-SAME{LITERAL}: [array<i64: 1, 0, 0, 0, 0, -1, -1, -1>]
    return %r : tensor<1xi64>
  }

  // Test 4: elementwise/broadcast builder with NumPy-style broadcast.
  // Add of `tensor<?x4xf32>` and `tensor<1x4xf32>`: output dim 0 must
  // come from the lhs (rhs is broadcast-1 along axis 0), output dim 1
  // is Static(4) from either side.
  func.func @test_shape_after_broadcast_add(
      %a: tensor<?x4xf32>, %b: tensor<1x4xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_after_broadcast_add
    %s = "onnx.Add"(%a, %b)
        : (tensor<?x4xf32>, tensor<1x4xf32>) -> tensor<?x4xf32>
    %r = "onnx.Shape"(%s) : (tensor<?x4xf32>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: hip.add
    // CHECK: hip.shape
    // CHECK-SAME: element_dim_specs = [
    // dim 0 -> InputDim(0, 0) via lhs (rhs's dim-0 is the static 1 broadcast leg)
    // CHECK-SAME{LITERAL}: [array<i64: 1, 0, 0, 0, 0, -1, -1, -1>],
    // dim 1 -> Static(4); either operand works since both are static-4 and
    // the broadcast builder picks the first non-broadcast operand (lhs).
    // CHECK-SAME{LITERAL}: [array<i64: 0, 4, 0, 0, 0, -1, -1, -1>]
    return %r : tensor<2xi64>
  }

  // Test 5: elementwise unary builder. Sigmoid of dynamic input: output
  // dim 0 -> InputDim(0, 0). The same `buildBroadcastDimSpec` covers
  // unary ops; with a single data operand and no broadcasting, the
  // builder degenerates to "input operand dim k for output dim k".
  func.func @test_shape_after_sigmoid_dyn(
      %x: tensor<?x4xf32>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_shape_after_sigmoid_dyn
    %s = "onnx.Sigmoid"(%x) : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %r = "onnx.Shape"(%s) : (tensor<?x4xf32>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Shape
    // CHECK: hip.sigmoid
    // CHECK: hip.shape
    // CHECK-SAME: element_dim_specs = [
    // CHECK-SAME{LITERAL}: [array<i64: 1, 0, 0, 0, 0, -1, -1, -1>],
    // CHECK-SAME{LITERAL}: [array<i64: 0, 4, 0, 0, 0, -1, -1, -1>]
    return %r : tensor<2xi64>
  }
}
