// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the convert-onnx-to-hip lowering for onnx.Concat.
//
// The only Concat form currently handled is the "shape vector concat"
// produced by dyn-shape graphs upstream of Reshape / Expand /
// ConstantOfShape: axis=0, rank-1 i64 result, every operand is rank-0
// or rank-1 with a statically known length. Those lower to a single
// `tensor.from_elements` whose elements are `tensor.extract`s of the
// per-operand i64 scalars. The downstream consumer can then
// `tensor.extract` the per-dim sizes back out via the standard
// from_elements / extract canonicalisation.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: classic Reshape shape-vector pattern: two rank-1[1] i64
  // operands (e.g. Gather(Shape(K), 0) and Gather(Shape(K), 1)) become
  // a rank-1[2] from_elements. Both elements get extract+from_elements
  // canonicalised, so the final IR has 2 extracts on the operands and
  // one from_elements.
  func.func @test_concat_two_rank1_i64(
      %a: tensor<1xi64>, %b: tensor<1xi64>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_concat_two_rank1_i64
    %r = "onnx.Concat"(%a, %b) {axis = 0 : si64}
        : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Concat
    // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    // CHECK-DAG: %[[EA:.+]] = tensor.extract %{{.+}}[%[[C0]]] : tensor<1xi64>
    // CHECK-DAG: %[[EB:.+]] = tensor.extract %{{.+}}[%[[C0]]] : tensor<1xi64>
    // CHECK: tensor.from_elements %{{.+}}, %{{.+}} : tensor<2xi64>
    return %r : tensor<2xi64>
  }

  // Test 2: mix of rank-0 and rank-1[1] operands (rank-0 happens when
  // upstream Gather produced a scalar). Each contributes one element.
  func.func @test_concat_mixed_rank(
      %a: tensor<i64>, %b: tensor<1xi64>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_concat_mixed_rank
    %r = "onnx.Concat"(%a, %b) {axis = 0 : si64}
        : (tensor<i64>, tensor<1xi64>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Concat
    // CHECK-DAG: tensor.extract %{{.+}}[] : tensor<i64>
    // CHECK-DAG: tensor.extract %{{.+}}[%{{.+}}] : tensor<1xi64>
    // CHECK: tensor.from_elements %{{.+}}, %{{.+}} : tensor<2xi64>
    return %r : tensor<2xi64>
  }

  // Test 3: longer concat (four 1-elem operands -> rank-1[4]) for the
  // ConstantOfShape pattern used by causal mask generation.
  func.func @test_concat_four_rank1(
      %a: tensor<1xi64>, %b: tensor<1xi64>,
      %c: tensor<1xi64>, %d: tensor<1xi64>) -> tensor<4xi64> {
    // CHECK-LABEL: func.func @test_concat_four_rank1
    %r = "onnx.Concat"(%a, %b, %c, %d) {axis = 0 : si64}
        : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
        -> tensor<4xi64>
    // CHECK-NOT: onnx.Concat
    // CHECK: tensor.from_elements %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<4xi64>
    return %r : tensor<4xi64>
  }

  // Test 4: negative axis: -1 on a rank-1 result == 0, still legal.
  func.func @test_concat_neg_axis(
      %a: tensor<1xi64>, %b: tensor<1xi64>) -> tensor<2xi64> {
    // CHECK-LABEL: func.func @test_concat_neg_axis
    %r = "onnx.Concat"(%a, %b) {axis = -1 : si64}
        : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    // CHECK-NOT: onnx.Concat
    // CHECK: tensor.from_elements %{{.+}}, %{{.+}} : tensor<2xi64>
    return %r : tensor<2xi64>
  }
}
