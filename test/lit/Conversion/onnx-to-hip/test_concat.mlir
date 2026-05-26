// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the ONNX Concat lowering: every onnx.Concat is rewritten to a
// `tensor.empty` + N x `tensor.insert_slice` sequence. No `hip.*` op is
// produced; bufferization later folds the inserts into `memref.subview`
// copies against a pooled output buffer.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: classic 2-input concat along axis 0 (all-static shapes). The
  // running axis offset for input 1 is the static dim of input 0.
  func.func @test_concat_axis0_static(%a: tensor<2x4xf32>,
                                       %b: tensor<3x4xf32>) -> tensor<5x4xf32> {
    // CHECK-LABEL: func.func @test_concat_axis0_static
    %r = "onnx.Concat"(%a, %b) {axis = 0 : si64}
        : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<5x4xf32>

    // CHECK-NOT: onnx.Concat
    // CHECK-NOT: hip.concat
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<5x4xf32>
    // CHECK: %[[I0:.*]] = tensor.insert_slice %{{.*}} into %[[INIT]][0, 0] [2, 4] [1, 1]
    // CHECK: tensor.insert_slice %{{.*}} into %[[I0]][2, 0] [3, 4] [1, 1]

    return %r : tensor<5x4xf32>
  }

  // Test 2: 3-input concat along an inner axis with a negative axis attr
  // (axis = -1 normalises to rank-1 = 1).
  func.func @test_concat_axis_neg(%a: tensor<2x3xf32>,
                                   %b: tensor<2x4xf32>,
                                   %c: tensor<2x5xf32>) -> tensor<2x12xf32> {
    // CHECK-LABEL: func.func @test_concat_axis_neg
    %r = "onnx.Concat"(%a, %b, %c) {axis = -1 : si64}
        : (tensor<2x3xf32>, tensor<2x4xf32>, tensor<2x5xf32>) -> tensor<2x12xf32>

    // CHECK-NOT: onnx.Concat
    // Offsets along axis 1: 0, 3, 7. Untouched axis 0 stays at 0.
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x12xf32>
    // CHECK: %[[I0:.*]] = tensor.insert_slice %{{.*}} into %[[INIT]][0, 0] [2, 3] [1, 1]
    // CHECK: %[[I1:.*]] = tensor.insert_slice %{{.*}} into %[[I0]][0, 3] [2, 4] [1, 1]
    // CHECK: tensor.insert_slice %{{.*}} into %[[I1]][0, 7] [2, 5] [1, 1]

    return %r : tensor<2x12xf32>
  }

  // Test 3: single-input Concat is just an identity-shaped insert at
  // offset 0 into a fresh empty buffer.
  func.func @test_concat_single(%a: tensor<3x4xf32>) -> tensor<3x4xf32> {
    // CHECK-LABEL: func.func @test_concat_single
    %r = "onnx.Concat"(%a) {axis = 0 : si64}
        : (tensor<3x4xf32>) -> tensor<3x4xf32>

    // CHECK-NOT: onnx.Concat
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<3x4xf32>
    // CHECK: tensor.insert_slice %{{.*}} into %[[INIT]][0, 0] [3, 4] [1, 1]

    return %r : tensor<3x4xf32>
  }

  // Test 4: dynamic shapes — both inputs have a dynamic concat axis dim
  // and the output extent is the runtime sum. The untouched axis is
  // sourced from the first input with a static dim (here both are
  // static = 4 so we expect a constant 4 — the tensor.empty doesn't need
  // a dyn-size operand for it).
  func.func @test_concat_dynamic_axis(%a: tensor<?x4xf16>,
                                       %b: tensor<?x4xf16>) -> tensor<?x4xf16> {
    // CHECK-LABEL: func.func @test_concat_dynamic_axis
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<?x4xf16>, %[[B:.*]]: tensor<?x4xf16>)
    %r = "onnx.Concat"(%a, %b) {axis = 0 : si64}
        : (tensor<?x4xf16>, tensor<?x4xf16>) -> tensor<?x4xf16>

    // CHECK-NOT: onnx.Concat
    // Sum the two dynamic axis-0 dims for the output extent.
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[DA:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x4xf16>
    // CHECK: %[[DB:.*]] = tensor.dim %[[B]], %{{.*}} : tensor<?x4xf16>
    // CHECK: %[[SUM:.*]] = arith.addi %[[DA]], %[[DB]]
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[SUM]]) : tensor<?x4xf16>
    // Input 0 inserted at offset 0 with its runtime axis size.
    // CHECK: %[[I0:.*]] = tensor.insert_slice %[[A]] into %[[INIT]][0, 0] [%{{.*}}, 4] [1, 1]
    // Input 1 inserted at offset = dim_a (the running offset is just %[[DA]]
    // because staticOffset == 0 when we switch to dynamic).
    // CHECK: tensor.insert_slice %[[B]] into %[[I0]][%{{.*}}, 0] [%{{.*}}, 4] [1, 1]

    return %r : tensor<?x4xf16>
  }

  // Test 5: mixed static / dynamic inputs along the concat axis. Input 0
  // has static axis dim 3 (so the running offset starts at 3 statically);
  // input 1 has dynamic axis dim (so the running offset becomes 3 +
  // dim_b for input 2's start).
  func.func @test_concat_mixed(%a: tensor<3x4xf32>,
                                %b: tensor<?x4xf32>,
                                %c: tensor<2x4xf32>) -> tensor<?x4xf32> {
    // CHECK-LABEL: func.func @test_concat_mixed
    %r = "onnx.Concat"(%a, %b, %c) {axis = 0 : si64}
        : (tensor<3x4xf32>, tensor<?x4xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>

    // CHECK-NOT: onnx.Concat
    // Output dim 0 = 3 + dim_b + 2 (the pass emits the three extents in
    // order, summed left-to-right).
    // CHECK: tensor.empty(%{{.*}}) : tensor<?x4xf32>
    // Input 0 at constant offset 0.
    // CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [3, 4] [1, 1]
    // Input 1 at constant offset 3 (the static running offset before any
    // dynamic input).
    // CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[3, 0] [%{{.*}}, 4] [1, 1]
    // Input 2 at offset = 3 + dim_b (now dynamic).
    // CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [2, 4] [1, 1]

    return %r : tensor<?x4xf32>
  }
}
