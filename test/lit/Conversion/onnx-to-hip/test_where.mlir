// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Where (elementwise selection) is correctly lowered
// to hip.where operation in tensor-first mode.
//
// This test validates:
// - Ternary elementwise lowering (onnx.Where -> hip.where)
// - Three-input operand handling (condition, x, y)
// - Bool (i1) condition + matching x/y/output element type
// - Same-rank operands and multidirectional broadcasting (NumPy-style)
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    return %arg0 : tensor<2x4xf32>
  }

  // Test 1: Same-shape Where (2x4 f32)
  func.func @test_where_same_shape(%cond: tensor<2x4xi1>,
                                   %x: tensor<2x4xf32>,
                                   %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_where_same_shape
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<2x4xi1>, %[[X:.*]]: tensor<2x4xf32>, %[[Y:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // CHECK: tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>) outs({{.*}} : tensor<2x4xf32>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<2x4xf32>
  }

  // Test 2: Multidirectional broadcasting (NumPy-style)
  // condition [1,4] + x [2,1] + y [2,4] -> output [2,4]
  func.func @test_where_broadcast(%cond: tensor<1x4xi1>,
                                  %x: tensor<2x1xf32>,
                                  %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_where_broadcast
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<1x4xi1>, %[[X:.*]]: tensor<2x1xf32>, %[[Y:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<1x4xi1>, tensor<2x1xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // CHECK: tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<1x4xi1>, tensor<2x1xf32>, tensor<2x4xf32>) outs({{.*}} : tensor<2x4xf32>)

    return %output : tensor<2x4xf32>
  }

  // Test 3: i64 element type (X/Y are int64)
  func.func @test_where_i64(%cond: tensor<3x5xi1>,
                            %x: tensor<3x5xi64>,
                            %y: tensor<3x5xi64>) -> tensor<3x5xi64> {
    // CHECK-LABEL: func.func @test_where_i64
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<3x5xi1>, %[[X:.*]]: tensor<3x5xi64>, %[[Y:.*]]: tensor<3x5xi64>) -> tensor<3x5xi64>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<3x5xi1>, tensor<3x5xi64>, tensor<3x5xi64>) -> tensor<3x5xi64>

    // CHECK: tensor.empty() : tensor<3x5xi64>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<3x5xi1>, tensor<3x5xi64>, tensor<3x5xi64>) outs({{.*}} : tensor<3x5xi64>)

    return %output : tensor<3x5xi64>
  }

  // Test 4: Dynamic shapes
  func.func @test_where_dynamic(%cond: tensor<?x?xi1>,
                                %x: tensor<?x?xf16>,
                                %y: tensor<?x?xf16>) -> tensor<?x?xf16> {
    // CHECK-LABEL: func.func @test_where_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<?x?xi1>, %[[X:.*]]: tensor<?x?xf16>, %[[Y:.*]]: tensor<?x?xf16>) -> tensor<?x?xf16>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<?x?xi1>, tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>

    // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf16>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<?x?xi1>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
    // CHECK-NOT: hip.alloc

    return %output : tensor<?x?xf16>
  }
}
