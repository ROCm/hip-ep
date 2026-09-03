// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Sub (elementwise subtraction) is correctly lowered
// to hip.sub operation in tensor-first mode.
//
// This test validates:
// - Elementwise binary operation lowering (onnx.Sub -> hip.sub)
// - Two-input operand handling
// - i64 element type support
// - 2D tensor shape preservation
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B attention mask reformatting (ReduceSum - 1)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x1xi64>) -> tensor<1x1xi64> {
    return %arg0 : tensor<1x1xi64>
  }

  func.func @test_sub(%lhs: tensor<1x1xi64>, %rhs: tensor<1x1xi64>) -> tensor<1x1xi64> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @test_sub
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x1xi64>, %[[RHS:.*]]: tensor<1x1xi64>) -> tensor<1x1xi64>

    %output = "onnx.Sub"(%lhs, %rhs) : (tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x1xi64>

    // After conversion: tensor.empty() for init, hip.sub in tensor mode
    // CHECK: tensor.empty() : tensor<1x1xi64>
    // CHECK: hip.sub(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x1xi64>, tensor<1x1xi64>) outs({{.*}} : tensor<1x1xi64>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x1xi64>
  }

  // Dynamic shape test
  func.func @sub_dynamic(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %output = "onnx.Sub"(%lhs, %rhs) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %output : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @sub_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
  // CHECK: hip.sub(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>) : tensor<?x?xf32>
  // CHECK-NOT: hip.alloc

  // Broadcast: lhs [1] subtracted from rhs [32] -> output [32]
  func.func @sub_broadcast_i64(%lhs: tensor<1xi64>, %rhs: tensor<32xi64>) -> tensor<32xi64> {
    %output = "onnx.Sub"(%lhs, %rhs) : (tensor<1xi64>, tensor<32xi64>) -> tensor<32xi64>
    return %output : tensor<32xi64>
  }

  // CHECK-LABEL: func.func @sub_broadcast_i64
  // CHECK: tensor.empty() : tensor<32xi64>
  // CHECK: hip.sub(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1xi64>, tensor<32xi64>) outs({{.*}} : tensor<32xi64>)

  // rank-6 Sub with a broadcast axis. hip.sub only has a 4-D lowering, so the
  // axes are grouped such that each operand is either fully broadcast or fully
  // present within every group.
  func.func @sub_6d_broadcast(%lhs: tensor<2x3x4x5x6x7xf32>, %rhs: tensor<2x1x4x5x6x7xf32>) -> tensor<2x3x4x5x6x7xf32> {
    %output = "onnx.Sub"(%lhs, %rhs) : (tensor<2x3x4x5x6x7xf32>, tensor<2x1x4x5x6x7xf32>) -> tensor<2x3x4x5x6x7xf32>
    return %output : tensor<2x3x4x5x6x7xf32>
  }

  // CHECK-LABEL: func.func @sub_6d_broadcast
  // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] : tensor<2x3x4x5x6x7xf32> into tensor<2x3x4x210xf32>
  // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] : tensor<2x1x4x5x6x7xf32> into tensor<2x1x4x210xf32>
  // CHECK: hip.sub({{.*}}) ins({{.*}}, {{.*}} : tensor<2x3x4x210xf32>, tensor<2x1x4x210xf32>) outs({{.*}} : tensor<2x3x4x210xf32>)
  // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] output_shape [2, 3, 4, 5, 6, 7] : tensor<2x3x4x210xf32> into tensor<2x3x4x5x6x7xf32>
}
