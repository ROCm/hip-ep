// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Mul (elementwise multiplication) is correctly lowered
// to hip.mul operation in tensor-first mode.
//
// This test validates:
// - Elementwise binary operation lowering (onnx.Mul → hip.mul)
// - Two-input operand handling
// - f16 element type support
// - 3D tensor shape preservation
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B SiLU activation (gate_proj * sigmoid)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @test_mul(%lhs: tensor<1x128x14336xf16>, %rhs: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @test_mul
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x14336xf16>, %[[RHS:.*]]: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>

    %output = "onnx.Mul"(%lhs, %rhs) : (tensor<1x128x14336xf16>, tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>

    // After conversion: tensor.empty() for init, hip.mul in tensor mode
    // CHECK: tensor.empty() : tensor<1x128x14336xf16>
    // CHECK: hip.mul(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x128x14336xf16>, tensor<1x128x14336xf16>) outs({{.*}} : tensor<1x128x14336xf16>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<1x128x14336xf16>
  }

  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x128x14336xf16>, %arg1: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16> {
    return %arg0 : tensor<1x128x14336xf16>
  }

  func.func @mul_5d_simplebev(
      %lhs: tensor<6x128x200x8x200xf32>,
      %rhs: tensor<6x1x200x8x200xf32>)
      -> tensor<6x128x200x8x200xf32> {
    // CHECK-LABEL: func.func @mul_5d_simplebev
    // CHECK: %[[PACKED_LHS:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<6x128x200x8x200xf32> into tensor<6x128x200x1600xf32>
    // CHECK: %[[PACKED_RHS:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<6x1x200x8x200xf32> into tensor<6x1x200x1600xf32>
    // CHECK: %[[PACKED_INIT:.*]] = tensor.empty() : tensor<6x128x200x1600xf32>
    // CHECK: %[[PACKED_RESULT:.*]] = hip.mul({{.*}}) ins(%[[PACKED_LHS]], %[[PACKED_RHS]] : tensor<6x128x200x1600xf32>, tensor<6x1x200x1600xf32>) outs(%[[PACKED_INIT]] : tensor<6x128x200x1600xf32>)
    // CHECK: %[[RESULT:.*]] = tensor.expand_shape %[[PACKED_RESULT]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [6, 128, 200, 8, 200] : tensor<6x128x200x1600xf32> into tensor<6x128x200x8x200xf32>
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<6x128x200x8x200xf32>, tensor<6x1x200x8x200xf32>)
        -> tensor<6x128x200x8x200xf32>
    return %result : tensor<6x128x200x8x200xf32>
  }

  func.func @mul_6d_same_shape(
      %lhs: tensor<1x2x3x4x5x6xf16>,
      %rhs: tensor<1x2x3x4x5x6xf16>)
      -> tensor<1x2x3x4x5x6xf16> {
    // CHECK-LABEL: func.func @mul_6d_same_shape
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] : tensor<1x2x3x4x5x6xf16> into tensor<1x2x3x120xf16>
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] : tensor<1x2x3x4x5x6xf16> into tensor<1x2x3x120xf16>
    // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<1x2x3x120xf16>, tensor<1x2x3x120xf16>) outs({{.*}} : tensor<1x2x3x120xf16>)
    // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] output_shape [1, 2, 3, 4, 5, 6] : tensor<1x2x3x120xf16> into tensor<1x2x3x4x5x6xf16>
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<1x2x3x4x5x6xf16>, tensor<1x2x3x4x5x6xf16>)
        -> tensor<1x2x3x4x5x6xf16>
    return %result : tensor<1x2x3x4x5x6xf16>
  }

  func.func @mul_5d_short_rhs(
      %lhs: tensor<2x3x4x5x6xf32>, %rhs: tensor<4x5x6xf32>)
      -> tensor<2x3x4x5x6xf32> {
    // CHECK-LABEL: func.func @mul_5d_short_rhs
    // CHECK: %[[ALIGNED_RHS:.*]] = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2], [3], [4]] output_shape [1, 1, 4, 5, 6] : tensor<4x5x6xf32> into tensor<1x1x4x5x6xf32>
    // CHECK: tensor.collapse_shape %[[ALIGNED_RHS]] {{\[\[}}0], [1], [2], [3, 4]] : tensor<1x1x4x5x6xf32> into tensor<1x1x4x30xf32>
    // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<2x3x4x30xf32>, tensor<1x1x4x30xf32>) outs({{.*}} : tensor<2x3x4x30xf32>)
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x3x4x5x6xf32>, tensor<4x5x6xf32>)
        -> tensor<2x3x4x5x6xf32>
    return %result : tensor<2x3x4x5x6xf32>
  }
}
