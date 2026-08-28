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
}
