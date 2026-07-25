// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX MatMulNBits custom op is correctly lowered to hip.matmul_nbits
// in tensor-first mode.
//
// This test validates:
// - MatMulNBits lowering (onnx.Custom → hip.matmul_nbits)
// - Basic 3-operand case (A, B, scales)
// - Optional zero_points operand handling
// - Attribute propagation (K, N, bits, block_size, accuracy_level)
// - Tensor-first DPS: tensor.empty() used as output init
// - Proper !hip.context threading through operations
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: Basic MatMulNBits (3 operands: A, B, scales) =====

  func.func @main_graph(%A: tensor<1x128x2880xf16>) -> tensor<1x128x5120xf16> {
    %B = "onnx.Constant"() {value = dense<1> : tensor<5120x90x16xui8>} : () -> tensor<5120x90x16xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<5120x90xf16>} : () -> tensor<5120x90xf16>
    %Y = "onnx.Custom"(%A, %B, %scales) {
      function_name = "MatMulNBits",
      domain_name = "com.microsoft",
      K = 2880 : si64, N = 5120 : si64,
      bits = 4 : si64, block_size = 32 : si64, accuracy_level = 4 : si64,
      onnx_node_name = "MatMulNBits_0"
    } : (tensor<1x128x2880xf16>, tensor<5120x90x16xui8>, tensor<5120x90xf16>) -> tensor<1x128x5120xf16>
    return %Y : tensor<1x128x5120xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<1x128x2880xf16>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x5120xf16>
  // CHECK: hip.matmul_nbits(%[[CTX]]) ins(%[[A]], %{{.*}}, %{{.*}} : tensor<1x128x2880xf16>, tensor<5120x90x16xui8>, tensor<5120x90xf16>) outs(%[[INIT]] : tensor<1x128x5120xf16>)
  // CHECK-SAME: K = 2880
  // CHECK-SAME: N = 5120
  // CHECK-SAME: bits = 4
  // CHECK-SAME: block_size = 32
  // CHECK-NOT: onnx.Custom

  // ===== Test 2: MatMulNBits with zero_points =====

  func.func @test_matmul_nbits_with_zp(%A: tensor<1x128x2880xf16>) -> tensor<1x128x5120xf16> {
    %B = "onnx.Constant"() {value = dense<1> : tensor<5120x90x16xui8>} : () -> tensor<5120x90x16xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<5120x90xf16>} : () -> tensor<5120x90xf16>
    %zp = "onnx.Constant"() {value = dense<8> : tensor<5120x90xui8>} : () -> tensor<5120x90xui8>
    %Y = "onnx.Custom"(%A, %B, %scales, %zp) {
      function_name = "MatMulNBits",
      domain_name = "com.microsoft",
      K = 2880 : si64, N = 5120 : si64,
      bits = 4 : si64, block_size = 32 : si64, accuracy_level = 4 : si64,
      onnx_node_name = "MatMulNBits_1"
    } : (tensor<1x128x2880xf16>, tensor<5120x90x16xui8>, tensor<5120x90xf16>, tensor<5120x90xui8>) -> tensor<1x128x5120xf16>
    return %Y : tensor<1x128x5120xf16>
  }

  // CHECK-LABEL: func.func @test_matmul_nbits_with_zp
  // CHECK: hip.matmul_nbits({{.*}}) ins({{.*}} : tensor<1x128x2880xf16>, tensor<5120x90x16xui8>, tensor<5120x90xf16>)
  // CHECK-SAME: zero_points(%{{.*}} : tensor<5120x90xui8>)
  // CHECK-SAME: outs(%{{.*}} : tensor<1x128x5120xf16>)
  // CHECK-NOT: onnx.Custom
}
