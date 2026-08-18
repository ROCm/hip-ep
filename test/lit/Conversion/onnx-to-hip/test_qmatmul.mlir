// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ MatMul Fusion Pattern
//
// Demonstrates pattern matching that fuses:
//   onnx.QuantizeLinear -> onnx.MatMul -> onnx.DequantizeLinear
// into:
//   hip.qmatmul
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {
  func.func @main_graph(%input: tensor<4x128xf32>, %weight: tensor<128x256xf32>) -> tensor<4x256xf32> {
    // Quantization scales (constants)
    %lhs_scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %lhs_zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %output_scale = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %output_zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>

    // QDQ pattern (should be fused)
    %quantized = "onnx.QuantizeLinear"(%input, %lhs_scale, %lhs_zp)
                 : (tensor<4x128xf32>, tensor<f32>, tensor<i8>) -> tensor<4x128xi8>

    %matmul_out = "onnx.MatMul"(%quantized, %weight)
                  : (tensor<4x128xi8>, tensor<128x256xf32>) -> tensor<4x256xf32>

    %result = "onnx.DequantizeLinear"(%matmul_out, %output_scale, %output_zp)
              : (tensor<4x256xf32>, tensor<f32>, tensor<i8>) -> tensor<4x256xf32>

    return %result : tensor<4x256xf32>
  }
}

// Verify fusion happened
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<4x128xf32>, %[[WEIGHT:.*]]: tensor<128x256xf32>)

// Check that QDQ ops are gone
// CHECK-NOT: onnx.QuantizeLinear
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Constant

// Check that fused op exists with correct scales
// CHECK: hip.qmatmul(%[[CTX]])
// CHECK-SAME: ins(%[[INPUT]], %[[WEIGHT]]
// CHECK-SAME: lhs_scale = 1.000000e-01
// CHECK-SAME: rhs_scale = 1.000000e+00
// CHECK-SAME: output_scale = 2.000000e-01
