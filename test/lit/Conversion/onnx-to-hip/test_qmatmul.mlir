// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ MatMul Fusion Pattern (PDLL-based two-phase approach)
//
// Demonstrates PDLL pattern matching:
//   Phase 1 (PDLL): Match QDQ chain and mark operations with attributes
//   Phase 2 (C++):  Process marked operations and create fused op
//
// Pattern fuses:
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

    // QDQ pattern (should be fused by PDLL + C++ two-phase approach)
    %quantized = "onnx.QuantizeLinear"(%input, %lhs_scale, %lhs_zp)
                 : (tensor<4x128xf32>, tensor<f32>, tensor<i8>) -> tensor<4x128xi8>

    %matmul_out = "onnx.MatMul"(%quantized, %weight)
                  : (tensor<4x128xi8>, tensor<128x256xf32>) -> tensor<4x256xf32>

    %result = "onnx.DequantizeLinear"(%matmul_out, %output_scale, %output_zp)
              : (tensor<4x256xf32>, tensor<f32>, tensor<i8>) -> tensor<4x256xf32>

    return %result : tensor<4x256xf32>
  }
}

// CHECK-LABEL: module
// CHECK-NEXT:  func.func @main_graph(%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<4x128xf32>, %[[WEIGHT:.*]]: tensor<128x256xf32>) -> tensor<4x256xf32> {
// CHECK-NEXT:    %[[C0:.*]] = hip.constant {serialization_order = 0 : i64, value = dense<1.000000e-01> : tensor<f32>} : tensor<f32>
// CHECK-NEXT:    %[[C1:.*]] = hip.constant {serialization_order = 1 : i64, value = dense<0> : tensor<i8>} : tensor<i8>
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<4x256xf32>
// CHECK-NEXT:    %[[QMATMUL:.*]] = hip.qmatmul(%[[CTX]]) ins(%[[INPUT]], %[[WEIGHT]] : tensor<4x128xf32>, tensor<128x256xf32>) outs(%[[EMPTY]] : tensor<4x256xf32>) {lhs_scale = 1.000000e-01 : f32, output_scale = 2.000000e-01 : f32, rhs_scale = 1.000000e+00 : f32} : tensor<4x256xf32>
// CHECK-NEXT:    return %[[QMATMUL]] : tensor<4x256xf32>
// CHECK-NEXT:  }
// CHECK-NEXT: }
