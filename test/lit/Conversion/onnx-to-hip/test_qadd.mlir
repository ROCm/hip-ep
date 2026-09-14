// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true
// ============================================================================
// TEST: QDQ Add Fusion Pattern (Pure PDLL approach)
//
// Demonstrates pure PDLL fusion with native constraints:
// - PDLL matches the QDQ chain pattern
// - PDLL calls native constraints to get the context, extract the f32 scales
//   and extract the i64 zero points
// - PDLL creates the fused hip.qadd operation
//
// Pattern fuses:
//   onnx.DequantizeLinear, onnx.DequantizeLinear
//     -> onnx.Add -> onnx.QuantizeLinear
// into:
//   hip.qadd
//
// Zero points are deliberately non-zero (asymmetric quantization) so the
// checks prove the extracted values reach the attributes with their sign
// intact, rather than matching an incidental zero.
//
// PDL fusion runs before lowerOnnxConstants and folds scale/zp into hip.qadd
// attributes, so the onnx.Constant carriers are dead and get DCE'd — no
// hip.constant survivors in the output IR.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {
  func.func @main_graph(%lhs: tensor<1x128x32xi8>,
                        %rhs: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
    // Quantization scales and zero points (constants)
    %lhs_scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %lhs_zp = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %rhs_scale = "onnx.Constant"() {value = dense<0.05> : tensor<f32>} : () -> tensor<f32>
    %rhs_zp = "onnx.Constant"() {value = dense<3> : tensor<i8>} : () -> tensor<i8>
    %output_scale = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %output_zp = "onnx.Constant"() {value = dense<7> : tensor<i8>} : () -> tensor<i8>

    // QDQ pattern (fused by pure PDLL with native constraints)
    %lhs_dequantized = "onnx.DequantizeLinear"(%lhs, %lhs_scale, %lhs_zp)
                       : (tensor<1x128x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x128x32xf32>

    %rhs_dequantized = "onnx.DequantizeLinear"(%rhs, %rhs_scale, %rhs_zp)
                       : (tensor<1x128x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x128x32xf32>

    %sum = "onnx.Add"(%lhs_dequantized, %rhs_dequantized)
           : (tensor<1x128x32xf32>, tensor<1x128x32xf32>) -> tensor<1x128x32xf32>

    %result = "onnx.QuantizeLinear"(%sum, %output_scale, %output_zp)
              : (tensor<1x128x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x128x32xi8>

    return %result : tensor<1x128x32xi8>
  }

  // The MorphiZen importer represents com.microsoft QDQ functions as
  // onnx.Custom. Canonicalization must happen before PDLL so this graph fuses
  // exactly like the native ONNX graph above.
  func.func @custom_qdq_add(%lhs: tensor<4xui16>,
                            %rhs: tensor<4xui16>) -> tensor<4xui16> {
    %lhs_scale = "onnx.Constant"() {value = dense<0.1> : tensor<f32>} : () -> tensor<f32>
    %lhs_zp = "onnx.Constant"() {value = dense<5> : tensor<ui16>} : () -> tensor<ui16>
    %rhs_scale = "onnx.Constant"() {value = dense<0.05> : tensor<f32>} : () -> tensor<f32>
    %rhs_zp = "onnx.Constant"() {value = dense<3> : tensor<ui16>} : () -> tensor<ui16>
    %output_scale = "onnx.Constant"() {value = dense<0.2> : tensor<f32>} : () -> tensor<f32>
    %output_zp = "onnx.Constant"() {value = dense<7> : tensor<ui16>} : () -> tensor<ui16>

    %lhs_dq = "onnx.Custom"(%lhs, %lhs_scale, %lhs_zp) {
      domain_name = "com.microsoft", function_name = "DequantizeLinear"
    } : (tensor<4xui16>, tensor<f32>, tensor<ui16>) -> tensor<4xf32>
    %rhs_dq = "onnx.Custom"(%rhs, %rhs_scale, %rhs_zp) {
      domain_name = "com.microsoft", function_name = "DequantizeLinear"
    } : (tensor<4xui16>, tensor<f32>, tensor<ui16>) -> tensor<4xf32>
    %sum = "onnx.Add"(%lhs_dq, %rhs_dq)
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %result = "onnx.Custom"(%sum, %output_scale, %output_zp) {
      domain_name = "com.microsoft", function_name = "QuantizeLinear"
    } : (tensor<4xf32>, tensor<f32>, tensor<ui16>) -> tensor<4xui16>
    return %result : tensor<4xui16>
  }
}

// CHECK-LABEL: module
// CHECK-NEXT:  func.func @main_graph(%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x32xi8>, %[[RHS:.*]]: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<1x128x32xi8>
// CHECK-NEXT:    %[[QADD:.*]] = hip.qadd(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x128x32xi8>, tensor<1x128x32xi8>) outs(%[[EMPTY]] : tensor<1x128x32xi8>) {lhs_scale = 1.000000e-01 : f32, lhs_zp = -5 : i64, output_scale = 2.000000e-01 : f32, output_zp = 7 : i64, rhs_scale = 5.000000e-02 : f32, rhs_zp = 3 : i64} : tensor<1x128x32xi8>
// CHECK-NEXT:    return %[[QADD]] : tensor<1x128x32xi8>
// CHECK-NEXT:  }
// CHECK-LABEL: func.func @custom_qdq_add
// CHECK:       %[[QADD:.*]] = hip.qadd
// CHECK-SAME:  lhs_scale = 1.000000e-01
// CHECK-SAME:  lhs_zp = 5 : i64
// CHECK-SAME:  output_scale = 2.000000e-01
// CHECK-SAME:  output_zp = 7 : i64
// CHECK-SAME:  rhs_scale = 5.000000e-02
// CHECK-SAME:  rhs_zp = 3 : i64
// CHECK-NOT:   onnx.Custom
// CHECK:       return %[[QADD]]
// CHECK-NEXT: }
