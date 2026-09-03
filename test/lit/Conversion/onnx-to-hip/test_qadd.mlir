// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

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
}

// CHECK-LABEL: module
// CHECK-NEXT:  func.func @main_graph(%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x32xi8>, %[[RHS:.*]]: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<1x128x32xi8>
// CHECK-NEXT:    %[[QADD:.*]] = hip.qadd(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x128x32xi8>, tensor<1x128x32xi8>) outs(%[[EMPTY]] : tensor<1x128x32xi8>) {lhs_scale = 1.000000e-01 : f32, lhs_zp = -5 : i64, output_scale = 2.000000e-01 : f32, output_zp = 7 : i64, rhs_scale = 5.000000e-02 : f32, rhs_zp = 3 : i64} : tensor<1x128x32xi8>
// CHECK-NEXT:    return %[[QADD]] : tensor<1x128x32xi8>
// CHECK-NEXT:  }
// CHECK-NEXT: }
