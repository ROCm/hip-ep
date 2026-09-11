// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true
// ============================================================================
// TEST: QDQ MatMul Fusion Pattern (Pure PDLL approach)
//
// Pattern fuses:
//   onnx.DequantizeLinear, onnx.DequantizeLinear
//     -> onnx.MatMul -> onnx.QuantizeLinear
// into:
//   hip.qmatmul
//
// Zero points are deliberately non-zero (asymmetric quantization) so the
// checks prove the extracted values reach the attributes with their sign
// intact, rather than matching an incidental zero.
//
// PDL fusion runs before lowerOnnxConstants and folds scale/zp into hip.qmatmul
// attributes, so the onnx.Constant carriers are dead and get DCE'd — no
// hip.constant survivors in the output IR.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {
  // ===== Fused: 2D, all i8, asymmetric zero points =====
  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<64x128xi8>, %[[B:.*]]: tensor<128x32xi8>) -> tensor<64x32xi8> {
  // CHECK-NEXT:   %[[EMPTY:.*]] = tensor.empty() : tensor<64x32xi8>
  // CHECK-NEXT:   %[[QMM:.*]] = hip.qmatmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<64x128xi8>, tensor<128x32xi8>) outs(%[[EMPTY]] : tensor<64x32xi8>) {A_scale = 2.500000e-01 : f32, A_zero_point = -5 : i64, B_scale = 1.250000e-01 : f32, B_zero_point = 3 : i64, Y_scale = 5.000000e-01 : f32, Y_zero_point = 7 : i64} : tensor<64x32xi8>
  // CHECK-NEXT:   return %[[QMM]] : tensor<64x32xi8>
  // CHECK-NEXT: }

  // Nothing of the fused chain survives: no onnx QDQ ops and no hip.constant
  // scale/zp carriers anywhere in the output.
  // CHECK-NOT: onnx.DequantizeLinear
  // CHECK-NOT: onnx.MatMul
  // CHECK-NOT: onnx.QuantizeLinear
  // CHECK-NOT: hip.constant
  func.func @main_graph(%A: tensor<64x128xi8>,
                        %B: tensor<128x32xi8>) -> tensor<64x32xi8> {
    %a_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %b_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<3> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<7> : tensor<i8>} : () -> tensor<i8>

    %a_dq = "onnx.DequantizeLinear"(%A, %a_scale, %a_zp)
            : (tensor<64x128xi8>, tensor<f32>, tensor<i8>) -> tensor<64x128xf32>
    %b_dq = "onnx.DequantizeLinear"(%B, %b_scale, %b_zp)
            : (tensor<128x32xi8>, tensor<f32>, tensor<i8>) -> tensor<128x32xf32>

    %prod = "onnx.MatMul"(%a_dq, %b_dq)
            : (tensor<64x128xf32>, tensor<128x32xf32>) -> tensor<64x32xf32>

    %result = "onnx.QuantizeLinear"(%prod, %y_scale, %y_zp)
              : (tensor<64x32xf32>, tensor<f32>, tensor<i8>) -> tensor<64x32xi8>

    return %result : tensor<64x32xi8>
  }
}
