// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ Gemm Fusion Patterns (Pure PDLL approach)
//
// Pattern fuses:
//   onnx.DequantizeLinear x2 or x3 -> onnx.Gemm -> onnx.QuantizeLinear
// into:
//   hip.qgemm
//
// ONNX makes Gemm's C optional and a PDLL operand list matches an exact
// arity, so the two forms are separate patterns; both cases below are needed
// to cover them.
//
// Zero points are deliberately non-zero (asymmetric quantization) so the
// checks prove the extracted values reach the attributes with their sign
// intact, rather than matching an incidental zero. alpha / beta / transB are
// non-default for the same reason: a default-valued attribute is elided on
// print and would match whether or not the rewrite wired it.
//
// PDL fusion runs before lowerOnnxConstants and folds scale/zp into hip.qgemm
// attributes, so the onnx.Constant carriers are dead and get DCE'd — no
// hip.constant survivors in the output IR.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {
  // transB=1 reads B as [N, K], so Y is [64, 32]; the rank-1 C broadcasts
  // along M.
  func.func @main_graph(%A: tensor<64x128xi8>,
                        %B: tensor<32x128xi8>,
                        %C: tensor<32xi8>) -> tensor<64x32xi8> {
    %a_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %b_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<3> : tensor<i8>} : () -> tensor<i8>
    %c_scale = "onnx.Constant"() {value = dense<0.03125> : tensor<f32>} : () -> tensor<f32>
    %c_zp = "onnx.Constant"() {value = dense<-2> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<7> : tensor<i8>} : () -> tensor<i8>

    %a_dq = "onnx.DequantizeLinear"(%A, %a_scale, %a_zp)
            : (tensor<64x128xi8>, tensor<f32>, tensor<i8>) -> tensor<64x128xf32>
    %b_dq = "onnx.DequantizeLinear"(%B, %b_scale, %b_zp)
            : (tensor<32x128xi8>, tensor<f32>, tensor<i8>) -> tensor<32x128xf32>
    %c_dq = "onnx.DequantizeLinear"(%C, %c_scale, %c_zp)
            : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>

    %y = "onnx.Gemm"(%a_dq, %b_dq, %c_dq)
         {alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : i64}
         : (tensor<64x128xf32>, tensor<32x128xf32>, tensor<32xf32>) -> tensor<64x32xf32>

    %result = "onnx.QuantizeLinear"(%y, %y_scale, %y_zp)
              : (tensor<64x32xf32>, tensor<f32>, tensor<i8>) -> tensor<64x32xi8>

    return %result : tensor<64x32xi8>
  }

  // Trailing optional inputs may simply be omitted, which is the bias-free
  // form. A ui8 activation also exercises the unsigned zero-point read.
  func.func @gemm_no_bias(%A: tensor<16x64xui8>,
                          %B: tensor<64x8xi8>) -> tensor<16x8xui8> {
    %a_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %a_zp = "onnx.Constant"() {value = dense<128> : tensor<ui8>} : () -> tensor<ui8>
    %b_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %b_zp = "onnx.Constant"() {value = dense<-3> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<200> : tensor<ui8>} : () -> tensor<ui8>

    %a_dq = "onnx.DequantizeLinear"(%A, %a_scale, %a_zp)
            : (tensor<16x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<16x64xf32>
    %b_dq = "onnx.DequantizeLinear"(%B, %b_scale, %b_zp)
            : (tensor<64x8xi8>, tensor<f32>, tensor<i8>) -> tensor<64x8xf32>

    %y = "onnx.Gemm"(%a_dq, %b_dq)
         : (tensor<16x64xf32>, tensor<64x8xf32>) -> tensor<16x8xf32>

    %result = "onnx.QuantizeLinear"(%y, %y_scale, %y_zp)
              : (tensor<16x8xf32>, tensor<f32>, tensor<ui8>) -> tensor<16x8xui8>

    return %result : tensor<16x8xui8>
  }
}

// transA, and in the bias-free case alpha too, keep their ONNX defaults and so
// are elided on print.
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<64x128xi8>, %[[B:.*]]: tensor<32x128xi8>, %[[C:.*]]: tensor<32xi8>) -> tensor<64x32xi8> {
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<64x32xi8>
// CHECK-NEXT:    %[[QGEMM:.*]] = hip.qgemm(%[[CTX]]) ins(%[[A]], %[[B]], %[[C]] : tensor<64x128xi8>, tensor<32x128xi8>, tensor<32xi8>) outs(%[[EMPTY]] : tensor<64x32xi8>) {A_scale = 2.500000e-01 : f32, A_zero_point = -5 : i64, B_scale = 1.250000e-01 : f32, B_zero_point = 3 : i64, C_scale = 3.125000e-02 : f32, C_zero_point = -2 : i64, Y_scale = 5.000000e-01 : f32, Y_zero_point = 7 : i64, alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : i64} : tensor<64x32xi8>
// CHECK-NEXT:    return %[[QGEMM]] : tensor<64x32xi8>

// The bias-free form leaves C_scale / C_zero_point / beta unset, so they are
// absent here rather than carrying a meaningless value.
// CHECK-LABEL: func.func @gemm_no_bias
// CHECK-SAME:  (%[[CTX2:.*]]: !hip.context, %[[A2:.*]]: tensor<16x64xui8>, %[[B2:.*]]: tensor<64x8xi8>) -> tensor<16x8xui8> {
// CHECK-NEXT:    %[[EMPTY2:.*]] = tensor.empty() : tensor<16x8xui8>
// CHECK-NEXT:    %[[QGEMM2:.*]] = hip.qgemm(%[[CTX2]]) ins(%[[A2]], %[[B2]] : tensor<16x64xui8>, tensor<64x8xi8>) outs(%[[EMPTY2]] : tensor<16x8xui8>) {A_scale = 5.000000e-01 : f32, A_zero_point = 128 : i64, B_scale = 2.500000e-01 : f32, B_zero_point = -3 : i64, Y_scale = 1.250000e-01 : f32, Y_zero_point = 200 : i64} : tensor<16x8xui8>
// CHECK-NEXT:    return %[[QGEMM2]] : tensor<16x8xui8>

// Nothing of the fused chains survives: no onnx QDQ/Gemm ops and no
// hip.constant scale/zp carriers anywhere in the output.
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Gemm
// CHECK-NOT: onnx.QuantizeLinear
// CHECK-NOT: hip.constant
