// UNSUPPORTED: *
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ Mul Fusion Pattern (Pure PDLL approach)
//
// Demonstrates pure PDLL fusion with native constraints:
// - PDLL matches the QDQ chain pattern
// - PDLL calls native constraints to get the context, extract the f32 scales
//   and extract the i64 zero points
// - PDLL creates the fused hip.qmul operation
//
// Pattern fuses:
//   onnx.DequantizeLinear, onnx.DequantizeLinear
//     -> onnx.Mul -> onnx.QuantizeLinear
// into:
//   hip.qmul
//
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s --check-prefix=NOLEFTOVER
// ============================================================================

// The exhaustive CHECK-NEXT chains below already pin every surviving op, but
// spell the fusion contract out once so a partial match cannot pass silently.
// NOLEFTOVER-NOT: onnx.DequantizeLinear
// NOLEFTOVER-NOT: onnx.Mul
// NOLEFTOVER-NOT: onnx.QuantizeLinear
// NOLEFTOVER-NOT: onnx.Constant
// NOLEFTOVER-NOT: hip.constant

module {
// ONNX Mul declares NumPy broadcasting. PDL fusion runs ahead of the ONNX
// broadcast pre-passes, so the rank-1 rhs must reach hip.qmul unpadded and
// unpacked -- no collapse_shape/expand_shape around the fused op.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x32xi8>, %[[RHS:.*]]: tensor<32xi8>) -> tensor<1x128x32xi8> {
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<1x128x32xi8>
// CHECK-NEXT:    %[[QMUL:.*]] = hip.qmul(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x128x32xi8>, tensor<32xi8>) outs(%[[EMPTY]] : tensor<1x128x32xi8>) {lhs_scale = 2.500000e-01 : f32, lhs_zp = -12 : i64, output_scale = 1.250000e-01 : f32, output_zp = 4 : i64, rhs_scale = 5.000000e-01 : f32, rhs_zp = 9 : i64} : tensor<1x128x32xi8>
// CHECK-NEXT:    return %[[QMUL]] : tensor<1x128x32xi8>
// CHECK-NEXT:  }
  func.func @main_graph(%lhs: tensor<1x128x32xi8>,
                            %rhs: tensor<32xi8>) -> tensor<1x128x32xi8> {
    %lhs_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %lhs_zp = "onnx.Constant"() {value = dense<-12> : tensor<i8>} : () -> tensor<i8>
    %rhs_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %rhs_zp = "onnx.Constant"() {value = dense<9> : tensor<i8>} : () -> tensor<i8>
    %output_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %output_zp = "onnx.Constant"() {value = dense<4> : tensor<i8>} : () -> tensor<i8>

    %lhs_dequantized = "onnx.DequantizeLinear"(%lhs, %lhs_scale, %lhs_zp)
                       : (tensor<1x128x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x128x32xf32>

    %rhs_dequantized = "onnx.DequantizeLinear"(%rhs, %rhs_scale, %rhs_zp)
                       : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>

    %product = "onnx.Mul"(%lhs_dequantized, %rhs_dequantized)
               : (tensor<1x128x32xf32>, tensor<32xf32>) -> tensor<1x128x32xf32>

    %result = "onnx.QuantizeLinear"(%product, %output_scale, %output_zp)
              : (tensor<1x128x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x128x32xi8>

    return %result : tensor<1x128x32xi8>
  }

}
