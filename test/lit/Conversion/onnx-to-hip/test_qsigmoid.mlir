// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST: QDQ Sigmoid Fusion Pattern (Pure PDLL approach)
//
// Demonstrates pure PDLL fusion with native constraints:
// - PDLL matches the QDQ chain pattern
// - PDLL calls native constraints to get the context, extract the f32 scales
//   and extract the i64 zero points
// - PDLL creates the fused hip.qsigmoid operation
//
// Pattern fuses:
//   onnx.DequantizeLinear -> onnx.Sigmoid -> onnx.QuantizeLinear
// into:
//   hip.qsigmoid
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s --check-prefix=NOLEFTOVER
// ============================================================================

// The exhaustive CHECK-NEXT chain below already pins every surviving op, but
// spell the fusion contract out once so a partial match cannot pass silently.
// NOLEFTOVER-NOT: onnx.DequantizeLinear
// NOLEFTOVER-NOT: onnx.Sigmoid
// NOLEFTOVER-NOT: onnx.QuantizeLinear
// NOLEFTOVER-NOT: onnx.Constant
// NOLEFTOVER-NOT: hip.constant

module {

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<1x128x32xi8>
// CHECK-NEXT:    %[[QSIGMOID:.*]] = hip.qsigmoid(%[[CTX]]) ins(%[[X]] : tensor<1x128x32xi8>) outs(%[[EMPTY]] : tensor<1x128x32xi8>) {x_scale = 2.500000e-01 : f32, x_zero_point = -5 : i64, y_scale = 1.250000e-01 : f32, y_zero_point = 4 : i64} : tensor<1x128x32xi8>
// CHECK-NEXT:    return %[[QSIGMOID]] : tensor<1x128x32xi8>
// CHECK-NEXT:  }
  func.func @main_graph(%x: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
    %x_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %x_zero_point = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %y_zero_point = "onnx.Constant"() {value = dense<4> : tensor<i8>} : () -> tensor<i8>

    %dequantized = "onnx.DequantizeLinear"(%x, %x_scale, %x_zero_point)
                  : (tensor<1x128x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x128x32xf32>

    %sigmoid = "onnx.Sigmoid"(%dequantized)
              : (tensor<1x128x32xf32>) -> tensor<1x128x32xf32>

    %result = "onnx.QuantizeLinear"(%sigmoid, %y_scale, %y_zero_point)
              : (tensor<1x128x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x128x32xi8>

    return %result : tensor<1x128x32xi8>
  }

}
