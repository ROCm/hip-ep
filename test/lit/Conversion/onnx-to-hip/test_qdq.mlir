// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify onnx.QuantizeLinear / onnx.DequantizeLinear are lowered to
// hip.quantize_linear / hip.dequantize_linear in tensor-first mode.
//
// Validates:
//   - Optional zero_point operand handling (present + absent)
//   - Attribute propagation, and the ONNX defaults filled in for attributes
//     the source op omits (axis=1, block_size=0, output_dtype=0, precision=0,
//     saturate=1); DequantizeLinear has no precision / saturate
//   - Output init shape follows the input positionally
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s



module {
  
// ===== Test 1: Q/DQ pair, both with zero_point =====
// The source ops omit output_dtype (and precision on the quantize), so both
// hip ops must come out carrying the ONNX default of 0.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x64x32xf32>, %[[SCALE:.*]]: tensor<64xf32>, %[[ZP:.*]]: tensor<64xi8>)
// CHECK-NEXT:  %[[QINIT:.*]] = tensor.empty() : tensor<1x64x32xi8>
// CHECK-NEXT:  %[[Q:.*]] = hip.quantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<1x64x32xf32>, tensor<64xf32>) zero_point(%[[ZP]] : tensor<64xi8>) outs(%[[QINIT]] : tensor<1x64x32xi8>) {axis = 1 : i64, block_size = 0 : i64, output_dtype = 0 : i64, precision = 0 : i64, saturate = 1 : i64} : tensor<1x64x32xi8>
// CHECK-NEXT:  %[[DQINIT:.*]] = tensor.empty() : tensor<1x64x32xf32>
// CHECK-NEXT:  %[[DQ:.*]] = hip.dequantize_linear(%[[CTX]]) ins(%[[Q]], %[[SCALE]] : tensor<1x64x32xi8>, tensor<64xf32>) zero_point(%[[ZP]] : tensor<64xi8>) outs(%[[DQINIT]] : tensor<1x64x32xf32>) {axis = 1 : i64, block_size = 0 : i64, output_dtype = 0 : i64} : tensor<1x64x32xf32>
// CHECK-NEXT:  return %[[DQ]] : tensor<1x64x32xf32>
  func.func @main_graph(%x: tensor<1x64x32xf32>, %scale: tensor<64xf32>,
                        %zp: tensor<64xi8>) -> tensor<1x64x32xf32> {
    %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {
      axis = 1 : si64,
      block_size = 0 : si64,
      saturate = 1 : si64,
      onnx_node_name = "QuantizeLinear_0"
    } : (tensor<1x64x32xf32>, tensor<64xf32>, tensor<64xi8>)
        -> tensor<1x64x32xi8>
    %y = "onnx.DequantizeLinear"(%q, %scale, %zp) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "DequantizeLinear_0"
    } : (tensor<1x64x32xi8>, tensor<64xf32>, tensor<64xi8>)
        -> tensor<1x64x32xf32>
    return %y : tensor<1x64x32xf32>
  }

// ===== Test 2: QuantizeLinear without zero_point, all attributes set =====
// Every attribute carries a non-default value, so nothing here can pass by
// accidentally matching a default. The absent zero_point must leave the
// optional clause off the printed op entirely.

// CHECK-LABEL: func.func @test_quantize_linear_attrs
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<4x128xf16>, %[[SCALE:.*]]: tensor<4x4xf16>)
// CHECK-NEXT:  %[[INIT:.*]] = tensor.empty() : tensor<4x128xi8>
// CHECK-NEXT:  %[[Y:.*]] = hip.quantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<4x128xf16>, tensor<4x4xf16>) outs(%[[INIT]] : tensor<4x128xi8>) {axis = 0 : i64, block_size = 32 : i64, output_dtype = 3 : i64, precision = 16 : i64, saturate = 0 : i64} : tensor<4x128xi8>
// CHECK-NEXT:  return %[[Y]] : tensor<4x128xi8>
  func.func @test_quantize_linear_attrs(%x: tensor<4x128xf16>,
                                        %scale: tensor<4x4xf16>)
      -> tensor<4x128xi8> {
    %y = "onnx.QuantizeLinear"(%x, %scale) {
      axis = 0 : si64,
      block_size = 32 : si64,
      output_dtype = 3 : si64,
      precision = 16 : si64,
      saturate = 0 : si64,
      onnx_node_name = "QuantizeLinear_1"
    } : (tensor<4x128xf16>, tensor<4x4xf16>) -> tensor<4x128xi8>
    return %y : tensor<4x128xi8>
  }
}
