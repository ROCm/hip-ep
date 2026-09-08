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
//     the source op omits (axis=1, block_size=0, precision=0, saturate=1);
//     DequantizeLinear has no precision / saturate
//   - ONNX `output_dtype` is dropped, since the result element type already
//     encodes it (the importer maps UINT8 -> ui8, INT8 -> i8, and so on)
//   - Output init shape follows the input positionally
//   - INT4 / UINT4, the one case the result type cannot encode: both
//     directions forward the `packed_int4` marker onto the hip op
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s



module {

// ===== Test 1: Q/DQ pair, both with zero_point =====
// The quantize omits precision, so the hip op must come out carrying the
// ONNX default of 0.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x64x32xf32>, %[[SCALE:.*]]: tensor<64xf32>, %[[ZP:.*]]: tensor<64xi8>)
// CHECK-NEXT:  %[[QINIT:.*]] = tensor.empty() : tensor<1x64x32xi8>
// CHECK-NEXT:  %[[Q:.*]] = hip.quantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<1x64x32xf32>, tensor<64xf32>) zero_point(%[[ZP]] : tensor<64xi8>) outs(%[[QINIT]] : tensor<1x64x32xi8>) {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64, saturate = 1 : i64} : tensor<1x64x32xi8>
// CHECK-NEXT:  %[[DQINIT:.*]] = tensor.empty() : tensor<1x64x32xf32>
// CHECK-NEXT:  %[[DQ:.*]] = hip.dequantize_linear(%[[CTX]]) ins(%[[Q]], %[[SCALE]] : tensor<1x64x32xi8>, tensor<64xf32>) zero_point(%[[ZP]] : tensor<64xi8>) outs(%[[DQINIT]] : tensor<1x64x32xf32>) {axis = 1 : i64, block_size = 0 : i64} : tensor<1x64x32xf32>
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
// optional clause off the printed op entirely. output_dtype = 3 (ONNX INT8)
// is supplied and must NOT reach the hip op -- the i8 result type says the
// same thing.

// CHECK-LABEL: func.func @test_quantize_linear_attrs
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<128x4xf16>, %[[SCALE:.*]]: tensor<4x4xf16>)
// CHECK-NEXT:  %[[INIT:.*]] = tensor.empty() : tensor<128x4xi8>
// CHECK-NEXT:  %[[Y:.*]] = hip.quantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<128x4xf16>, tensor<4x4xf16>) outs(%[[INIT]] : tensor<128x4xi8>) {axis = 0 : i64, block_size = 32 : i64, precision = 16 : i64, saturate = 0 : i64} : tensor<128x4xi8>
// CHECK-NEXT:  return %[[Y]] : tensor<128x4xi8>
  func.func @test_quantize_linear_attrs(%x: tensor<128x4xf16>,
                                        %scale: tensor<4x4xf16>)
      -> tensor<128x4xi8> {
    %y = "onnx.QuantizeLinear"(%x, %scale) {
      axis = 0 : si64,
      block_size = 32 : si64,
      output_dtype = 3 : si64,
      precision = 16 : si64,
      saturate = 0 : si64,
      onnx_node_name = "QuantizeLinear_1"
    } : (tensor<128x4xf16>, tensor<4x4xf16>) -> tensor<128x4xi8>
    return %y : tensor<128x4xi8>
  }

// ===== Test 3: DequantizeLinear of a packed INT4 source =====
// ONNX INT4 imports as i8 at the LOGICAL element count, so nothing in the
// types distinguishes this from Test 1; only `packed_int4` -- stamped by
// constant lowering once it saw a half-size buffer -- does, and it has to
// survive onto the hip op. Both the input and the zero_point are packed, and
// every shape stays logical.

// CHECK-LABEL: func.func @test_dequantize_linear_packed_int4
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<4096x64xi8>, %[[SCALE:.*]]: tensor<128x64xf16>, %[[ZP:.*]]: tensor<128x64xi8>)
// CHECK-NEXT:  %[[INIT:.*]] = tensor.empty() : tensor<4096x64xf16>
// CHECK-NEXT:  %[[Y:.*]] = hip.dequantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<4096x64xi8>, tensor<128x64xf16>) zero_point(%[[ZP]] : tensor<128x64xi8>) outs(%[[INIT]] : tensor<4096x64xf16>) {axis = 0 : i64, block_size = 32 : i64, packed_int4} : tensor<4096x64xf16>
// CHECK-NEXT:  return %[[Y]] : tensor<4096x64xf16>
  func.func @test_dequantize_linear_packed_int4(%x: tensor<4096x64xi8>,
                                                %scale: tensor<128x64xf16>,
                                                %zp: tensor<128x64xi8>)
      -> tensor<4096x64xf16> {
    %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {
      axis = 0 : si64,
      block_size = 32 : si64,
      packed_int4,
      onnx_node_name = "DequantizeLinear_int4"
    } : (tensor<4096x64xi8>, tensor<128x64xf16>, tensor<128x64xi8>)
        -> tensor<4096x64xf16>
    return %y : tensor<4096x64xf16>
  }

// ===== Test 4: QuantizeLinear marked as a packed UINT4 target =====
// The mirror of Test 3 on the quantize side. No pass stamps `packed_int4`
// onto a QuantizeLinear yet -- a 4-bit target leaves no trace in the IR the
// way a half-size constant buffer does -- so the marker is written by hand
// here to cover the forwarding the rest of the stack depends on. output_dtype
// 21 (ONNX UINT4) must not reach the hip op; the marker is what carries the
// width, and the result type stays ui8 at the logical element count.

// CHECK-LABEL: func.func @test_quantize_linear_packed_uint4
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<128x64xf32>, %[[SCALE:.*]]: tensor<f32>)
// CHECK-NEXT:  %[[INIT:.*]] = tensor.empty() : tensor<128x64xui8>
// CHECK-NEXT:  %[[Y:.*]] = hip.quantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<128x64xf32>, tensor<f32>) outs(%[[INIT]] : tensor<128x64xui8>) {axis = 1 : i64, block_size = 0 : i64, packed_int4, precision = 0 : i64, saturate = 1 : i64} : tensor<128x64xui8>
// CHECK-NEXT:  return %[[Y]] : tensor<128x64xui8>
  func.func @test_quantize_linear_packed_uint4(%x: tensor<128x64xf32>,
                                               %scale: tensor<f32>)
      -> tensor<128x64xui8> {
    %y = "onnx.QuantizeLinear"(%x, %scale) {
      axis = 1 : si64,
      block_size = 0 : si64,
      output_dtype = 21 : si64,
      packed_int4,
      saturate = 1 : si64,
      onnx_node_name = "QuantizeLinear_packed_uint4"
    } : (tensor<128x64xf32>, tensor<f32>) -> tensor<128x64xui8>
    return %y : tensor<128x64xui8>
  }
}
