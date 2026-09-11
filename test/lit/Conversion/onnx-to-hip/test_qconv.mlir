// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// UNSUPPORTED: true
// ============================================================================
// TEST: QDQ Conv Fusion Pattern (W4A16)
//
// Pattern fuses:
//   onnx.DequantizeLinear(activation), onnx.DequantizeLinear(weights)
//     -> onnx.Conv -> onnx.QuantizeLinear
// into:
//   hip.qconv
//
// The shapes and constant forms here mirror what an exported W4A16 LLM
// actually produces: a 1x1 Conv standing in for a linear layer, UINT16
// per-tensor activations with non-zero zero points, and INT4 weights packed
// two values per byte behind an EXTERNAL constant.
//
// The external form is load-bearing, not incidental. 4-bit weights import as
// i8 at the LOGICAL element count, so packing is observable only as a backing
// byte count of ceil(numel/2) -- 16 bytes for 32 weights, 2 bytes for 4 zero
// points. A dense value attribute would carry full-width bytes and would
// (correctly) fail to match. This also runs before lowerOnnxConstants stamps
// `packed_int4`, so the pattern cannot consult that marker and derives the
// packing itself.
//
// Unlike qadd/qmul, the weight scale and zero point stay OPERANDS, so their
// constants survive fusion as hip.constant carriers. Only the per-tensor
// activation and output scales/zero points fold into attributes and get DCE'd.
//
// Every function shares one module because metadata generation requires a
// @main_graph, which is the fusable case here.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
// ============================================================================

module {

// ===== The fusable W4A16 1x1 Conv =====
// Zero points are deliberately non-zero (asymmetric) so the checks prove the
// extracted values reach the attributes rather than matching an incidental 0.

// CHECK-LABEL: func.func @main_graph
// CHECK-DAG:   %[[W:.*]] = hip.constant {{.*}}tensor<4x8x1x1xi8>
// CHECK-DAG:   %[[WSCALE:.*]] = hip.constant {{.*}}tensor<4xf32>
// CHECK-DAG:   %[[WZP:.*]] = hip.constant {{.*}}tensor<4xi8>
// CHECK-DAG:   %[[EMPTY:.*]] = tensor.empty() : tensor<1x4x1x2xui16>
// CHECK:       hip.qconv
// CHECK-SAME:  ins(%{{.*}}, %[[W]], %[[WSCALE]], %[[WZP]] :
// CHECK-SAME:  outs(%[[EMPTY]] : tensor<1x4x1x2xui16>)
// CHECK-SAME:  dilations = [1, 1]
// CHECK-SAME:  group = 1 : i64
// CHECK-SAME:  input_scale = 1.638800e-04 : f32
// CHECK-SAME:  input_zp = 35275 : i64
// CHECK-SAME:  kernel_shape = [1, 1]
// CHECK-SAME:  output_scale = 3.687020e-04 : f32
// CHECK-SAME:  output_zp = 36322 : i64
// CHECK-SAME:  packed_int4
// CHECK-SAME:  pads = [0, 0, 0, 0]
// CHECK-SAME:  strides = [1, 1]
// CHECK-SAME:  weight_axis = 0 : i64
  func.func @main_graph(%x: tensor<1x8x1x2xui16>) -> tensor<1x4x1x2xui16> {
    // 32 logical weights in 16 bytes, and 4 zero points in 2 bytes: packed INT4.
    %w = "onnx.Constant"() {location = "w.bin", offset = 0 : i64, size = 16 : i64}
         : () -> tensor<4x8x1x1xi8>
    %w_zp = "onnx.Constant"() {location = "w.bin", offset = 16 : i64, size = 2 : i64}
            : () -> tensor<4xi8>
    // One f32 scale per output channel, full width.
    %w_scale = "onnx.Constant"() {location = "w.bin", offset = 32 : i64, size = 16 : i64}
               : () -> tensor<4xf32>
    %x_scale = "onnx.Constant"() {value = dense<1.638800e-04> : tensor<f32>} : () -> tensor<f32>
    %x_zp = "onnx.Constant"() {value = dense<35275> : tensor<ui16>} : () -> tensor<ui16>
    %y_scale = "onnx.Constant"() {value = dense<3.687020e-04> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<36322> : tensor<ui16>} : () -> tensor<ui16>

    %x_dq = "onnx.DequantizeLinear"(%x, %x_scale, %x_zp) {axis = 1 : si64, block_size = 0 : si64}
            : (tensor<1x8x1x2xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x8x1x2xf32>
    %w_dq = "onnx.DequantizeLinear"(%w, %w_scale, %w_zp) {axis = 0 : si64, block_size = 0 : si64}
            : (tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>) -> tensor<4x8x1x1xf32>
    %conv = "onnx.Conv"(%x_dq, %w_dq) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
                                       kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}
            : (tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>) -> tensor<1x4x1x2xf32>
    %y = "onnx.QuantizeLinear"(%conv, %y_scale, %y_zp) {axis = 1 : si64, block_size = 0 : si64}
         : (tensor<1x4x1x2xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x4x1x2xui16>
    return %y : tensor<1x4x1x2xui16>
  }

// ===== Full-width INT8 weights are not a 4-bit source =====
// 32 weights in 32 bytes is plain int8. Nothing about the i8 element type
// distinguishes it from the fusable case -- only the byte count does -- so this
// is what proves the packing check is the gate.

// CHECK-LABEL: func.func @qconv_int8_weights_not_fused
// CHECK-NOT:   hip.qconv
  func.func @qconv_int8_weights_not_fused(%x: tensor<1x8x1x2xui16>) -> tensor<1x4x1x2xui16> {
    %w = "onnx.Constant"() {location = "w.bin", offset = 0 : i64, size = 32 : i64}
         : () -> tensor<4x8x1x1xi8>
    %w_zp = "onnx.Constant"() {location = "w.bin", offset = 32 : i64, size = 4 : i64}
            : () -> tensor<4xi8>
    %w_scale = "onnx.Constant"() {location = "w.bin", offset = 36 : i64, size = 16 : i64}
               : () -> tensor<4xf32>
    %x_scale = "onnx.Constant"() {value = dense<1.638800e-04> : tensor<f32>} : () -> tensor<f32>
    %x_zp = "onnx.Constant"() {value = dense<35275> : tensor<ui16>} : () -> tensor<ui16>
    %y_scale = "onnx.Constant"() {value = dense<3.687020e-04> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<36322> : tensor<ui16>} : () -> tensor<ui16>

    %x_dq = "onnx.DequantizeLinear"(%x, %x_scale, %x_zp) {axis = 1 : si64, block_size = 0 : si64}
            : (tensor<1x8x1x2xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x8x1x2xf32>
    %w_dq = "onnx.DequantizeLinear"(%w, %w_scale, %w_zp) {axis = 0 : si64, block_size = 0 : si64}
            : (tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>) -> tensor<4x8x1x1xf32>
    %conv = "onnx.Conv"(%x_dq, %w_dq) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
                                       kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}
            : (tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>) -> tensor<1x4x1x2xf32>
    %y = "onnx.QuantizeLinear"(%conv, %y_scale, %y_zp) {axis = 1 : si64, block_size = 0 : si64}
         : (tensor<1x4x1x2xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x4x1x2xui16>
    return %y : tensor<1x4x1x2xui16>
  }

// ===== A per-tensor weight scale is not per-channel =====
// A scalar weight scale would fold into a single coefficient like qadd's, but
// the fused kernel indexes a scale per output channel and has no per-tensor
// path, so this must fall through rather than read a scalar as [Cout].

// CHECK-LABEL: func.func @qconv_per_tensor_weight_scale_not_fused
// CHECK-NOT:   hip.qconv
  func.func @qconv_per_tensor_weight_scale_not_fused(%x: tensor<1x8x1x2xui16>) -> tensor<1x4x1x2xui16> {
    %w = "onnx.Constant"() {location = "w.bin", offset = 0 : i64, size = 16 : i64}
         : () -> tensor<4x8x1x1xi8>
    %w_zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %w_scale = "onnx.Constant"() {value = dense<2.500000e-02> : tensor<f32>} : () -> tensor<f32>
    %x_scale = "onnx.Constant"() {value = dense<1.638800e-04> : tensor<f32>} : () -> tensor<f32>
    %x_zp = "onnx.Constant"() {value = dense<35275> : tensor<ui16>} : () -> tensor<ui16>
    %y_scale = "onnx.Constant"() {value = dense<3.687020e-04> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<36322> : tensor<ui16>} : () -> tensor<ui16>

    %x_dq = "onnx.DequantizeLinear"(%x, %x_scale, %x_zp) {axis = 1 : si64, block_size = 0 : si64}
            : (tensor<1x8x1x2xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x8x1x2xf32>
    %w_dq = "onnx.DequantizeLinear"(%w, %w_scale, %w_zp) {axis = 0 : si64, block_size = 0 : si64}
            : (tensor<4x8x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<4x8x1x1xf32>
    %conv = "onnx.Conv"(%x_dq, %w_dq) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
                                       kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}
            : (tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>) -> tensor<1x4x1x2xf32>
    %y = "onnx.QuantizeLinear"(%conv, %y_scale, %y_zp) {axis = 1 : si64, block_size = 0 : si64}
         : (tensor<1x4x1x2xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x4x1x2xui16>
    return %y : tensor<1x4x1x2xui16>
  }

// ===== A 3x3 kernel is not the degenerate dot-product case =====
// Only a 1x1 window collapses to a per-position dot product down the channel
// axis. A real window needs the im2col gather the fused kernel does not have,
// and would silently mismatch the hard-coded kernel_shape = [1, 1].

// CHECK-LABEL: func.func @qconv_3x3_not_fused
// CHECK-NOT:   hip.qconv
  func.func @qconv_3x3_not_fused(%x: tensor<1x8x8x8xui16>) -> tensor<1x4x8x8xui16> {
    %w = "onnx.Constant"() {location = "w.bin", offset = 0 : i64, size = 144 : i64}
         : () -> tensor<4x8x3x3xi8>
    %w_zp = "onnx.Constant"() {location = "w.bin", offset = 144 : i64, size = 2 : i64}
            : () -> tensor<4xi8>
    %w_scale = "onnx.Constant"() {location = "w.bin", offset = 146 : i64, size = 16 : i64}
               : () -> tensor<4xf32>
    %x_scale = "onnx.Constant"() {value = dense<1.638800e-04> : tensor<f32>} : () -> tensor<f32>
    %x_zp = "onnx.Constant"() {value = dense<35275> : tensor<ui16>} : () -> tensor<ui16>
    %y_scale = "onnx.Constant"() {value = dense<3.687020e-04> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<36322> : tensor<ui16>} : () -> tensor<ui16>

    %x_dq = "onnx.DequantizeLinear"(%x, %x_scale, %x_zp) {axis = 1 : si64, block_size = 0 : si64}
            : (tensor<1x8x8x8xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x8x8x8xf32>
    %w_dq = "onnx.DequantizeLinear"(%w, %w_scale, %w_zp) {axis = 0 : si64, block_size = 0 : si64}
            : (tensor<4x8x3x3xi8>, tensor<4xf32>, tensor<4xi8>) -> tensor<4x8x3x3xf32>
    %conv = "onnx.Conv"(%x_dq, %w_dq) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
                                       kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
            : (tensor<1x8x8x8xf32>, tensor<4x8x3x3xf32>) -> tensor<1x4x8x8xf32>
    %y = "onnx.QuantizeLinear"(%conv, %y_scale, %y_zp) {axis = 1 : si64, block_size = 0 : si64}
         : (tensor<1x4x8x8xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x4x8x8xui16>
    return %y : tensor<1x4x8x8xui16>
  }

// ===== An INT8 activation is not the supported width =====
// The kernel accumulates a UINT16 activation against a 4-bit weight. An 8-bit
// activation is a different requantization range, so it stays unfused rather
// than reaching a kernel that would reject it at runtime.

// CHECK-LABEL: func.func @qconv_int8_activation_not_fused
// CHECK-NOT:   hip.qconv
  func.func @qconv_int8_activation_not_fused(%x: tensor<1x8x1x2xi8>) -> tensor<1x4x1x2xi8> {
    %w = "onnx.Constant"() {location = "w.bin", offset = 0 : i64, size = 16 : i64}
         : () -> tensor<4x8x1x1xi8>
    %w_zp = "onnx.Constant"() {location = "w.bin", offset = 16 : i64, size = 2 : i64}
            : () -> tensor<4xi8>
    %w_scale = "onnx.Constant"() {location = "w.bin", offset = 32 : i64, size = 16 : i64}
               : () -> tensor<4xf32>
    %x_scale = "onnx.Constant"() {value = dense<1.638800e-04> : tensor<f32>} : () -> tensor<f32>
    %x_zp = "onnx.Constant"() {value = dense<-5> : tensor<i8>} : () -> tensor<i8>
    %y_scale = "onnx.Constant"() {value = dense<3.687020e-04> : tensor<f32>} : () -> tensor<f32>
    %y_zp = "onnx.Constant"() {value = dense<7> : tensor<i8>} : () -> tensor<i8>

    %x_dq = "onnx.DequantizeLinear"(%x, %x_scale, %x_zp) {axis = 1 : si64, block_size = 0 : si64}
            : (tensor<1x8x1x2xi8>, tensor<f32>, tensor<i8>) -> tensor<1x8x1x2xf32>
    %w_dq = "onnx.DequantizeLinear"(%w, %w_scale, %w_zp) {axis = 0 : si64, block_size = 0 : si64}
            : (tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>) -> tensor<4x8x1x1xf32>
    %conv = "onnx.Conv"(%x_dq, %w_dq) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
                                       kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}
            : (tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>) -> tensor<1x4x1x2xf32>
    %y = "onnx.QuantizeLinear"(%conv, %y_scale, %y_zp) {axis = 1 : si64, block_size = 0 : si64}
         : (tensor<1x4x1x2xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4x1x2xi8>
    return %y : tensor<1x4x1x2xi8>
  }

}
