// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The fusion is W4A16 over a 1x1 window: UINT16 per-tensor activations and
// INT4 weights quantized per output channel. The negative cases below each
// disable exactly one of those preconditions, leaving the rest fusable, so a
// missing CHECK-NOT names the constraint that stopped gating.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A 1x1 Conv standing in for a linear layer. The weight scale and zero point
// hold one value per output channel, so they stay operands and their carriers
// survive the fusion; only the per-tensor activation and output parameters
// fold into attributes.
// CHECK-LABEL: func.func @qconv
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x8x1x2xui16>) -> tensor<1x4x1x2xui16> {
// CHECK-NEXT:    %[[W:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, size = 16 : i64} : tensor<4x8x1x1xi8>
// CHECK-NEXT:    %[[WZP:.*]] = hip.constant {location = "w.bin", offset = 16 : i64, size = 2 : i64} : tensor<4xi8>
// CHECK-NEXT:    %[[WSCALE:.*]] = hip.constant {location = "w.bin", offset = 32 : i64, size = 16 : i64} : tensor<4xf32>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<1x4x1x2xui16>
// CHECK-NEXT:    %[[QCONV:.*]] = hip.qconv(%[[CTX]]) ins(%[[X]], %[[W]], %[[WSCALE]], %[[WZP]] : tensor<1x8x1x2xui16>, tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>) outs(%[[INIT]] : tensor<1x4x1x2xui16>) {dilations = [1, 1], group = 1 : i64, input_scale = 1.638800e-04 : f32, input_zp = 35275 : i64, kernel_shape = [1, 1], output_scale = 3.687020e-04 : f32, output_zp = 36322 : i64, packed_int4, pads = [0, 0, 0, 0], strides = [1, 1], weight_axis = 0 : i64} : tensor<1x4x1x2xui16>
// CHECK-NEXT:    return %[[QCONV]] : tensor<1x4x1x2xui16>
// CHECK-NEXT:  }
func.func @qconv(%ctx: !hip.context,
                 %x: tensor<1x8x1x2xui16>) -> tensor<1x4x1x2xui16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 16 : i64} : tensor<4x8x1x1xi8>
  %w_zp = hip.constant {location = "w.bin", offset = 16 : i64, size = 2 : i64} : tensor<4xi8>
  %w_scale = hip.constant {location = "w.bin", offset = 32 : i64, size = 16 : i64} : tensor<4xf32>
  // Zero points are deliberately non-zero so the checks prove the extracted
  // values reach the attributes rather than matching an incidental 0.
  %x_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %x_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x8x1x2xf32>
  %dq_x = hip.dequantize_linear(%ctx)
      ins(%x, %x_scale : tensor<1x8x1x2xui16>, tensor<f32>)
      zero_point(%x_zp : tensor<ui16>)
      outs(%e0 : tensor<1x8x1x2xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x1x2xf32>

  %e1 = tensor.empty() : tensor<4x8x1x1xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<4x8x1x1xi8>, tensor<4xf32>)
      zero_point(%w_zp : tensor<4xi8>)
      outs(%e1 : tensor<4x8x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4} : tensor<4x8x1x1xf32>

  %e2 = tensor.empty() : tensor<1x4x1x2xf32>
  %conv = hip.conv(%ctx)
      ins(%dq_x, %dq_w : tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>)
      outs(%e2 : tensor<1x4x1x2xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x1x2xf32>

  %e3 = tensor.empty() : tensor<1x4x1x2xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%conv, %y_scale : tensor<1x4x1x2xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e3 : tensor<1x4x1x2xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x4x1x2xui16>

  return %q : tensor<1x4x1x2xui16>
}

// -----

// Full-width 8-bit weights are not a 4-bit source. A packed operand keeps an
// i8 element type and its logical element count, so packed_int4 is the only
// thing that distinguishes this from the fusable case, and there is no 8-bit
// qconv pattern for it to fall through to.
// CHECK-LABEL: func.func @qconv_w8_not_fused
// CHECK-NOT:   hip.qconv
func.func @qconv_w8_not_fused(%ctx: !hip.context,
                              %x: tensor<1x8x1x2xui16>) -> tensor<1x4x1x2xui16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 32 : i64} : tensor<4x8x1x1xi8>
  %w_zp = hip.constant {location = "w.bin", offset = 32 : i64, size = 4 : i64} : tensor<4xi8>
  %w_scale = hip.constant {location = "w.bin", offset = 36 : i64, size = 16 : i64} : tensor<4xf32>
  %x_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %x_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x8x1x2xf32>
  %dq_x = hip.dequantize_linear(%ctx)
      ins(%x, %x_scale : tensor<1x8x1x2xui16>, tensor<f32>)
      zero_point(%x_zp : tensor<ui16>)
      outs(%e0 : tensor<1x8x1x2xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x1x2xf32>

  %e1 = tensor.empty() : tensor<4x8x1x1xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<4x8x1x1xi8>, tensor<4xf32>)
      zero_point(%w_zp : tensor<4xi8>)
      outs(%e1 : tensor<4x8x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64} : tensor<4x8x1x1xf32>

  %e2 = tensor.empty() : tensor<1x4x1x2xf32>
  %conv = hip.conv(%ctx)
      ins(%dq_x, %dq_w : tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>)
      outs(%e2 : tensor<1x4x1x2xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x1x2xf32>

  %e3 = tensor.empty() : tensor<1x4x1x2xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%conv, %y_scale : tensor<1x4x1x2xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e3 : tensor<1x4x1x2xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x4x1x2xui16>

  return %q : tensor<1x4x1x2xui16>
}

// -----

// A per-tensor weight scale would fold into a single coefficient like qadd's,
// but the fused kernel indexes a scale per output channel and has no
// per-tensor path, so this must fall through rather than read a scalar as
// [Cout].
// CHECK-LABEL: func.func @qconv_per_tensor_weight_scale_not_fused
// CHECK-NOT:   hip.qconv
func.func @qconv_per_tensor_weight_scale_not_fused(%ctx: !hip.context,
                                                   %x: tensor<1x8x1x2xui16>) -> tensor<1x4x1x2xui16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 16 : i64} : tensor<4x8x1x1xi8>
  %w_zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %w_scale = hip.constant {value = dense<2.500000e-02> : tensor<f32>} : tensor<f32>
  %x_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %x_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x8x1x2xf32>
  %dq_x = hip.dequantize_linear(%ctx)
      ins(%x, %x_scale : tensor<1x8x1x2xui16>, tensor<f32>)
      zero_point(%x_zp : tensor<ui16>)
      outs(%e0 : tensor<1x8x1x2xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x1x2xf32>

  %e1 = tensor.empty() : tensor<4x8x1x1xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<4x8x1x1xi8>, tensor<f32>)
      zero_point(%w_zp : tensor<i8>)
      outs(%e1 : tensor<4x8x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4} : tensor<4x8x1x1xf32>

  %e2 = tensor.empty() : tensor<1x4x1x2xf32>
  %conv = hip.conv(%ctx)
      ins(%dq_x, %dq_w : tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>)
      outs(%e2 : tensor<1x4x1x2xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x1x2xf32>

  %e3 = tensor.empty() : tensor<1x4x1x2xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%conv, %y_scale : tensor<1x4x1x2xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e3 : tensor<1x4x1x2xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x4x1x2xui16>

  return %q : tensor<1x4x1x2xui16>
}

// -----

// Only a 1x1 window collapses to a per-position dot product down the channel
// axis. A real window needs the im2col gather the fused kernel does not have,
// and would silently mismatch the hard-coded kernel_shape and pads.
// CHECK-LABEL: func.func @qconv_3x3_not_fused
// CHECK-NOT:   hip.qconv
func.func @qconv_3x3_not_fused(%ctx: !hip.context,
                               %x: tensor<1x8x8x8xui16>) -> tensor<1x4x8x8xui16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 144 : i64} : tensor<4x8x3x3xi8>
  %w_zp = hip.constant {location = "w.bin", offset = 144 : i64, size = 2 : i64} : tensor<4xi8>
  %w_scale = hip.constant {location = "w.bin", offset = 146 : i64, size = 16 : i64} : tensor<4xf32>
  %x_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %x_zp = hip.constant {value = dense<35275> : tensor<ui16>} : tensor<ui16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<36322> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x8x8x8xf32>
  %dq_x = hip.dequantize_linear(%ctx)
      ins(%x, %x_scale : tensor<1x8x8x8xui16>, tensor<f32>)
      zero_point(%x_zp : tensor<ui16>)
      outs(%e0 : tensor<1x8x8x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x8x8xf32>

  %e1 = tensor.empty() : tensor<4x8x3x3xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<4x8x3x3xi8>, tensor<4xf32>)
      zero_point(%w_zp : tensor<4xi8>)
      outs(%e1 : tensor<4x8x3x3xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4} : tensor<4x8x3x3xf32>

  %e2 = tensor.empty() : tensor<1x4x8x8xf32>
  %conv = hip.conv(%ctx)
      ins(%dq_x, %dq_w : tensor<1x8x8x8xf32>, tensor<4x8x3x3xf32>)
      outs(%e2 : tensor<1x4x8x8xf32>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x8x8xf32>

  %e3 = tensor.empty() : tensor<1x4x8x8xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%conv, %y_scale : tensor<1x4x8x8xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<ui16>)
      outs(%e3 : tensor<1x4x8x8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x4x8x8xui16>

  return %q : tensor<1x4x8x8xui16>
}

// -----

// A signed 16-bit activation has the supported width but not the supported
// signedness: it spans a different range and widens differently, so it stays
// unfused rather than reaching a kernel that reads the codes as unsigned.
// CHECK-LABEL: func.func @qconv_i16_activation_not_fused
// CHECK-NOT:   hip.qconv
func.func @qconv_i16_activation_not_fused(%ctx: !hip.context,
                                          %x: tensor<1x8x1x2xi16>) -> tensor<1x4x1x2xi16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 16 : i64} : tensor<4x8x1x1xi8>
  %w_zp = hip.constant {location = "w.bin", offset = 16 : i64, size = 2 : i64} : tensor<4xi8>
  %w_scale = hip.constant {location = "w.bin", offset = 32 : i64, size = 16 : i64} : tensor<4xf32>
  %x_scale = hip.constant {value = dense<1.638800e-04> : tensor<f32>} : tensor<f32>
  %x_zp = hip.constant {value = dense<-5> : tensor<i16>} : tensor<i16>
  %y_scale = hip.constant {value = dense<3.687020e-04> : tensor<f32>} : tensor<f32>
  %y_zp = hip.constant {value = dense<7> : tensor<i16>} : tensor<i16>

  %e0 = tensor.empty() : tensor<1x8x1x2xf32>
  %dq_x = hip.dequantize_linear(%ctx)
      ins(%x, %x_scale : tensor<1x8x1x2xi16>, tensor<f32>)
      zero_point(%x_zp : tensor<i16>)
      outs(%e0 : tensor<1x8x1x2xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x1x2xf32>

  %e1 = tensor.empty() : tensor<4x8x1x1xf32>
  %dq_w = hip.dequantize_linear(%ctx)
      ins(%w, %w_scale : tensor<4x8x1x1xi8>, tensor<4xf32>)
      zero_point(%w_zp : tensor<4xi8>)
      outs(%e1 : tensor<4x8x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4} : tensor<4x8x1x1xf32>

  %e2 = tensor.empty() : tensor<1x4x1x2xf32>
  %conv = hip.conv(%ctx)
      ins(%dq_x, %dq_w : tensor<1x8x1x2xf32>, tensor<4x8x1x1xf32>)
      outs(%e2 : tensor<1x4x1x2xf32>)
      {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0],
       dilations = [1, 1], group = 1 : i64} : tensor<1x4x1x2xf32>

  %e3 = tensor.empty() : tensor<1x4x1x2xi16>
  %q = hip.quantize_linear(%ctx)
      ins(%conv, %y_scale : tensor<1x4x1x2xf32>, tensor<f32>)
      zero_point(%y_zp : tensor<i16>)
      outs(%e3 : tensor<1x4x1x2xi16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x4x1x2xi16>

  return %q : tensor<1x4x1x2xi16>
}
