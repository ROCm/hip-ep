// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.quantize_linear and hip.dequantize_linear name a sub-byte payload with
// `packed_bits`, and the packing is defined against byte storage. A nonzero
// `packed_bits` therefore requires an 8-bit integer quantized operand and zero
// point; otherwise a consumer that reads `packed_bits` as the value width would
// see a 4-bit payload in a buffer whose byte count nothing describes.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @dequantize_packed_i8
// CHECK:         hip.dequantize_linear
// CHECK-SAME:      packed_bits = 4
func.func @dequantize_packed_i8(%ctx: !hip.context, %x: tensor<2x3xi8>,
                                %scale: tensor<f32>, %zp: tensor<i8>,
                                %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>)
      outs(%init : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_bits = 4 : i64}
      : tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: func.func @quantize_packed_ui8_memref
// CHECK:         hip.quantize_linear
// CHECK-SAME:      packed_bits = 2
func.func @quantize_packed_ui8_memref(%ctx: !hip.context,
                                      %x: memref<2x3xf32, 1>,
                                      %scale: memref<f32, 1>,
                                      %out: memref<2x3xui8, 1>) {
  hip.quantize_linear(%ctx)
      ins(%x, %scale : memref<2x3xf32, 1>, memref<f32, 1>)
      outs(%out : memref<2x3xui8, 1>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64, packed_bits = 2 : i64}
  return
}

// -----

// Unpacked storage at another width is unaffected.
// CHECK-LABEL: func.func @dequantize_unpacked_i16
// CHECK:         hip.dequantize_linear
func.func @dequantize_unpacked_i16(%ctx: !hip.context, %x: tensor<2x3xi16>,
                                   %scale: tensor<f32>,
                                   %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xi16>, tensor<f32>)
      outs(%init : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64}
      : tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// -----

func.func @dequantize_packed_i4(%ctx: !hip.context, %x: tensor<2x3xi4>,
                                %scale: tensor<f32>,
                                %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error @+1 {{packed_bits = 4 expects 8-bit integer storage, got 'i4'}}
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xi4>, tensor<f32>)
      outs(%init : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_bits = 4 : i64}
      : tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// -----

func.func @dequantize_packed_i16_zero_point(%ctx: !hip.context,
                                            %x: tensor<2x3xi8>,
                                            %scale: tensor<f32>,
                                            %zp: tensor<i16>,
                                            %init: tensor<2x3xf32>)
    -> tensor<2x3xf32> {
  // expected-error @+1 {{packed_bits = 4 expects 8-bit integer storage, got 'i16'}}
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xi8>, tensor<f32>)
      zero_point(%zp : tensor<i16>)
      outs(%init : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_bits = 4 : i64}
      : tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// -----

func.func @quantize_packed_i16(%ctx: !hip.context, %x: tensor<2x3xf32>,
                               %scale: tensor<f32>,
                               %init: tensor<2x3xi16>) -> tensor<2x3xi16> {
  // expected-error @+1 {{packed_bits = 2 expects 8-bit integer storage, got 'i16'}}
  %r = hip.quantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xf32>, tensor<f32>)
      outs(%init : tensor<2x3xi16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64, packed_bits = 2 : i64} : tensor<2x3xi16>
  return %r : tensor<2x3xi16>
}

// -----

func.func @dequantize_packed_bits_3(%ctx: !hip.context, %x: tensor<2x3xi8>,
                                    %scale: tensor<f32>,
                                    %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // The attribute is parsed, and so rejected, at the attribute dictionary.
  // expected-error @+4 {{attribute 'packed_bits' failed to satisfy constraint}}
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x3xi8>, tensor<f32>)
      outs(%init : tensor<2x3xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_bits = 3 : i64}
      : tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}
