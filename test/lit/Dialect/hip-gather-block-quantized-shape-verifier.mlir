// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// Packed int4 may use a partial final block, and packed uint8 zero points may
// use ceil(scales[quantize_axis] / 2) bytes.
func.func @valid_partial_block(
    %ctx: !hip.context,
    %data: memref<4x65xui8, 1>,
    %indices: memref<3xi64, 1>,
    %scales: memref<4x5xf16, 1>,
    %zero_points: memref<4x3xui8, 1>,
    %output: memref<3x130xf16, 1>) {
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x65xui8, 1>, memref<3xi64, 1>, memref<4x5xf16, 1>)
    zero_points(%zero_points : memref<4x3xui8, 1>)
    outs(%output : memref<3x130xf16, 1>)
    {bits = 4, block_size = 32, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @wrong_logical_output(
    %ctx: !hip.context,
    %data: memref<4x65xui8, 1>,
    %indices: memref<3xi64, 1>,
    %scales: memref<4x5xf16, 1>,
    %output: memref<3x65xf16, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 130}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x65xui8, 1>, memref<3xi64, 1>, memref<4x5xf16, 1>)
    outs(%output : memref<3x65xf16, 1>)
    {bits = 4, block_size = 32, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @wrong_scales_grid(
    %ctx: !hip.context,
    %data: memref<4x65xui8, 1>,
    %indices: memref<3xi64, 1>,
    %scales: memref<4x4xf16, 1>,
    %output: memref<3x130xf16, 1>) {
  // expected-error @+1 {{data/scales mismatch at axis 1: expected scales extent 5 but got 4}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x65xui8, 1>, memref<3xi64, 1>, memref<4x4xf16, 1>)
    outs(%output : memref<3x130xf16, 1>)
    {bits = 4, block_size = 32, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @wrong_zero_points_grid(
    %ctx: !hip.context,
    %data: memref<4x65xi8, 1>,
    %indices: memref<3xi32, 1>,
    %scales: memref<4x5xf32, 1>,
    %zero_points: memref<4x2xi8, 1>,
    %output: memref<3x130xf32, 1>) {
  // expected-error @+1 {{zero_points extent at axis 1 must match scales or its packed-byte extent}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x65xi8, 1>, memref<3xi32, 1>, memref<4x5xf32, 1>)
    zero_points(%zero_points : memref<4x2xi8, 1>)
    outs(%output : memref<3x130xf32, 1>)
    {bits = 4, block_size = 32, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @bad_block_size(
    %ctx: !hip.context,
    %data: memref<4x64xi8, 1>,
    %indices: memref<3xi64, 1>,
    %scales: memref<4x6xf16, 1>,
    %output: memref<3x128xf16, 1>) {
  // expected-error @+1 {{block_size must be a power of two and at least 16}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xi8, 1>, memref<3xi64, 1>, memref<4x6xf16, 1>)
    outs(%output : memref<3x128xf16, 1>)
    {bits = 4, block_size = 24, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @bad_bits(
    %ctx: !hip.context,
    %data: memref<4x64xi8, 1>,
    %indices: memref<3xi64, 1>,
    %scales: memref<4x4xf16, 1>,
    %output: memref<3x64xf16, 1>) {
  // expected-error @+1 {{bits must be 4 or 8}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xi8, 1>, memref<3xi64, 1>, memref<4x4xf16, 1>)
    outs(%output : memref<3x64xf16, 1>)
    {bits = 3, block_size = 32, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

// Byte-packed UINT4 is not subject to the UINT8-only axis restriction.
func.func @valid_uint4_nonzero_gather_axis(
    %ctx: !hip.context,
    %data: memref<4x64xui8, 1>,
    %indices: memref<2xi64, 1>,
    %scales: memref<4x4xf16, 1>,
    %output: memref<4x2xf16, 1>) {
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xui8, 1>, memref<2xi64, 1>, memref<4x4xf16, 1>)
    outs(%output : memref<4x2xf16, 1>)
    {bits = 4, block_size = 32, gather_axis = 1, quantize_axis = 1}
  return
}

// -----

func.func @valid_signed_i8_general_axes(
    %ctx: !hip.context,
    %data: memref<64x512xsi8, 1>,
    %indices: memref<2xi32, 1>,
    %scales: memref<2x512xf32, 1>,
    %output: memref<64x2xf32, 1>) {
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<64x512xsi8, 1>, memref<2xi32, 1>, memref<2x512xf32, 1>)
    outs(%output : memref<64x2xf32, 1>)
    {bits = 8, block_size = 32, gather_axis = 1, quantize_axis = 0}
  return
}

// -----

func.func @valid_unsigned_signless_i8(
    %ctx: !hip.context,
    %data: memref<4x64xi8, 1>,
    %indices: memref<2xi64, 1>,
    %scales: memref<4x2xf16, 1>,
    %output: memref<2x64xf16, 1>) {
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xi8, 1>, memref<2xi64, 1>, memref<4x2xf16, 1>)
    outs(%output : memref<2x64xf16, 1>)
    {bits = 8, block_size = 32, gather_axis = 0, quantize_axis = 1,
     unsigned_quant_storage}
  return
}

// -----

func.func @bad_gather_axis_for_uint8(
    %ctx: !hip.context,
    %data: memref<4x64xui8, 1>,
    %indices: memref<2xi64, 1>,
    %scales: memref<4x2xf16, 1>,
    %output: memref<4x2xf16, 1>) {
  // expected-error @+1 {{gather_axis must be 0 for uint8 storage}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xui8, 1>, memref<2xi64, 1>, memref<4x2xf16, 1>)
    outs(%output : memref<4x2xf16, 1>)
    {bits = 8, block_size = 32, gather_axis = 1, quantize_axis = 1}
  return
}

// -----

func.func @bad_quantize_axis_for_uint8(
    %ctx: !hip.context,
    %data: memref<4x64xui8, 1>,
    %indices: memref<2xi64, 1>,
    %scales: memref<1x64xf16, 1>,
    %output: memref<2x64xf16, 1>) {
  // expected-error @+1 {{quantize_axis must be the last dimension for uint8 storage}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xui8, 1>, memref<2xi64, 1>, memref<1x64xf16, 1>)
    outs(%output : memref<2x64xf16, 1>)
    {bits = 8, block_size = 32, gather_axis = 0, quantize_axis = 0}
  return
}

// -----

func.func @conflicting_signed_storage_marker(
    %ctx: !hip.context,
    %data: memref<64x512xsi8, 1>,
    %indices: memref<2xi32, 1>,
    %scales: memref<2x512xf32, 1>,
    %output: memref<64x2xf32, 1>) {
  // expected-error @+1 {{unsigned_quant_storage conflicts with explicitly signed data storage}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<64x512xsi8, 1>, memref<2xi32, 1>, memref<2x512xf32, 1>)
    outs(%output : memref<64x2xf32, 1>)
    {bits = 8, block_size = 32, gather_axis = 1, quantize_axis = 0,
     unsigned_quant_storage}
  return
}

// -----

func.func @bad_quantize_axis(
    %ctx: !hip.context,
    %data: memref<4x64xi8, 1>,
    %indices: memref<2xi64, 1>,
    %scales: memref<4x64xf16, 1>,
    %output: memref<2x128xf16, 1>) {
  // expected-error @+1 {{quantize_axis must be in [-2, 1]}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xi8, 1>, memref<2xi64, 1>, memref<4x64xf16, 1>)
    outs(%output : memref<2x128xf16, 1>)
    {bits = 4, block_size = 32, gather_axis = 0, quantize_axis = 2}
  return
}

// -----

func.func @bad_data_rank(
    %ctx: !hip.context,
    %data: memref<64xi8, 1>,
    %indices: memref<2xi64, 1>,
    %scales: memref<4xf16, 1>,
    %output: memref<2xf16, 1>) {
  // expected-error @+1 {{data rank must be in [2, 8], got 1}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<64xi8, 1>, memref<2xi64, 1>, memref<4xf16, 1>)
    outs(%output : memref<2xf16, 1>)
    {bits = 4, block_size = 32, gather_axis = 0, quantize_axis = 0}
  return
}

// -----

func.func @bad_element_types(
    %ctx: !hip.context,
    %data: memref<4x64xi16, 1>,
    %indices: memref<2xi16, 1>,
    %scales: memref<4x2xf64, 1>,
    %output: memref<2x64xf32, 1>) {
  // expected-error @+1 {{data storage must have an 8-bit integer element type}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xi16, 1>, memref<2xi16, 1>, memref<4x2xf64, 1>)
    outs(%output : memref<2x64xf32, 1>)
    {bits = 8, block_size = 32, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @wrong_output_type(
    %ctx: !hip.context,
    %data: memref<4x64xui8, 1>,
    %indices: memref<2xi64, 1>,
    %scales: memref<4x2xf16, 1>,
    %output: memref<2x64xf32, 1>) {
  // expected-error @+1 {{output element type must match scales}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x64xui8, 1>, memref<2xi64, 1>, memref<4x2xf16, 1>)
    outs(%output : memref<2x64xf32, 1>)
    {bits = 8, block_size = 32, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @logical_int4_extent_overflow(
    %ctx: !hip.context,
    %data: memref<4x4611686018427387904xi8, 1>,
    %indices: memref<3xi64, 1>,
    %scales: memref<4x?xf16, 1>,
    %output: memref<3x?xf16, 1>) {
  // expected-error @+1 {{logical quantized extent overflows i64}}
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x4611686018427387904xi8, 1>, memref<3xi64, 1>,
        memref<4x?xf16, 1>)
    outs(%output : memref<3x?xf16, 1>)
    {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
  return
}

// -----

func.func @logical_int4_extent_i64_boundary(
    %ctx: !hip.context,
    %data: memref<4x4611686018427387903xi8, 1>,
    %indices: memref<3xi64, 1>,
    %scales: memref<4x?xf16, 1>,
    %output: memref<3x9223372036854775806xf16, 1>) {
  hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        memref<4x4611686018427387903xi8, 1>, memref<3xi64, 1>,
        memref<4x?xf16, 1>)
    outs(%output : memref<3x9223372036854775806xf16, 1>)
    {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
  return
}
