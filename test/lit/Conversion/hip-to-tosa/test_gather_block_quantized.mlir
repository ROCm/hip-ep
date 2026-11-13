// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.gather_block_quantized decomposes to TOSA as a gather followed by
// a block-wise dequantize: gather the packed rows and the matching per-block
// scales with one index vector, widen the sub-byte values, repeat each scale
// across its block, then (q - zp) * scale.
//
// FILE LAYOUT:
// Converting cases first, each in its own --split-input-file chunk; the
// declined shapes follow. Declining is not an error -- the op is dynamically
// legal, so an unsupported shape stays a hip op instead of failing the pass.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// uint4, no zero_points: vocab=8, block_size=16, 2 blocks -> 32 logical values
// per row held in 16 packed bytes.
// CHECK-LABEL: func.func @gbq_uint4_default_zp
// CHECK: tosa.gather
// CHECK: tosa.bitwise_and
// CHECK: tosa.logical_right_shift
// CHECK: tosa.bitwise_and
// CHECK: tosa.concat
// CHECK: tosa.gather
// CHECK: tosa.tile
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: hip.gather_block_quantized
func.func @gbq_uint4_default_zp(%ctx: !hip.context, %data: tensor<8x16xui8>,
                                %indices: tensor<3xi64>,
                                %scales: tensor<8x2xf16>,
                                %init: tensor<3x32xf16>) -> tensor<3x32xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x2xf16>)
        outs(%init : tensor<3x32xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x32xf16>
  return %r : tensor<3x32xf16>
}

// -----

// Explicit zero_points ride the same index vector as the scales, so a third
// gather appears.
// CHECK-LABEL: func.func @gbq_uint4_with_zp
// CHECK: tosa.gather
// CHECK: tosa.concat
// CHECK: tosa.gather
// CHECK: tosa.gather
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: hip.gather_block_quantized
func.func @gbq_uint4_with_zp(%ctx: !hip.context, %data: tensor<8x16xui8>,
                             %indices: tensor<3xi64>, %scales: tensor<8x2xf16>,
                             %zp: tensor<8x2xui8>,
                             %init: tensor<3x32xf16>) -> tensor<3x32xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x2xf16>)
        zero_points(%zp : tensor<8x2xui8>)
        outs(%init : tensor<3x32xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x32xf16>
  return %r : tensor<3x32xf16>
}

// -----

// Signed int4 (signless storage, no unsigned_quant_storage) sign-extends each
// nibble with the branch-free (v ^ 8) - 8.
// CHECK-LABEL: func.func @gbq_int4_sign_extend
// CHECK: tosa.concat
// CHECK: tosa.bitwise_xor
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: hip.gather_block_quantized
func.func @gbq_int4_sign_extend(%ctx: !hip.context, %data: tensor<8x16xi8>,
                                %indices: tensor<3xi64>,
                                %scales: tensor<8x2xf32>,
                                %init: tensor<3x32xf32>) -> tensor<3x32xf32>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xi8>, tensor<3xi64>, tensor<8x2xf32>)
        outs(%init : tensor<3x32xf32>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x32xf32>
  return %r : tensor<3x32xf32>
}

// -----

// unsigned_quant_storage marks uint4 held in signless bytes, so the nibble is
// taken as-is rather than sign-extended.
// CHECK-LABEL: func.func @gbq_unsigned_quant_storage
// CHECK: tosa.concat
// CHECK-NOT: tosa.bitwise_xor
// CHECK: tosa.mul
// CHECK-NOT: hip.gather_block_quantized
func.func @gbq_unsigned_quant_storage(%ctx: !hip.context,
                                      %data: tensor<8x16xi8>,
                                      %indices: tensor<3xi64>,
                                      %scales: tensor<8x2xf16>,
                                      %init: tensor<3x32xf16>)
    -> tensor<3x32xf16> attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xi8>, tensor<3xi64>, tensor<8x2xf16>)
        outs(%init : tensor<3x32xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64, unsigned_quant_storage}
        : tensor<3x32xf16>
  return %r : tensor<3x32xf16>
}

// -----

// bits=8 is unpacked, so there is no nibble interleave; the mask only undoes
// tosa.cast's sign extension for uint8 storage.
// CHECK-LABEL: func.func @gbq_uint8_no_unpack
// CHECK-NOT: tosa.concat
// CHECK: tosa.gather
// CHECK: tosa.bitwise_and
// CHECK: tosa.mul
// CHECK-NOT: hip.gather_block_quantized
func.func @gbq_uint8_no_unpack(%ctx: !hip.context, %data: tensor<4x32xui8>,
                               %indices: tensor<2xi32>,
                               %scales: tensor<4x1xf16>,
                               %init: tensor<2x32xf16>) -> tensor<2x32xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<4x32xui8>, tensor<2xi32>, tensor<4x1xf16>)
        outs(%init : tensor<2x32xf16>)
        {bits = 8 : i64, block_size = 32 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<2x32xf16>
  return %r : tensor<2x32xf16>
}

// -----

// Negative axes normalize against the data rank before the gate runs.
// CHECK-LABEL: func.func @gbq_negative_axes
// CHECK: tosa.gather
// CHECK: tosa.mul
// CHECK-NOT: hip.gather_block_quantized
func.func @gbq_negative_axes(%ctx: !hip.context, %data: tensor<8x16xui8>,
                             %indices: tensor<3xi64>, %scales: tensor<8x2xf16>,
                             %init: tensor<3x32xf16>) -> tensor<3x32xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x2xf16>)
        outs(%init : tensor<3x32xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = -2 : i64,
         quantize_axis = -1 : i64}
        : tensor<3x32xf16>
  return %r : tensor<3x32xf16>
}

// -----

// Rank-2 indices keep their shape in the result; the rows collapse for the
// gather and expand again at the end.
// CHECK-LABEL: func.func @gbq_rank2_indices
// CHECK: tosa.gather
// CHECK: tosa.mul
// CHECK-NOT: hip.gather_block_quantized
func.func @gbq_rank2_indices(%ctx: !hip.context, %data: tensor<8x16xui8>,
                             %indices: tensor<2x3xi64>,
                             %scales: tensor<8x2xf16>,
                             %init: tensor<2x3x32xf16>) -> tensor<2x3x32xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<2x3xi64>, tensor<8x2xf16>)
        outs(%init : tensor<2x3x32xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<2x3x32xf16>
  return %r : tensor<2x3x32xf16>
}

// -----

// gather_axis must be 0: the index vector no longer addresses whole scale rows
// and a gathered slice need not start on a byte boundary.
// CHECK-LABEL: func.func @gbq_reject_gather_axis
// CHECK: hip.gather_block_quantized
func.func @gbq_reject_gather_axis(%ctx: !hip.context, %data: tensor<8x16xui8>,
                                  %indices: tensor<3xi64>,
                                  %scales: tensor<8x2xf16>,
                                  %init: tensor<8x3xf16>) -> tensor<8x3xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x2xf16>)
        outs(%init : tensor<8x3xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 1 : i64,
         quantize_axis = 1 : i64}
        : tensor<8x3xf16>
  return %r : tensor<8x3xf16>
}

// -----

// Only bits 4 and 8 are handled; 2-bit packing is four values per byte.
// CHECK-LABEL: func.func @gbq_reject_bits2
// CHECK: hip.gather_block_quantized
func.func @gbq_reject_bits2(%ctx: !hip.context, %data: tensor<8x16xui8>,
                            %indices: tensor<3xi64>, %scales: tensor<8x4xf16>,
                            %init: tensor<3x64xf16>) -> tensor<3x64xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x4xf16>)
        outs(%init : tensor<3x64xf16>)
        {bits = 2 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x64xf16>
  return %r : tensor<3x64xf16>
}

// -----

// The per-block extent must tile the quantized axis: 3 blocks of 16 is 48
// logical values, which 16 packed bytes cannot hold.
// CHECK-LABEL: func.func @gbq_reject_block_extent
// CHECK: hip.gather_block_quantized
func.func @gbq_reject_block_extent(%ctx: !hip.context, %data: tensor<8x16xui8>,
                                   %indices: tensor<3xi64>,
                                   %scales: tensor<8x3xf16>,
                                   %init: tensor<3x48xf16>) -> tensor<3x48xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x3xf16>)
        outs(%init : tensor<3x48xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x48xf16>
  return %r : tensor<3x48xf16>
}

// -----

// block_size must be a power of two and at least 16.
// CHECK-LABEL: func.func @gbq_reject_block_size
// CHECK: hip.gather_block_quantized
func.func @gbq_reject_block_size(%ctx: !hip.context, %data: tensor<8x8xui8>,
                                 %indices: tensor<3xi64>,
                                 %scales: tensor<8x2xf16>,
                                 %init: tensor<3x16xf16>) -> tensor<3x16xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x8xui8>, tensor<3xi64>, tensor<8x2xf16>)
        outs(%init : tensor<3x16xf16>)
        {bits = 4 : i64, block_size = 8 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x16xf16>
  return %r : tensor<3x16xf16>
}

// -----

// TOSA has no f64 tensor type.
// CHECK-LABEL: func.func @gbq_reject_f64
// CHECK: hip.gather_block_quantized
func.func @gbq_reject_f64(%ctx: !hip.context, %data: tensor<8x16xui8>,
                          %indices: tensor<3xi64>, %scales: tensor<8x2xf64>,
                          %init: tensor<3x32xf64>) -> tensor<3x32xf64>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x2xf64>)
        outs(%init : tensor<3x32xf64>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x32xf64>
  return %r : tensor<3x32xf64>
}

// -----

// The op takes its output element type from scales, but nothing verifies it,
// so the mismatch has to be caught here. f64 scales are the sharp case: the
// output type alone looks supported, and the gather is emitted on the scales
// before any cast could rescue it -- on a tensor TOSA cannot represent.
// CHECK-LABEL: func.func @gbq_reject_scales_f64_output_f32
// CHECK: hip.gather_block_quantized
// CHECK-NOT: tosa.gather
func.func @gbq_reject_scales_f64_output_f32(%ctx: !hip.context,
                                            %data: tensor<8x16xui8>,
                                            %indices: tensor<3xi64>,
                                            %scales: tensor<8x2xf64>,
                                            %init: tensor<3x32xf32>)
    -> tensor<3x32xf32> attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x2xf64>)
        outs(%init : tensor<3x32xf32>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x32xf32>
  return %r : tensor<3x32xf32>
}

// -----

// Both types are representable here, which is what makes this one worth
// declining rather than converting: casting the scales up to the output type
// would produce plausible IR that computes in a precision the op never asked
// for.
// CHECK-LABEL: func.func @gbq_reject_scales_output_mismatch
// CHECK: hip.gather_block_quantized
// CHECK-NOT: tosa.gather
func.func @gbq_reject_scales_output_mismatch(%ctx: !hip.context,
                                             %data: tensor<8x16xui8>,
                                             %indices: tensor<3xi64>,
                                             %scales: tensor<8x2xf16>,
                                             %init: tensor<3x32xf32>)
    -> tensor<3x32xf32> attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x16xui8>, tensor<3xi64>, tensor<8x2xf16>)
        outs(%init : tensor<3x32xf32>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x32xf32>
  return %r : tensor<3x32xf32>
}

// -----

// zero_points must carry one value per block. The packed-nibble form is
// ambiguous with the per-byte form when there is a single block.
// CHECK-LABEL: func.func @gbq_reject_packed_zp
// CHECK: hip.gather_block_quantized
func.func @gbq_reject_packed_zp(%ctx: !hip.context, %data: tensor<8x32xui8>,
                                %indices: tensor<3xi64>,
                                %scales: tensor<8x4xf16>,
                                %zp: tensor<8x2xui8>,
                                %init: tensor<3x64xf16>) -> tensor<3x64xf16>
    attributes {rock.kernel} {
  %r = hip.gather_block_quantized(%ctx) ins(%data, %indices, %scales :
        tensor<8x32xui8>, tensor<3xi64>, tensor<8x4xf16>)
        zero_points(%zp : tensor<8x2xui8>)
        outs(%init : tensor<3x64xf16>)
        {bits = 4 : i64, block_size = 16 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
        : tensor<3x64xf16>
  return %r : tensor<3x64xf16>
}
