// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.matmul_nbits decomposes to TOSA the same way MIGraphX does:
// unpack packed uint8 int4 weights, block-broadcast scales/zero-points,
// dequantize (cast(x) - zp) * scale, transpose to [K, N], then tosa.matmul.
//
// FILE LAYOUT:
// Converting cases in the first --split-input-file chunk; each rejection
// in its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// K=16, N=2, block_size=16, bits=4 -> B is [2, 1, 8] packed uint8.
// CHECK-LABEL: func.func @matmul_nbits_basic
// CHECK: tosa.bitwise_and
// CHECK: tosa.logical_right_shift
// CHECK: tosa.concat
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK: tosa.transpose
// CHECK: tosa.matmul
// CHECK-NOT: hip.matmul_nbits
func.func @matmul_nbits_basic(%ctx: !hip.context, %a: tensor<2x16xf16>,
                              %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>,
                              %init: tensor<2x2xf16>) -> tensor<2x2xf16>
    attributes {rock.kernel} {
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        outs(%init : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// zp_elem_size=1 with k_blocks=1 is packed [N,1]: unpack the low nibble,
// do not treat the raw byte as the zero-point.
// CHECK-LABEL: func.func @matmul_nbits_packed_zp_kblocks1
// CHECK: tosa.bitwise_and
// CHECK: tosa.logical_right_shift
// CHECK: tosa.concat
// CHECK: tosa.sub
// CHECK: tosa.matmul
// CHECK-NOT: hip.matmul_nbits
func.func @matmul_nbits_packed_zp_kblocks1(%ctx: !hip.context, %a: tensor<2x16xf16>,
                                     %b: tensor<2x1x8xui8>,
                                     %scales: tensor<2x1xf16>,
                                     %zp: tensor<2x1xui8>,
                                     %init: tensor<2x2xf16>) -> tensor<2x2xf16>
    attributes {rock.kernel} {
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        zero_points(%zp : tensor<2x1xui8>)
        outs(%init : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 1 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// CHECK-LABEL: func.func @matmul_nbits_fp16_zp_bias
// CHECK: tosa.matmul
// CHECK: tosa.add
// CHECK-NOT: hip.matmul_nbits
func.func @matmul_nbits_fp16_zp_bias(%ctx: !hip.context, %a: tensor<2x16xf16>,
                                     %b: tensor<2x1x8xui8>,
                                     %scales: tensor<2x1xf16>,
                                     %zp: tensor<2x1xf16>,
                                     %bias: tensor<2xf16>,
                                     %init: tensor<2x2xf16>) -> tensor<2x2xf16>
    attributes {rock.kernel} {
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        zero_points(%zp : tensor<2x1xf16>)
        bias(%bias : tensor<2xf16>)
        outs(%init : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 2 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// Rank-3 A collapses leading dims the same way hip.matmul does.
// CHECK-LABEL: func.func @matmul_nbits_batched
// CHECK: tosa.matmul
// CHECK-NOT: hip.matmul_nbits
func.func @matmul_nbits_batched(%ctx: !hip.context, %a: tensor<1x2x16xf16>,
                                %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>,
                                %init: tensor<1x2x2xf16>) -> tensor<1x2x2xf16>
    attributes {rock.kernel} {
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<1x2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        outs(%init : tensor<1x2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<1x2x2xf16>
  return %r : tensor<1x2x2xf16>
}

// CHECK-LABEL: func.func @matmul_nbits_poison_ctx
// CHECK: tosa.matmul
// CHECK-NOT: hip.matmul_nbits
func.func @matmul_nbits_poison_ctx(%a: tensor<2x16xf16>, %b: tensor<2x1x8xui8>,
                                   %scales: tensor<2x1xf16>,
                                   %init: tensor<2x2xf16>) -> tensor<2x2xf16>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        outs(%init : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// ONNX MatMulNBits allows f32 scales with f16 activations; cast scales before
// dequant rather than requiring the element types to already match.
// CHECK-LABEL: func.func @matmul_nbits_f32_scales
// CHECK: tosa.cast
// CHECK: tosa.mul
// CHECK: tosa.matmul
// CHECK-NOT: hip.matmul_nbits
func.func @matmul_nbits_f32_scales(%ctx: !hip.context, %a: tensor<2x16xf16>,
                                   %b: tensor<2x1x8xui8>,
                                   %scales: tensor<2x1xf32>,
                                   %init: tensor<2x2xf16>) -> tensor<2x2xf16>
    attributes {rock.kernel} {
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf32>)
        outs(%init : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64,
         scale_elem_size = 4 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// -----

func.func @matmul_nbits_bits8(%ctx: !hip.context, %a: tensor<2x16xf16>,
                              %b: tensor<2x1x16xui8>, %scales: tensor<2x1xf16>,
                              %init: tensor<2x2xf16>) -> tensor<2x2xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.matmul_nbits'}}
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x16xui8>, tensor<2x1xf16>)
        outs(%init : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 8 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// -----

func.func @matmul_nbits_gidx(%ctx: !hip.context, %a: tensor<2x16xf16>,
                             %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>,
                             %g: tensor<16xi32>, %init: tensor<2x2xf16>)
    -> tensor<2x2xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.matmul_nbits'}}
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<2x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        g_idx(%g : tensor<16xi32>)
        outs(%init : tensor<2x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<2x2xf16>
  return %r : tensor<2x2xf16>
}

// -----

func.func @matmul_nbits_dynamic(%ctx: !hip.context, %a: tensor<?x16xf16>,
                                %b: tensor<2x1x8xui8>, %scales: tensor<2x1xf16>,
                                %init: tensor<?x2xf16>) -> tensor<?x2xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.matmul_nbits'}}
  %r = hip.matmul_nbits(%ctx) ins(%a, %b, %scales :
        tensor<?x16xf16>, tensor<2x1x8xui8>, tensor<2x1xf16>)
        outs(%init : tensor<?x2xf16>)
        {K = 16 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
         accuracy_level = 0 : i64, zp_elem_size = 0 : i64} : tensor<?x2xf16>
  return %r : tensor<?x2xf16>
}
