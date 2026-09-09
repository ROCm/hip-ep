// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.quantize_linear / hip.dequantize_linear decompose to TOSA
// arithmetic inside a rock.kernel function, so rocMLIR can absorb them into a
// fused kernel.
//
// Dequantize: (cast(x) - cast(zp)) * scale
// Quantize:   saturate(x * reciprocal(scale) + zp), with float->int via
//             tosa.custom fp_to_int_cast (plain tosa.cast is illegal in
//             rock-tosa-to-elementwise).
//
// FILE LAYOUT:
// Converting cases in the first --split-input-file chunk; each rejection
// in its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Per-tensor dequant, no zero_point: cast then mul.
// CHECK-LABEL: func.func @dequantize_no_zp
// CHECK: %[[CAST:.*]] = tosa.cast %arg1 : (tensor<2x8xi8>) -> tensor<2x8xf32>
// CHECK: tosa.mul %[[CAST]]
// CHECK-NOT: hip.dequantize_linear
func.func @dequantize_no_zp(%ctx: !hip.context, %x: tensor<2x8xi8>,
                             %scale: tensor<f32>, %init: tensor<2x8xf32>)
    -> tensor<2x8xf32> attributes {rock.kernel} {
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xi8>, tensor<f32>)
         outs(%init : tensor<2x8xf32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// Per-axis dequant: 1-D scale/zp reshape onto axis 1 before sub/mul.
// CHECK-LABEL: func.func @dequantize_per_axis
// CHECK: tosa.reshape %arg2
// CHECK: tosa.reshape %arg3
// CHECK: tosa.cast %arg1
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: hip.dequantize_linear
func.func @dequantize_per_axis(%ctx: !hip.context, %x: tensor<1x64x32xi8>,
                                %scale: tensor<64xf32>, %zp: tensor<64xi8>,
                                %init: tensor<1x64x32xf32>)
    -> tensor<1x64x32xf32> attributes {rock.kernel} {
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<1x64x32xi8>, tensor<64xf32>)
         zero_point(%zp : tensor<64xi8>)
         outs(%init : tensor<1x64x32xf32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<1x64x32xf32>
  return %r : tensor<1x64x32xf32>
}

// Quantize without zp: reciprocal, mul, fp_to_int_cast.
// CHECK-LABEL: func.func @quantize_no_zp
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.custom {{.*}}operator_name = "fp_to_int_cast"
// CHECK-NOT: hip.quantize_linear
func.func @quantize_no_zp(%ctx: !hip.context, %x: tensor<2x8xf32>,
                            %scale: tensor<f32>, %init: tensor<2x8xi8>)
    -> tensor<2x8xi8> attributes {rock.kernel} {
  %r = hip.quantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xf32>, tensor<f32>)
         outs(%init : tensor<2x8xi8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}

// Quantize with zp: widen to i32, add, clamp, custom-cast to i8.
// CHECK-LABEL: func.func @quantize_with_zp
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.custom {{.*}}operator_name = "fp_to_int_cast"
// CHECK: tosa.add
// CHECK: tosa.clamp
// CHECK: tosa.cast
// CHECK-NOT: hip.quantize_linear
func.func @quantize_with_zp(%ctx: !hip.context, %x: tensor<2x8xf32>,
                            %scale: tensor<f32>, %zp: tensor<i8>,
                            %init: tensor<2x8xi8>) -> tensor<2x8xi8>
    attributes {rock.kernel} {
  %r = hip.quantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xf32>, tensor<f32>)
         zero_point(%zp : tensor<i8>)
         outs(%init : tensor<2x8xi8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}

// CHECK-LABEL: func.func @dequantize_outlined_kernel
// CHECK: tosa.mul
// CHECK-NOT: hip.dequantize_linear
func.func @dequantize_outlined_kernel(%x: tensor<2x8xi8>, %scale: tensor<f32>,
                                     %init: tensor<2x8xf32>)
    -> tensor<2x8xf32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xi8>, tensor<f32>)
         outs(%init : tensor<2x8xf32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @dequantize_not_a_kernel
// CHECK: hip.dequantize_linear
// CHECK-NOT: tosa.mul
func.func @dequantize_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xi8>,
                                    %scale: tensor<f32>,
                                    %init: tensor<2x8xf32>)
    -> tensor<2x8xf32> {
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xi8>, tensor<f32>)
         outs(%init : tensor<2x8xf32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

func.func @dequantize_dynamic(%ctx: !hip.context, %x: tensor<?x8xi8>,
                              %scale: tensor<f32>, %init: tensor<?x8xf32>)
    -> tensor<?x8xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.dequantize_linear'}}
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<?x8xi8>, tensor<f32>)
         outs(%init : tensor<?x8xf32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<?x8xf32>
  return %r : tensor<?x8xf32>
}

// -----

func.func @dequantize_packed_int4(%ctx: !hip.context, %x: tensor<2x8xi8>,
                                   %scale: tensor<f32>, %init: tensor<2x8xf32>)
    -> tensor<2x8xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.dequantize_linear'}}
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xi8>, tensor<f32>)
         outs(%init : tensor<2x8xf32>)
         {axis = 1 : i64, block_size = 0 : i64, packed_int4}
         : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

func.func @quantize_blocked(%ctx: !hip.context, %x: tensor<128x4xf16>,
                              %scale: tensor<4x4xf16>, %init: tensor<128x4xi8>)
    -> tensor<128x4xi8> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.quantize_linear'}}
  %r = hip.quantize_linear(%ctx)
         ins(%x, %scale : tensor<128x4xf16>, tensor<4x4xf16>)
         outs(%init : tensor<128x4xi8>)
         {axis = 0 : i64, block_size = 32 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<128x4xi8>
  return %r : tensor<128x4xi8>
}

// -----

func.func @quantize_no_saturate(%ctx: !hip.context, %x: tensor<2x8xf32>,
                                 %scale: tensor<f32>, %init: tensor<2x8xi8>)
    -> tensor<2x8xi8> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.quantize_linear'}}
  %r = hip.quantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xf32>, tensor<f32>)
         outs(%init : tensor<2x8xi8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 0 : i64} : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}
