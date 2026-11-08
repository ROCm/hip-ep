// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.quantize_linear / hip.dequantize_linear decompose to TOSA
// arithmetic inside a rock.kernel function, so rocMLIR can absorb them into a
// fused kernel.
//
// Dequantize: (cast(x) - cast(zp)) * scale
// Quantize:   saturate(round(x * reciprocal(scale)) + zp), rounding half to
//             even with a plain tosa.cast to i32. The fp_to_int_cast custom
//             op truncates, as ONNX Cast does, and would be off by one.
//
// rocMLIR takes only signless integers, so a kernel whose unsigned values
// only meet signedness-neutral ops is retyped to signless; the rocmlir
// unsigned_cast custom op keeps the unsigned meaning.
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

// Quantize without zp: reciprocal, mul, round to i32, clamp, narrow.
// CHECK-LABEL: func.func @quantize_no_zp
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK-NOT: fp_to_int_cast
// CHECK: tosa.cast {{.*}} -> tensor<2x8xi32>
// CHECK: tosa.clamp {{.*}} {max_val = 127 : i32, min_val = -128 : i32
// CHECK: tosa.cast {{.*}} -> tensor<2x8xi8>
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

// Quantize with zp: round to i32, add, clamp, cast to i8.
// CHECK-LABEL: func.func @quantize_with_zp
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK-NOT: fp_to_int_cast
// CHECK: tosa.cast {{.*}} -> tensor<2x8xi32>
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

// Rank-0 data with a length-1 vector scale: reshape the scale to a scalar
// so tosa.mul has matching ranks.
// CHECK-LABEL: func.func @dequantize_rank0_scale1
// CHECK: tosa.reshape %arg2
// CHECK: tosa.mul
// CHECK-NOT: hip.dequantize_linear
func.func @dequantize_rank0_scale1(%ctx: !hip.context, %x: tensor<i8>,
                                     %scale: tensor<1xf32>,
                                     %init: tensor<f32>) -> tensor<f32>
    attributes {rock.kernel} {
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<i8>, tensor<1xf32>)
         outs(%init : tensor<f32>)
         {axis = 0 : i64, block_size = 0 : i64} : tensor<f32>
  return %r : tensor<f32>
}

// A ui8 dequantize reaches rocMLIR as i8 bytes. The zero point keeps its bit
// pattern (128 reads back as -128 signless), and unsigned_cast widens it as
// unsigned.
// CHECK-LABEL: func.func @dequantize_ui8_signless
// CHECK-SAME: (%{{.*}}: tensor<2x8xi8>, %{{.*}}: tensor<f32>) -> tensor<2x8xf32>
// CHECK-NOT: ui8
// CHECK: tosa.const{{.*}}dense<-128>
// CHECK-NOT: ui8
// CHECK: tosa.custom {{.*}}operator_name = "unsigned_cast"
// CHECK-NOT: ui8
// CHECK: return
func.func @dequantize_ui8_signless(%x: tensor<2x8xui8>, %scale: tensor<f32>)
    -> tensor<2x8xf32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %zp = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %init = tensor.empty() : tensor<2x8xf32>
  %r = hip.dequantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xui8>, tensor<f32>)
         zero_point(%zp : tensor<ui8>)
         outs(%init : tensor<2x8xf32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// A ui8 quantize returns i8 bytes; the i32 result narrows through
// unsigned_cast after the [0, 255] clamp.
// CHECK-LABEL: func.func @quantize_ui8_signless
// CHECK-SAME: -> tensor<2x8xi8>
// CHECK: tosa.clamp {{.*}} {max_val = 255 : i32, min_val = 0 : i32
// CHECK: tosa.custom {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<2x8xi8>
// CHECK-NOT: ui8
// CHECK: return
func.func @quantize_ui8_signless(%x: tensor<2x8xf32>, %scale: tensor<f32>)
    -> tensor<2x8xui8> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %zp = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %init = tensor.empty() : tensor<2x8xui8>
  %r = hip.quantize_linear(%ctx)
         ins(%x, %scale : tensor<2x8xf32>, tensor<f32>)
         zero_point(%zp : tensor<ui8>)
         outs(%init : tensor<2x8xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<2x8xui8>
  return %r : tensor<2x8xui8>
}

// An unsigned value that reaches an op whose meaning depends on signedness
// keeps its type, so the kernel is not silently reinterpreted.
// CHECK-LABEL: func.func @ui8_max_keeps_type
// CHECK-SAME: tensor<2x8xui8>
func.func @ui8_max_keeps_type(%a: tensor<2x8xui8>, %b: tensor<2x8xui8>)
    -> tensor<2x8xui8> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %init = tensor.empty() : tensor<2x8xui8>
  %r = hip.max(%ctx) ins(%a, %b : tensor<2x8xui8>, tensor<2x8xui8>)
         outs(%init : tensor<2x8xui8>) : tensor<2x8xui8>
  return %r : tensor<2x8xui8>
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
