// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Both runtime entry points carry the float width as an argument and implement
// f32 and f16 alike, so a cast between those two widths next to a Q/DQ op is
// redundant: the Q/DQ op names the other width and the cast goes away. Every
// positive case therefore expects the cast to be gone with no fused op in its
// place, and the surviving Q/DQ op to carry the cast's float type.
//
// The negative cases pin the two preconditions: a cast the runtime cannot
// express, and a dequantize whose float result is still wanted at its own
// width.
//
// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @dequantize_cast_to_f16
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x4xui16>) -> tensor<2x4xf16> {
// CHECK:         %[[INIT:.*]] = tensor.empty() : tensor<2x4xf16>
// CHECK:         %[[DQ:.*]] = hip.dequantize_linear(%[[CTX]]) ins(%[[X]], {{.*}}) zero_point({{.*}}) outs(%[[INIT]] : tensor<2x4xf16>) {{.*}} : tensor<2x4xf16>
// CHECK-NOT:     hip.cast
// CHECK:         return %[[DQ]] : tensor<2x4xf16>
func.func @dequantize_cast_to_f16(%ctx: !hip.context,
                                  %x: tensor<2x4xui16>) -> tensor<2x4xf16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x4xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x4xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<2x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x4xf32>

  %e1 = tensor.empty() : tensor<2x4xf16>
  %c = hip.cast(%ctx) ins(%dq : tensor<2x4xf32>)
      outs(%e1 : tensor<2x4xf16>) {to = 10 : i64} : tensor<2x4xf16>

  return %c : tensor<2x4xf16>
}

// -----

// The quantize reads the cast's source, so its own result type does not move
// and no init has to be rebuilt.
// CHECK-LABEL: func.func @cast_to_f32_quantize
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x4xf16>) -> tensor<2x4xui16> {
// CHECK:         %[[Q:.*]] = hip.quantize_linear(%[[CTX]]) ins(%[[X]], {{.*}} : tensor<2x4xf16>, {{.*}})
// CHECK-NOT:     hip.cast
// CHECK:         return %[[Q]] : tensor<2x4xui16>
func.func @cast_to_f32_quantize(%ctx: !hip.context,
                                %x: tensor<2x4xf16>) -> tensor<2x4xui16> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x4xf32>
  %c = hip.cast(%ctx) ins(%x : tensor<2x4xf16>)
      outs(%e0 : tensor<2x4xf32>) {to = 1 : i64} : tensor<2x4xf32>

  %e1 = tensor.empty() : tensor<2x4xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%c, %scale : tensor<2x4xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e1 : tensor<2x4xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<2x4xui16>

  return %q : tensor<2x4xui16>
}

// -----

// i32 is outside the float side of the Q/DQ dispatch, so the dequantize cannot
// take the cast over and both ops have to survive.
// CHECK-LABEL: func.func @dequantize_cast_to_i32
// CHECK:         hip.dequantize_linear
// CHECK-SAME:    tensor<2x4xf32>
// CHECK:         hip.cast
func.func @dequantize_cast_to_i32(%ctx: !hip.context,
                                  %x: tensor<2x4xui16>) -> tensor<2x4xi32> {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x4xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x4xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<2x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x4xf32>

  %e1 = tensor.empty() : tensor<2x4xi32>
  %c = hip.cast(%ctx) ins(%dq : tensor<2x4xf32>)
      outs(%e1 : tensor<2x4xi32>) {to = 6 : i64} : tensor<2x4xi32>

  return %c : tensor<2x4xi32>
}

// -----

// The f32 result is returned as well, so retyping the dequantize would make
// the graph dequantize twice.
// CHECK-LABEL: func.func @dequantize_cast_second_user
// CHECK:         hip.dequantize_linear
// CHECK-SAME:    tensor<2x4xf32>
// CHECK:         hip.cast
func.func @dequantize_cast_second_user(%ctx: !hip.context,
                                       %x: tensor<2x4xui16>)
    -> (tensor<2x4xf16>, tensor<2x4xf32>) {
  %scale = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<2x4xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %scale : tensor<2x4xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>)
      outs(%e0 : tensor<2x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x4xf32>

  %e1 = tensor.empty() : tensor<2x4xf16>
  %c = hip.cast(%ctx) ins(%dq : tensor<2x4xf32>)
      outs(%e1 : tensor<2x4xf16>) {to = 10 : i64} : tensor<2x4xf16>

  return %c, %dq : tensor<2x4xf16>, tensor<2x4xf32>
}
