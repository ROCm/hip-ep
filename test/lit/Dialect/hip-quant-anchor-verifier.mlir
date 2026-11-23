// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.quant_conv and hip.quant_matmul verify the rules of their HipToTosa
// lowering for every static extent: an i8 or ui8 activation (rocMLIR has no
// si8 form), channel counts that fit the group, a kernel shape that matches
// the weights, an output that matches the window, and matmul extents that
// agree. A dynamic extent is left to the lowering. Both take all-tensor or
// all-memref operands, as does hip.rocmlir.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @quant_conv_dynamic_batch
// CHECK: hip.quant_conv
func.func @quant_conv_dynamic_batch(%ctx: !hip.context,
                                    %x: tensor<?x4x6x6xui8>,
                                    %w: tensor<8x2x3x3xi8>,
                                    %e: tensor<?x8x4x4xi32>)
    -> tensor<?x8x4x4xi32> {
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<?x4x6x6xui8>,
         tensor<8x2x3x3xi8>) outs(%e : tensor<?x8x4x4xi32>)
         {dilations = [1, 1], group = 2 : i64, kernel_shape = [3, 3],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<?x8x4x4xi32>
  return %r : tensor<?x8x4x4xi32>
}

// -----

func.func @quant_conv_si8(%ctx: !hip.context, %x: tensor<1x2x4x4xsi8>,
                          %w: tensor<2x2x1x1xi8>, %e: tensor<1x2x4x4xi32>)
    -> tensor<1x2x4x4xi32> {
  // expected-error @+1 {{input must be i8 or ui8}}
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x2x4x4xsi8>,
         tensor<2x2x1x1xi8>) outs(%e : tensor<1x2x4x4xi32>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xi32>
  return %r : tensor<1x2x4x4xi32>
}

// -----

func.func @quant_conv_group(%ctx: !hip.context, %x: tensor<1x3x4x4xi8>,
                            %w: tensor<2x2x1x1xi8>, %e: tensor<1x2x4x4xi32>)
    -> tensor<1x2x4x4xi32> {
  // expected-error @+1 {{input and output channels must be divisible by group}}
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x3x4x4xi8>,
         tensor<2x2x1x1xi8>) outs(%e : tensor<1x2x4x4xi32>)
         {dilations = [1, 1], group = 2 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xi32>
  return %r : tensor<1x2x4x4xi32>
}

// -----

func.func @quant_conv_weight_channels(%ctx: !hip.context,
                                      %x: tensor<1x4x4x4xi8>,
                                      %w: tensor<2x4x1x1xi8>,
                                      %e: tensor<1x2x4x4xi32>)
    -> tensor<1x2x4x4xi32> {
  // expected-error @+1 {{weight input channels must equal C / group}}
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x4x4x4xi8>,
         tensor<2x4x1x1xi8>) outs(%e : tensor<1x2x4x4xi32>)
         {dilations = [1, 1], group = 2 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xi32>
  return %r : tensor<1x2x4x4xi32>
}

// -----

func.func @quant_conv_kernel_shape(%ctx: !hip.context, %x: tensor<1x2x4x4xi8>,
                                   %w: tensor<2x2x1x1xi8>,
                                   %e: tensor<1x2x2x2xi32>)
    -> tensor<1x2x2x2xi32> {
  // expected-error @+1 {{kernel_shape disagrees with the weight tensor}}
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x2x4x4xi8>,
         tensor<2x2x1x1xi8>) outs(%e : tensor<1x2x2x2xi32>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x2x2xi32>
  return %r : tensor<1x2x2x2xi32>
}

// -----

func.func @quant_conv_window(%ctx: !hip.context, %x: tensor<1x2x4x4xi8>,
                             %w: tensor<2x2x3x3xi8>, %e: tensor<1x2x4x4xi32>)
    -> tensor<1x2x4x4xi32> {
  // expected-error @+1 {{output shape disagrees with the convolution window}}
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x2x4x4xi8>,
         tensor<2x2x3x3xi8>) outs(%e : tensor<1x2x4x4xi32>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xi32>
  return %r : tensor<1x2x4x4xi32>
}

// -----

func.func @quant_matmul_si8(%ctx: !hip.context, %a: tensor<2x3xsi8>,
                            %b: tensor<3x5xi8>, %e: tensor<2x5xi32>)
    -> tensor<2x5xi32> {
  // expected-error @+1 {{a must be i8 or ui8}}
  %r = hip.quant_matmul(%ctx) ins(%a, %b : tensor<2x3xsi8>, tensor<3x5xi8>)
         outs(%e : tensor<2x5xi32>) : tensor<2x5xi32>
  return %r : tensor<2x5xi32>
}

// -----

func.func @quant_matmul_contraction(%ctx: !hip.context, %a: tensor<2x3xi8>,
                                    %b: tensor<4x5xi8>, %e: tensor<2x5xi32>)
    -> tensor<2x5xi32> {
  // expected-error @+1 {{b rows must equal a columns}}
  %r = hip.quant_matmul(%ctx) ins(%a, %b : tensor<2x3xi8>, tensor<4x5xi8>)
         outs(%e : tensor<2x5xi32>) : tensor<2x5xi32>
  return %r : tensor<2x5xi32>
}

// -----

func.func @quant_matmul_output(%ctx: !hip.context, %a: tensor<2x3xui8>,
                               %b: tensor<3x5xi8>, %e: tensor<2x4xi32>)
    -> tensor<2x4xi32> {
  // expected-error @+1 {{output shape must be [M, N]}}
  %r = hip.quant_matmul(%ctx) ins(%a, %b : tensor<2x3xui8>, tensor<3x5xi8>)
         outs(%e : tensor<2x4xi32>) : tensor<2x4xi32>
  return %r : tensor<2x4xi32>
}

// -----

func.func @rocmlir_mixed_outs(%ctx: !hip.context, %x: tensor<4xf32>,
                              %t: tensor<4xf32>, %m: memref<4xf32>)
    -> tensor<4xf32> {
  // expected-error @+1 {{requires all outs operands to be tensors or all to be memrefs}}
  %r = hip.rocmlir(%ctx) @k ins(%x : tensor<4xf32>)
         outs(%t, %m : tensor<4xf32>, memref<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @rocmlir_mixed_ins(%ctx: !hip.context, %x: memref<4xf32>,
                             %t: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{requires ins operands of the same kind as its outs (all tensors or all memrefs)}}
  %r = hip.rocmlir(%ctx) @k ins(%x : memref<4xf32>)
         outs(%t : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @quant_conv_mixed(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>,
                            %w: tensor<2x2x1x1xi8>, %e: memref<1x2x4x4xi32>) {
  // expected-error @+1 {{all data operands must be the same kind (all tensor or all memref)}}
  hip.quant_conv(%ctx) ins(%x, %w : tensor<1x2x4x4xui8>, tensor<2x2x1x1xi8>)
      outs(%e : memref<1x2x4x4xi32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]}
  return
}

// -----

func.func @quant_matmul_memref_result(%ctx: !hip.context, %a: memref<2x3xi8>,
                                      %b: memref<3x5xi8>, %e: memref<2x5xi32>)
    -> tensor<2x5xi32> {
  // expected-error @+1 {{memref mode must have zero results, got 1}}
  %r = hip.quant_matmul(%ctx) ins(%a, %b : memref<2x3xi8>, memref<3x5xi8>)
         outs(%e : memref<2x5xi32>) : tensor<2x5xi32>
  return %r : tensor<2x5xi32>
}
