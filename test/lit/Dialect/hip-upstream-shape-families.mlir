// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --test-hip-whole-shape-dim-reify %s | FileCheck %s

// CHECK-LABEL: func.func @swish_source
// CHECK-SAME: %[[INPUT:[A-Za-z0-9_]+]]: tensor<?x8xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: return %[[D0]], %[[C8]]
func.func @swish_source(%ctx: !hip.context, %input: tensor<?x8xf32>,
    %output: tensor<?x?xf32>) -> (index, index) {
  %r = hip.swish(%ctx)
    ins(%input : tensor<?x8xf32>)
    outs(%output : tensor<?x?xf32>)
    {alpha = 0.5 : f64} : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: func.func @quantize_source
// CHECK-SAME: %[[INPUT:[A-Za-z0-9_]+]]: tensor<?x8xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: return %[[D0]], %[[C8]]
func.func @quantize_source(%ctx: !hip.context, %input: tensor<?x8xf32>,
    %scale: tensor<f32>, %output: tensor<?x?xui8>) -> (index, index) {
  %r = hip.quantize_linear(%ctx)
    ins(%input, %scale : tensor<?x8xf32>, tensor<f32>)
    outs(%output : tensor<?x?xui8>)
    {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64, saturate = 1 : i64} : tensor<?x?xui8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xui8>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xui8>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: func.func @dequantize_source
// CHECK-SAME: %[[INPUT:[A-Za-z0-9_]+]]: tensor<?x8xi8>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: return %[[D0]], %[[C8]]
func.func @dequantize_source(%ctx: !hip.context, %input: tensor<?x8xi8>,
    %scale: tensor<f32>, %output: tensor<?x?xf32>) -> (index, index) {
  %r = hip.dequantize_linear(%ctx)
    ins(%input, %scale : tensor<?x8xi8>, tensor<f32>)
    outs(%output : tensor<?x?xf32>)
    {axis = 1 : i64, block_size = 0 : i64} : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: func.func @qlp_source
// CHECK-SAME: %[[INPUT:[A-Za-z0-9_]+]]: tensor<?x8xui16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: return %[[D0]], %[[C8]]
func.func @qlp_source(%ctx: !hip.context, %input: tensor<?x8xui16>,
    %output: tensor<?x?xui16>) -> (index, index) {
  %r = hip.qlpnormalization(%ctx)
    ins(%input : tensor<?x8xui16>)
    outs(%output : tensor<?x?xui16>)
    {axis = -1 : i64, p = 2 : i64, input_scale = 0.1 : f32, input_zp = 3 : i64, output_scale = 0.2 : f32, output_zp = 7 : i64} : tensor<?x?xui16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xui16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xui16>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: func.func @qsigmoid_source
// CHECK-SAME: %[[INPUT:[A-Za-z0-9_]+]]: tensor<?x8xui16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK: return %[[D0]], %[[C8]]
func.func @qsigmoid_source(%ctx: !hip.context, %input: tensor<?x8xui16>,
    %output: tensor<?x?xui16>) -> (index, index) {
  %r = hip.qsigmoid(%ctx)
    ins(%input : tensor<?x8xui16>)
    outs(%output : tensor<?x?xui16>)
    {input_scale = 0.1 : f32, input_zp = 3 : i64, output_scale = 0.2 : f32, output_zp = 7 : i64} : tensor<?x?xui16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xui16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xui16>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: func.func @qmoe_amd_source
// CHECK-SAME: %[[HS:[A-Za-z0-9_]+]]: tensor<1x?x16xf16>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[HS]], %[[C1]]
// CHECK: return %[[D1]], %[[C16]]
func.func @qmoe_amd_source(%ctx: !hip.context, %hs: tensor<1x?x16xf16>,
    %ew: tensor<4x8x1x4xui8>, %es: tensor<4x8x1xf16>,
    %l1w: tensor<8x2x4xui8>, %l1s: tensor<8x2xf16>,
    %l2w: tensor<16x1x4xui8>, %l2s: tensor<16x1xf16>,
    %sw: tensor<16x2x4xui8>, %ss: tensor<16x2xf16>,
    %router: tensor<16x4xf16>, %correction: tensor<4xf16>,
    %out: tensor<1x?x?xf16>) -> (index, index) {
  %r = hip.qmoe_amd(%ctx)
    ins(%hs, %ew, %es, %ew, %es, %l1w, %l1s, %l2w, %l2s,
        %sw, %ss, %sw, %ss, %router, %correction :
        tensor<1x?x16xf16>, tensor<4x8x1x4xui8>, tensor<4x8x1xf16>,
        tensor<4x8x1x4xui8>, tensor<4x8x1xf16>,
        tensor<8x2x4xui8>, tensor<8x2xf16>,
        tensor<16x1x4xui8>, tensor<16x1xf16>,
        tensor<16x2x4xui8>, tensor<16x2xf16>,
        tensor<16x2x4xui8>, tensor<16x2xf16>,
        tensor<16x4xf16>, tensor<4xf16>)
    outs(%out : tensor<1x?x?xf16>)
    {k = 2 : i64, expert_weight_bits = 4 : i64, block_size = 8 : i64,
     normalize_routing_weights = 1 : i64, use_correction_bias = 1 : i64,
     routed_scaling_factor = 5.0 : f32,
     activation_type = "relu2", routing_type = "sigmoid"}
    : tensor<1x?x?xf16>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d1 = tensor.dim %r, %c1 : tensor<1x?x?xf16>
  %d2 = tensor.dim %r, %c2 : tensor<1x?x?xf16>
  return %d1, %d2 : index, index
}

func.func private @kernel()

// CHECK-LABEL: func.func @rocmlir_destination
// CHECK-SAME: %[[OUT:[A-Za-z0-9_]+]]: tensor<?x4xf32>
// CHECK: %[[DIM:.*]] = tensor.dim %[[OUT]], %{{.*}} : tensor<?x4xf32>
// CHECK: return %[[DIM]]
func.func @rocmlir_destination(%ctx: !hip.context,
    %input: tensor<?x8xf32>, %out: tensor<?x4xf32>) -> index {
  %r = hip.rocmlir(%ctx) @kernel ins(%input : tensor<?x8xf32>)
    outs(%out : tensor<?x4xf32>) : tensor<?x4xf32>
  %c0 = arith.constant 0 : index
  %d = tensor.dim %r, %c0 : tensor<?x4xf32>
  return %d : index
}

// CHECK-LABEL: func.func @qconv_destination
// CHECK-SAME: %[[OUT:[A-Za-z0-9_]+]]: tensor<1x4x1x?xui16>
// CHECK: %[[DIM:.*]] = tensor.dim %[[OUT]], %{{.*}} : tensor<1x4x1x?xui16>
// CHECK: return %[[DIM]]
func.func @qconv_destination(%ctx: !hip.context,
    %input: tensor<1x8x1x?xui16>, %weights: tensor<4x8x1x1xi8>,
    %scales: tensor<4xf32>, %zp: tensor<4xi8>,
    %out: tensor<1x4x1x?xui16>) -> index {
  %r = hip.qconv(%ctx)
    ins(%input, %weights, %scales, %zp :
        tensor<1x8x1x?xui16>, tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>)
    outs(%out : tensor<1x4x1x?xui16>)
    {input_scale = 0.1 : f32, input_zp = 3 : i64,
     output_scale = 0.2 : f32, output_zp = 7 : i64,
     weight_axis = 0 : i64, kernel_shape = [1, 1],
     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
     group = 1 : i64, packed_int4} : tensor<1x4x1x?xui16>
  %c3 = arith.constant 3 : index
  %d = tensor.dim %r, %c3 : tensor<1x4x1x?xui16>
  return %d : index
}
