// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// Packed bytes on a surviving quantize axis become logical elements.
// CHECK-LABEL: func.func @reify_static_packed_axis
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C192:.*]] = arith.constant 192 : index
// CHECK: return %[[C8]], %[[C192]] : index, index
func.func @reify_static_packed_axis(
    %ctx: !hip.context,
    %data: tensor<2048x96xui8>,
    %indices: tensor<8xi64>,
    %scales: tensor<2048x12xf16>,
    %init: tensor<8x192xf16>) -> (index, index) {
  %result = hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>)
    outs(%init : tensor<8x192xf16>)
    {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
    : tensor<8x192xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<8x192xf16>
  %d1 = tensor.dim %result, %c1 : tensor<8x192xf16>
  return %d0, %d1 : index, index
}

// -----

// CHECK-LABEL: func.func @reify_dynamic_packed_axis
// CHECK-SAME: %[[DATA:[A-Za-z0-9_]+]]: tensor<2048x?xi8>
// CHECK: %[[PACKED:.*]] = tensor.dim %[[DATA]], %{{.*}} : tensor<2048x?xi8>
// CHECK: %[[LOGICAL:.*]] = arith.muli %[[PACKED]], %{{.*}} : index
// CHECK: return %{{.*}}, %[[LOGICAL]] : index, index
func.func @reify_dynamic_packed_axis(
    %ctx: !hip.context,
    %data: tensor<2048x?xi8>,
    %indices: tensor<?xi64>,
    %scales: tensor<2048x?xf16>,
    %init: tensor<?x?xf16>) -> (index, index) {
  %result = hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        tensor<2048x?xi8>, tensor<?xi64>, tensor<2048x?xf16>)
    outs(%init : tensor<?x?xf16>)
    {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
    : tensor<?x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf16>
  return %d0, %d1 : index, index
}

// -----

// Gathering the packed quantize axis removes it, so no indices extent is
// doubled. The surviving dynamic prefix still comes from data.
// CHECK-LABEL: func.func @reify_gathered_quantize_axis
// CHECK-SAME: %[[DATA:[A-Za-z0-9_]+]]: tensor<?x96xi8>
// CHECK: %[[PREFIX:.*]] = tensor.dim %[[DATA]], %{{.*}} : tensor<?x96xi8>
// CHECK-NOT: arith.muli
// CHECK: return %[[PREFIX]], %{{.*}} : index, index
func.func @reify_gathered_quantize_axis(
    %ctx: !hip.context,
    %data: tensor<?x96xi8>,
    %indices: tensor<4xi64>,
    %scales: tensor<?x12xf16>,
    %init: tensor<?x4xf16>) -> (index, index) {
  %result = hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        tensor<?x96xi8>, tensor<4xi64>, tensor<?x12xf16>)
    outs(%init : tensor<?x4xf16>)
    {bits = 4, block_size = 16, gather_axis = 1, quantize_axis = 1}
    : tensor<?x4xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x4xf16>
  %d1 = tensor.dim %result, %c1 : tensor<?x4xf16>
  return %d0, %d1 : index, index
}

// -----

// CHECK-LABEL: func.func @reify_static_packed_i64_boundary
// CHECK: %[[BOUNDARY:.*]] = arith.constant 9223372036854775806 : index
// CHECK: return %[[BOUNDARY]] : index
func.func @reify_static_packed_i64_boundary(
    %ctx: !hip.context,
    %data: tensor<4x4611686018427387903xi8>,
    %indices: tensor<3xi64>,
    %scales: tensor<4x?xf16>,
    %init: tensor<3x9223372036854775806xf16>) -> index {
  %result = hip.gather_block_quantized(%ctx)
    ins(%data, %indices, %scales :
        tensor<4x4611686018427387903xi8>, tensor<3xi64>, tensor<4x?xf16>)
    outs(%init : tensor<3x9223372036854775806xf16>)
    {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
    : tensor<3x9223372036854775806xf16>
  %c1 = arith.constant 1 : index
  %d1 = tensor.dim %result, %c1 : tensor<3x9223372036854775806xf16>
  return %d1 : index
}
