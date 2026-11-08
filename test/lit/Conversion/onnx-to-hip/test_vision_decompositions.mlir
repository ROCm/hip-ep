// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
    return %arg0 : tensor<1x3x8x8xf32>
  }

  func.func @hard_swish(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %y = "onnx.HardSwish"(%x) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @hard_swish
  // CHECK-NOT: onnx.HardSwish
  // CHECK: hip.mul
  // CHECK: hip.max
  // CHECK: hip.mul

  // The emitted elementwise ops have no f64 runtime path, so the op stays
  // for another EP.
  func.func @hard_swish_f64_unconverted(%x: tensor<2x4xf64>) -> tensor<2x4xf64> {
    %y = "onnx.HardSwish"(%x) : (tensor<2x4xf64>) -> tensor<2x4xf64>
    return %y : tensor<2x4xf64>
  }

  // CHECK-LABEL: func.func @hard_swish_f64_unconverted
  // CHECK: onnx.HardSwish
  // CHECK-NOT: hip.mul

  func.func @batch_norm(%x: tensor<1x4x2x2xf32>, %scale: tensor<4xf32>, %bias: tensor<4xf32>, %mean: tensor<4xf32>, %var: tensor<4xf32>) -> tensor<1x4x2x2xf32> {
    %y = "onnx.BatchNormalization"(%x, %scale, %bias, %mean, %var) {epsilon = 1.000000e-05 : f32, momentum = 9.000000e-01 : f32, training_mode = 0 : i64} : (tensor<1x4x2x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x2x2xf32>
    return %y : tensor<1x4x2x2xf32>
  }

  // CHECK-LABEL: func.func @batch_norm
  // CHECK-NOT: onnx.BatchNormalization
  // CHECK: tensor.expand_shape
  // CHECK: hip.sub
  // CHECK: hip.sqrt
  // CHECK: hip.div
  // CHECK: hip.mul
  // CHECK: hip.add

  func.func @einsum_qk(%q: tensor<1x2x3x4xf32>, %k: tensor<1x2x5x4xf32>) -> tensor<1x2x3x5xf32> {
    %y = "onnx.Einsum"(%q, %k) {equation = "bhlk,bhtk->bhlt"} : (tensor<1x2x3x4xf32>, tensor<1x2x5x4xf32>) -> tensor<1x2x3x5xf32>
    return %y : tensor<1x2x3x5xf32>
  }

  // CHECK-LABEL: func.func @einsum_qk
  // CHECK-NOT: onnx.Einsum
  // CHECK: hip.matmul
  // CHECK-SAME: transB = 1

  func.func @einsum_av(%a: tensor<1x2x3x5xf32>, %v: tensor<1x2x5x6xf32>) -> tensor<1x2x3x6xf32> {
    %y = "onnx.Einsum"(%a, %v) {equation = "bhlt,bhtv->bhlv"} : (tensor<1x2x3x5xf32>, tensor<1x2x5x6xf32>) -> tensor<1x2x3x6xf32>
    return %y : tensor<1x2x3x6xf32>
  }

  // CHECK-LABEL: func.func @einsum_av
  // CHECK-NOT: onnx.Einsum
  // CHECK: hip.matmul

  func.func @einsum_bhwc_hkc(%a: tensor<2x3x4x5xf32>, %b: tensor<3x6x5xf32>) -> tensor<2x3x4x6xf32> {
    %y = "onnx.Einsum"(%a, %b) {equation = "bhwc,hkc->bhwk"} : (tensor<2x3x4x5xf32>, tensor<3x6x5xf32>) -> tensor<2x3x4x6xf32>
    return %y : tensor<2x3x4x6xf32>
  }

  // CHECK-LABEL: func.func @einsum_bhwc_hkc
  // CHECK: hip.transpose
  // CHECK: tensor.expand_shape
  // CHECK: hip.matmul
  // CHECK-NOT: onnx.Einsum

  func.func @einsum_bhwc_wkc(%a: tensor<2x3x4x5xf32>, %b: tensor<4x6x5xf32>) -> tensor<2x3x4x6xf32> {
    %y = "onnx.Einsum"(%a, %b) {equation = "bhwc,wkc->bhwk"} : (tensor<2x3x4x5xf32>, tensor<4x6x5xf32>) -> tensor<2x3x4x6xf32>
    return %y : tensor<2x3x4x6xf32>
  }

  // CHECK-LABEL: func.func @einsum_bhwc_wkc
  // CHECK: hip.matmul
  // CHECK: hip.transpose
  // CHECK-NOT: onnx.Einsum

  func.func @depth_to_space_crd(%x: tensor<1x12x2x2xf32>) -> tensor<1x3x4x4xf32> {
    %y = "onnx.DepthToSpace"(%x) {blocksize = 2 : si64, mode = "CRD"} : (tensor<1x12x2x2xf32>) -> tensor<1x3x4x4xf32>
    return %y : tensor<1x3x4x4xf32>
  }

  // CHECK-LABEL: func.func @depth_to_space_crd
  // CHECK: tensor.expand_shape
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 1, 4, 2, 5, 3]
  // CHECK: tensor.collapse_shape
  // CHECK-NOT: onnx.DepthToSpace

  func.func @depth_to_space_dcr(%x: tensor<1x12x2x2xf32>) -> tensor<1x3x4x4xf32> {
    %y = "onnx.DepthToSpace"(%x) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x12x2x2xf32>) -> tensor<1x3x4x4xf32>
    return %y : tensor<1x3x4x4xf32>
  }

  // CHECK-LABEL: func.func @depth_to_space_dcr
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 3, 4, 1, 5, 2]
  // CHECK-NOT: onnx.DepthToSpace
}
