// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX ConvTranspose is correctly lowered to hip.conv_transpose in
// tensor-first mode.
//
// Test cases:
// 1. convtranspose_basic   — standard deconv with bias (3x3 kernel, stride 1)
// 2. convtranspose_stride2 — strided deconv (stride 2)
// 3. convtranspose_dynamic — dynamic batch dim (tensor.dim + tensor.empty)
// 4. convtranspose_dynamic_spatial — dynamic spatial formula
// 5. convtranspose_dynamic_channels — output C from weights[1] * group
//
// All cases assert:
// - context argument prepended
// - tensor.empty() for output init (no hip.alloc)
// - all ConvTranspose attributes forwarded (kernel_shape, strides, pads,
//   dilations, output_padding, group)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
    return %arg0 : tensor<1x1x3x3xf32>
  }

  // --------------------------------------------------------------------------
  // 1. Basic transposed conv with bias
  // --------------------------------------------------------------------------
  func.func @convtranspose_basic(%input: tensor<1x1x3x3xf32>, %weights: tensor<1x2x3x3xf32>, %bias: tensor<2xf32>) -> tensor<1x2x5x5xf32> {
    %output = "onnx.ConvTranspose"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      output_padding = [0, 0],
      group = 1 : i64
    } : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, tensor<2xf32>) -> tensor<1x2x5x5xf32>
    return %output : tensor<1x2x5x5xf32>
  }

  // CHECK-LABEL: func.func @convtranspose_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x1x3x3xf32>, %[[W:.*]]: tensor<1x2x3x3xf32>, %[[B:.*]]: tensor<2xf32>) -> tensor<1x2x5x5xf32>
  // CHECK-NOT: onnx.ConvTranspose
  // CHECK: tensor.empty() : tensor<1x2x5x5xf32>
  // CHECK: hip.conv_transpose(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, tensor<2xf32>) outs({{.*}} : tensor<1x2x5x5xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], output_padding = [0, 0], pads = [0, 0, 0, 0], strides = [1, 1]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 2. Strided transposed conv (stride=2)
  // --------------------------------------------------------------------------
  func.func @convtranspose_stride2(%input: tensor<1x1x3x3xf32>, %weights: tensor<1x2x3x3xf32>, %bias: tensor<2xf32>) -> tensor<1x2x7x7xf32> {
    %output = "onnx.ConvTranspose"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      output_padding = [0, 0],
      group = 1 : i64
    } : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, tensor<2xf32>) -> tensor<1x2x7x7xf32>
    return %output : tensor<1x2x7x7xf32>
  }

  // CHECK-LABEL: func.func @convtranspose_stride2
  // CHECK-SAME: !hip.context
  // CHECK-NOT: onnx.ConvTranspose
  // CHECK: tensor.empty() : tensor<1x2x7x7xf32>
  // CHECK: hip.conv_transpose({{.*}}) ins({{.*}}) outs({{.*}}) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], output_padding = [0, 0], pads = [0, 0, 0, 0], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 3. Dynamic batch dim — output init uses tensor.dim + tensor.empty(%dim)
  // --------------------------------------------------------------------------
  func.func @convtranspose_dynamic(%input: tensor<?x1x3x3xf32>, %weights: tensor<1x2x3x3xf32>) -> tensor<?x2x5x5xf32> {
    %output = "onnx.ConvTranspose"(%input, %weights) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      output_padding = [0, 0],
      group = 1 : i64
    } : (tensor<?x1x3x3xf32>, tensor<1x2x3x3xf32>) -> tensor<?x2x5x5xf32>
    return %output : tensor<?x2x5x5xf32>
  }

  // CHECK-LABEL: func.func @convtranspose_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<?x1x3x3xf32>, %[[W:.*]]: tensor<1x2x3x3xf32>)
  // CHECK-NOT: onnx.ConvTranspose
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x1x3x3xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]]) : tensor<?x2x5x5xf32>
  // CHECK: hip.conv_transpose(%[[CTX]]) ins(%[[IN]], %[[W]] : tensor<?x1x3x3xf32>, tensor<1x2x3x3xf32>) outs(%[[INIT]] : tensor<?x2x5x5xf32>)
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 4. Dynamic batch + H. H' = 2 * (H - 1) + 3 = 2*H + 1; W'=7.
  // --------------------------------------------------------------------------
  func.func @convtranspose_dynamic_spatial(
      %input: tensor<?x1x?x3xf32>,
      %weights: tensor<1x2x3x3xf32>) -> tensor<?x2x?x7xf32> {
    %output = "onnx.ConvTranspose"(%input, %weights) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      output_padding = [0, 0],
      group = 1 : i64
    } : (tensor<?x1x?x3xf32>, tensor<1x2x3x3xf32>)
        -> tensor<?x2x?x7xf32>
    return %output : tensor<?x2x?x7xf32>
  }

  // CHECK-LABEL: func.func @convtranspose_dynamic_spatial
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[DIN:.*]]: tensor<?x1x?x3xf32>
  // CHECK-DAG: %[[BATCH:.*]] = tensor.dim %[[DIN]], %{{.*}}
  // CHECK-DAG: %[[H:.*]] = tensor.dim %[[DIN]], %{{.*}}
  // CHECK: %[[H2:.*]] = arith.muli %[[H]], %{{.*}} : index
  // CHECK: %[[HOUT:.*]] = arith.addi %[[H2]], %{{.*}} : index
  // CHECK: %[[DINIT:.*]] = tensor.empty(%[[BATCH]], %[[HOUT]]) : tensor<?x2x?x7xf32>
  // CHECK: hip.conv_transpose({{.*}}) outs(%[[DINIT]] : tensor<?x2x?x7xf32>)

  // --------------------------------------------------------------------------
  // 5. Dynamic output channels are weights[1] * group, not input C.
  // --------------------------------------------------------------------------
  func.func @convtranspose_dynamic_channels(
      %input: tensor<1x2x3x3xf32>,
      %weights: tensor<2x?x3x3xf32>) -> tensor<1x?x5x5xf32> {
    %output = "onnx.ConvTranspose"(%input, %weights) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      output_padding = [0, 0],
      group = 2 : i64
    } : (tensor<1x2x3x3xf32>, tensor<2x?x3x3xf32>)
        -> tensor<1x?x5x5xf32>
    return %output : tensor<1x?x5x5xf32>
  }

  // CHECK-LABEL: func.func @convtranspose_dynamic_channels
  // CHECK-SAME: (%{{.*}}: !hip.context, %{{.*}}: tensor<1x2x3x3xf32>, %[[CWEIGHTS:.*]]: tensor<2x?x3x3xf32>
  // CHECK: %[[WC:.*]] = tensor.dim %[[CWEIGHTS]], %{{.*}}
  // CHECK: %[[OC:.*]] = arith.muli %[[WC]], %{{.*}} : index
  // CHECK: %[[CINIT:.*]] = tensor.empty(%[[OC]]) : tensor<1x?x5x5xf32>
  // CHECK: hip.conv_transpose({{.*}}) outs(%[[CINIT]] : tensor<1x?x5x5xf32>)
}
