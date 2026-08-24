// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Packing is a pre-lowering ONNX pattern set inside convert-onnx-to-hip,
// alongside GatherShapeFold / ReshapeShapeFold.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<6x2500x8x1x2x4x2xf32>) -> tensor<6x2500x8x1x2x4x2xf32> {
    return %arg0 : tensor<6x2500x8x1x2x4x2xf32>
  }

  // CHECK-LABEL: func.func @pack_rank7_add
  // CHECK: %[[LHS:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3, 4], [5, 6]] : tensor<6x2500x1x1x1x4x2xf32> into tensor<6x2500x1x8xf32>
  // CHECK: %[[RHS:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3, 4], [5, 6]] : tensor<6x2500x8x1x2x4x2xf32> into tensor<6x2500x16x8xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<6x2500x16x8xf32>
  // CHECK: %[[PACKED:.*]] = hip.add(%{{.*}}) ins(%[[LHS]], %[[RHS]] : tensor<6x2500x1x8xf32>, tensor<6x2500x16x8xf32>) outs(%[[INIT]] : tensor<6x2500x16x8xf32>) -> tensor<6x2500x16x8xf32>
  // CHECK: tensor.expand_shape %[[PACKED]] {{\[\[}}0], [1], [2, 3, 4], [5, 6]] output_shape [6, 2500, 8, 1, 2, 4, 2] : tensor<6x2500x16x8xf32> into tensor<6x2500x8x1x2x4x2xf32>
  func.func @pack_rank7_add(
      %lhs: tensor<6x2500x1x1x1x4x2xf32>,
      %rhs: tensor<6x2500x8x1x2x4x2xf32>)
      -> tensor<6x2500x8x1x2x4x2xf32> {
    %result = "onnx.Add"(%lhs, %rhs) :
        (tensor<6x2500x1x1x1x4x2xf32>, tensor<6x2500x8x1x2x4x2xf32>)
        -> tensor<6x2500x8x1x2x4x2xf32>
    return %result : tensor<6x2500x8x1x2x4x2xf32>
  }

  // Adjacent axes mix broadcast and data. Conversion must emit hip.mul at
  // rank 6 and leave rejection to HIP-to-LLVM.
  // CHECK-LABEL: func.func @keep_unsafe_rank6_mul
  // CHECK-NOT: tensor.collapse_shape
  // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>) outs({{.*}} : tensor<2x2x3x4x5x6xf32>)
  // CHECK-NOT: tensor.expand_shape
  func.func @keep_unsafe_rank6_mul(
      %lhs: tensor<2x1x3x1x5x1xf32>,
      %rhs: tensor<1x2x1x4x1x6xf32>)
      -> tensor<2x2x3x4x5x6xf32> {
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>)
        -> tensor<2x2x3x4x5x6xf32>
    return %result : tensor<2x2x3x4x5x6xf32>
  }
}
