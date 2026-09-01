// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Packing is a pre-lowering ONNX pattern set inside convert-onnx-to-hip.
// Safe static rank > 4 binary ops collapse to rank <= 4, then hip.*, then
// expand. Unsafe, dynamic, and unranked cases leave the original rank (no
// collapse/expand). Shorter operands are right-aligned with leading 1s via
// tensor.expand_shape before collapse.
//
// Unranked tensors cannot become hip.add (the HIP contract requires ranked
// operands), so that case is a separate split-file that verifies the ONNX op
// remains available for fallback after packing and compute conversion decline
// it without mutating the IR.

// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/static.mlir --hip-add-context-arg --convert-onnx-to-hip | FileCheck %t/static.mlir
// RUN: hip-mlir-opt %t/unranked.mlir --hip-add-context-arg --convert-onnx-to-hip | FileCheck %t/unranked.mlir

//--- static.mlir
module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Adjacent axes mix broadcast and data, so no contiguous group is fully
  // broadcast or fully present. Rank 6 cannot pack into 4 groups.
  func.func @keep_unsafe_rank6_mul(
      %lhs: tensor<2x1x3x1x5x1xf32>,
      %rhs: tensor<1x2x1x4x1x6xf32>)
      -> tensor<2x2x3x4x5x6xf32> {
    // CHECK-LABEL: func.func @keep_unsafe_rank6_mul
    // CHECK-NOT: tensor.collapse_shape
    // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>) outs({{.*}} : tensor<2x2x3x4x5x6xf32>)
    // CHECK-NOT: tensor.expand_shape
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>)
        -> tensor<2x2x3x4x5x6xf32>
    return %result : tensor<2x2x3x4x5x6xf32>
  }

  func.func @keep_unsafe_rank5_add(
      %lhs: tensor<2x1x3x1x5xf32>,
      %rhs: tensor<1x4x1x6x1xf32>)
      -> tensor<2x4x3x6x5xf32> {
    // CHECK-LABEL: func.func @keep_unsafe_rank5_add
    // CHECK-NOT: tensor.collapse_shape
    // CHECK: hip.add({{.*}}) ins({{.*}}, {{.*}} : tensor<2x1x3x1x5xf32>, tensor<1x4x1x6x1xf32>) outs({{.*}} : tensor<2x4x3x6x5xf32>)
    // CHECK-NOT: tensor.expand_shape
    %result = "onnx.Add"(%lhs, %rhs) :
        (tensor<2x1x3x1x5xf32>, tensor<1x4x1x6x1xf32>)
        -> tensor<2x4x3x6x5xf32>
    return %result : tensor<2x4x3x6x5xf32>
  }

  func.func @keep_dynamic_rank5_add(
      %lhs: tensor<?x2x3x4x5xf32>,
      %rhs: tensor<1x2x3x4x5xf32>)
      -> tensor<?x2x3x4x5xf32> {
    // CHECK-LABEL: func.func @keep_dynamic_rank5_add
    // CHECK-NOT: tensor.collapse_shape
    // CHECK: hip.add({{.*}}) ins({{.*}}, {{.*}} : tensor<?x2x3x4x5xf32>, tensor<1x2x3x4x5xf32>) outs({{.*}} : tensor<?x2x3x4x5xf32>)
    // CHECK-NOT: tensor.expand_shape
    %result = "onnx.Add"(%lhs, %rhs) :
        (tensor<?x2x3x4x5xf32>, tensor<1x2x3x4x5xf32>)
        -> tensor<?x2x3x4x5xf32>
    return %result : tensor<?x2x3x4x5xf32>
  }

  func.func @right_align_rank1_to_5(
      %lhs: tensor<2x3x4x5x6xf32>, %rhs: tensor<6xf32>)
      -> tensor<2x3x4x5x6xf32> {
    // CHECK-LABEL: func.func @right_align_rank1_to_5
    // CHECK: %[[ALIGNED_RHS:.*]] = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2, 3, 4]] output_shape [1, 1, 1, 1, 6] : tensor<6xf32> into tensor<1x1x1x1x6xf32>
    // CHECK: tensor.collapse_shape %[[ALIGNED_RHS]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<1x1x1x1x6xf32> into tensor<1x1x1x6xf32>
    // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<2x3x20x6xf32>, tensor<1x1x1x6xf32>) outs({{.*}} : tensor<2x3x20x6xf32>)
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x3x4x5x6xf32>, tensor<6xf32>)
        -> tensor<2x3x4x5x6xf32>
    return %result : tensor<2x3x4x5x6xf32>
  }

  func.func @right_align_rank2_to_6(
      %lhs: tensor<2x3x4x5x6x7xf32>, %rhs: tensor<6x7xf32>)
      -> tensor<2x3x4x5x6x7xf32> {
    // CHECK-LABEL: func.func @right_align_rank2_to_6
    // CHECK: %[[ALIGNED_RHS:.*]] = tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2, 3, 4], [5]] output_shape [1, 1, 1, 1, 6, 7] : tensor<6x7xf32> into tensor<1x1x1x1x6x7xf32>
    // CHECK: tensor.collapse_shape %[[ALIGNED_RHS]] {{\[\[}}0], [1], [2, 3], [4, 5]] : tensor<1x1x1x1x6x7xf32> into tensor<1x1x1x42xf32>
    // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<2x3x20x42xf32>, tensor<1x1x1x42xf32>) outs({{.*}} : tensor<2x3x20x42xf32>)
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x3x4x5x6x7xf32>, tensor<6x7xf32>)
        -> tensor<2x3x4x5x6x7xf32>
    return %result : tensor<2x3x4x5x6x7xf32>
  }

  func.func @same_rank_no_pad(
      %lhs: tensor<2x3x4x5x6x7xf32>,
      %rhs: tensor<2x3x4x5x6x7xf32>)
      -> tensor<2x3x4x5x6x7xf32> {
    // CHECK-LABEL: func.func @same_rank_no_pad
    // CHECK-NOT: tensor.expand_shape {{.*}} : tensor<2x3x4x5x6x7xf32>
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] : tensor<2x3x4x5x6x7xf32> into tensor<2x3x4x210xf32>
    // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<2x3x4x210xf32>, tensor<2x3x4x210xf32>) outs({{.*}} : tensor<2x3x4x210xf32>)
    // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4, 5]] output_shape [2, 3, 4, 5, 6, 7] : tensor<2x3x4x210xf32> into tensor<2x3x4x5x6x7xf32>
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x3x4x5x6x7xf32>, tensor<2x3x4x5x6x7xf32>)
        -> tensor<2x3x4x5x6x7xf32>
    return %result : tensor<2x3x4x5x6x7xf32>
  }
}

//--- unranked.mlir
module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Packing and HIP compute conversion both require ranked operands. They
  // leave this ONNX operation unchanged for a later fallback partition.
  func.func @keep_unranked_lhs_add(
      %lhs: tensor<*xf32>,
      %rhs: tensor<2x3x4x5x6xf32>)
      -> tensor<2x3x4x5x6xf32> {
    // CHECK-NOT: tensor.collapse_shape
    // CHECK: "onnx.Add"
    // CHECK-NOT: hip.add
    %result = "onnx.Add"(%lhs, %rhs) :
        (tensor<*xf32>, tensor<2x3x4x5x6xf32>)
        -> tensor<2x3x4x5x6xf32>
    return %result : tensor<2x3x4x5x6xf32>
  }
}
