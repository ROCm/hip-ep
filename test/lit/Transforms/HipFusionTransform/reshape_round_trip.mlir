// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @expand_collapse
// CHECK-SAME: %[[X:[^, :]+]]: tensor<2x6xf32>
// CHECK-NOT: tensor.
// CHECK: return %[[X]] : tensor<2x6xf32>
func.func @expand_collapse(%x: tensor<2x6xf32>) -> tensor<2x6xf32> {
  %a = tensor.expand_shape %x [[0], [1, 2]] output_shape [2, 2, 3] : tensor<2x6xf32> into tensor<2x2x3xf32>
  %b = tensor.collapse_shape %a [[0], [1, 2]] : tensor<2x2x3xf32> into tensor<2x6xf32>
  return %b : tensor<2x6xf32>
}

// -----

// CHECK-LABEL: func.func @collapse_expand
// CHECK-SAME: %[[X:[^, :]+]]: tensor<2x3x4xf32>
// CHECK-NOT: tensor.
// CHECK: return %[[X]] : tensor<2x3x4xf32>
func.func @collapse_expand(%x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %a = tensor.collapse_shape %x [[0, 1], [2]] : tensor<2x3x4xf32> into tensor<6x4xf32>
  %b = tensor.expand_shape %a [[0, 1], [2]] output_shape [2, 3, 4] : tensor<6x4xf32> into tensor<2x3x4xf32>
  return %b : tensor<2x3x4xf32>
}

// -----

// The first reshape still serves another consumer.
// CHECK-LABEL: func.func @shared
// CHECK-SAME: %[[X:[^, :]+]]: tensor<2x6xf32>
// CHECK: %[[A:.*]] = tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK: return %[[A]], %[[X]]
func.func @shared(%x: tensor<2x6xf32>) -> (tensor<2x2x3xf32>, tensor<2x6xf32>) {
  %a = tensor.expand_shape %x [[0], [1, 2]] output_shape [2, 2, 3] : tensor<2x6xf32> into tensor<2x2x3xf32>
  %b = tensor.collapse_shape %a [[0], [1, 2]] : tensor<2x2x3xf32> into tensor<2x6xf32>
  return %a, %b : tensor<2x2x3xf32>, tensor<2x6xf32>
}

// -----

// A different final type is not an identity.
// CHECK-LABEL: func.func @different_type
// CHECK: tensor.expand_shape
// CHECK: tensor.collapse_shape
func.func @different_type(%x: tensor<2x6xf32>) -> tensor<4x3xf32> {
  %a = tensor.expand_shape %x [[0], [1, 2]] output_shape [2, 2, 3] : tensor<2x6xf32> into tensor<2x2x3xf32>
  %b = tensor.collapse_shape %a [[0, 1], [2]] : tensor<2x2x3xf32> into tensor<4x3xf32>
  return %b : tensor<4x3xf32>
}

// -----

// Mixed tensor.reshape / expand_shape pairs are not folded by the generic
// reshape folders.
// CHECK-LABEL: func.func @mixed_expand_reshape
// CHECK-SAME: %[[X:[^, :]+]]: tensor<2x6xf32>
// CHECK-NOT: tensor.
// CHECK: return %[[X]] : tensor<2x6xf32>
func.func @mixed_expand_reshape(%x: tensor<2x6xf32>, %s: tensor<2xindex>) -> tensor<2x6xf32> {
  %a = tensor.expand_shape %x [[0], [1, 2]] output_shape [2, 2, 3] : tensor<2x6xf32> into tensor<2x2x3xf32>
  %b = tensor.reshape %a(%s) : (tensor<2x2x3xf32>, tensor<2xindex>) -> tensor<2x6xf32>
  return %b : tensor<2x6xf32>
}

// -----

// CHECK-LABEL: func.func @mixed_reshape_collapse
// CHECK-SAME: %[[X:[^, :]+]]: tensor<2x6xf32>
// CHECK-NOT: tensor.
// CHECK: return %[[X]] : tensor<2x6xf32>
func.func @mixed_reshape_collapse(%x: tensor<2x6xf32>, %s: tensor<3xindex>) -> tensor<2x6xf32> {
  %a = tensor.reshape %x(%s) : (tensor<2x6xf32>, tensor<3xindex>) -> tensor<2x2x3xf32>
  %b = tensor.collapse_shape %a [[0], [1, 2]] : tensor<2x2x3xf32> into tensor<2x6xf32>
  return %b : tensor<2x6xf32>
}

// -----

// Equal dynamic types do not pin the shape down.
// CHECK-LABEL: func.func @dynamic
// CHECK: tensor.expand_shape
// CHECK: tensor.reshape
func.func @dynamic(%x: tensor<?x6xf32>, %n: index, %s: tensor<2xindex>) -> tensor<?x6xf32> {
  %a = tensor.expand_shape %x [[0], [1, 2]] output_shape [%n, 2, 3] : tensor<?x6xf32> into tensor<?x2x3xf32>
  %b = tensor.reshape %a(%s) : (tensor<?x2x3xf32>, tensor<2xindex>) -> tensor<?x6xf32>
  return %b : tensor<?x6xf32>
}
