// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics --hip-add-context-arg --convert-ggml-to-hip | FileCheck %s

// The code reads an input row-major, given its extents but not its strides.
// A copy the ggml scheduler made of a transposed view keeps the view's
// strides: read row-major, its elements would be taken in the wrong order.

#dense = #ggml.layout<ne = [4, 3, 1, 1], nb = [4, 16, 48, 48], offset = 0>
#transposed = #ggml.layout<ne = [4, 3, 1, 1], nb = [12, 4, 48, 48], offset = 0>

// expected-error @+1 {{function input 2 is not proven dense in row-major order}}
func.func @transposed(%lhs: tensor<1x1x3x4xf32, #dense>,
                      %rhs: tensor<1x1x3x4xf32, #transposed>)
    -> tensor<1x1x3x4xf32, #dense> {
  %sum = "ggml.add"(%lhs, %rhs)
    : (tensor<1x1x3x4xf32, #dense>, tensor<1x1x3x4xf32, #transposed>)
      -> tensor<1x1x3x4xf32, #dense>
  return %sum : tensor<1x1x3x4xf32, #dense>
}

// -----

// A window of wider rows has gaps between its rows.

#dense = #ggml.layout<ne = [4, 3, 1, 1], nb = [4, 16, 48, 48], offset = 0>
#window = #ggml.layout<ne = [4, 3, 1, 1], nb = [4, 32, 96, 96], offset = 0>

// expected-error @+1 {{function input 2 is not proven dense in row-major order}}
func.func @window(%lhs: tensor<1x1x3x4xf32, #dense>,
                  %rhs: tensor<1x1x3x4xf32, #window>)
    -> tensor<1x1x3x4xf32, #dense> {
  %product = "ggml.mul"(%lhs, %rhs)
    : (tensor<1x1x3x4xf32, #dense>, tensor<1x1x3x4xf32, #window>)
      -> tensor<1x1x3x4xf32, #dense>
  return %product : tensor<1x1x3x4xf32, #dense>
}

// -----

// Trailing dimensions of one element step nowhere, so their strides do not
// make an input strided.

#dense = #ggml.layout<ne = [4, 3, 1, 1], nb = [4, 16, 48, 48], offset = 0>
#unit = #ggml.layout<ne = [4, 3, 1, 1], nb = [4, 16, 1024, 4096], offset = 0>

// CHECK-LABEL: func.func @dense(
// CHECK: hip.add
// CHECK-NOT: ggml.
func.func @dense(%lhs: tensor<1x1x3x4xf32, #dense>,
                 %rhs: tensor<1x1x3x4xf32, #unit>)
    -> tensor<1x1x3x4xf32, #dense> {
  %sum = "ggml.add"(%lhs, %rhs)
    : (tensor<1x1x3x4xf32, #dense>, tensor<1x1x3x4xf32, #unit>)
      -> tensor<1x1x3x4xf32, #dense>
  return %sum : tensor<1x1x3x4xf32, #dense>
}
