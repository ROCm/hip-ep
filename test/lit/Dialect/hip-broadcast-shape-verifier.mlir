// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid_static(%ctx: !hip.context,
                        %lhs: memref<2x1xf32, 1>,
                        %rhs: memref<1x3xf32, 1>,
                        %out: memref<2x3xf32, 1>) {
  hip.sub(%ctx)
    ins(%lhs, %rhs : memref<2x1xf32, 1>, memref<1x3xf32, 1>)
    outs(%out : memref<2x3xf32, 1>)
  return
}

// -----

func.func @dynamic_legal(%ctx: !hip.context,
                         %lhs: memref<?x3xf32, 1>,
                         %rhs: memref<2x1xf32, 1>,
                         %out: memref<?x3xi1, 1>) {
  hip.less(%ctx)
    ins(%lhs, %rhs : memref<?x3xf32, 1>, memref<2x1xf32, 1>)
    outs(%out : memref<?x3xi1, 1>)
  return
}

// -----

func.func @incompatible_inputs(%ctx: !hip.context,
                               %lhs: memref<2x3xf32, 1>,
                               %rhs: memref<4x3xf32, 1>,
                               %out: memref<4x3xf32, 1>) {
  // expected-error @+1 {{incompatible broadcast shapes [2, 3] and [4, 3]}}
  hip.div(%ctx)
    ins(%lhs, %rhs : memref<2x3xf32, 1>, memref<4x3xf32, 1>)
    outs(%out : memref<4x3xf32, 1>)
  return
}

// -----

func.func @wrong_output_extent(%ctx: !hip.context,
                               %lhs: memref<2x1xf32, 1>,
                               %rhs: memref<1x3xf32, 1>,
                               %out: memref<2x4xf32, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 3}}
  hip.max(%ctx)
    ins(%lhs, %rhs : memref<2x1xf32, 1>, memref<1x3xf32, 1>)
    outs(%out : memref<2x4xf32, 1>)
  return
}

// -----

func.func @result_init_type_mismatch(%ctx: !hip.context,
                                     %lhs: tensor<2x3xf32>,
                                     %rhs: tensor<2x3xf32>,
                                     %out: tensor<2x3xf32>) {
  // expected-error @+1 {{must match DPS init type #0}}
  %result = hip.add(%ctx)
    ins(%lhs, %rhs : tensor<2x3xf32>, tensor<2x3xf32>)
    outs(%out : tensor<2x3xf32>) -> tensor<2x?xf32>
  return
}

// -----

func.func @mixed_tensor_memref_mode(%ctx: !hip.context,
                                    %lhs: tensor<2x3xf32>,
                                    %rhs: memref<2x3xf32, 1>,
                                    %out: tensor<2x3xf32>) {
  // expected-error @+1 {{all data operands must be the same kind (all tensor or all memref)}}
  %result = hip.add(%ctx)
    ins(%lhs, %rhs : tensor<2x3xf32>, memref<2x3xf32, 1>)
    outs(%out : tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}
