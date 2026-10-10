// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s > %t.before
// RUN: hip-mlir-opt --hip-infer-shapes --verify-diagnostics %s > %t.after
// RUN: diff %t.before %t.after

// The tensor types pass static broadcast validation. Reifying the slices folds
// their last dimensions to incompatible constants. Failure must discard all
// operations, including the merge emitted for the first dimension.
func.func @late_axis_conflict(%ctx: !hip.context, %m: index, %n: index,
                             %lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                             %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %a = tensor.extract_slice %lhs[0, 0] [%m, %c2] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>
  %b = tensor.extract_slice %rhs[0, 0] [%n, %c3] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>
  // expected-error @+1 {{incompatible broadcast dimensions 2 and 3}}
  %r = hip.add(%ctx)
    ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// The first two operands are compatible. The third operand exposes a conflict.
// The dimensions queried from %cond and both earlier merges must also disappear.
func.func @late_operand_conflict(%ctx: !hip.context, %m: index, %n: index,
                                %cond: tensor<?x?xi1>,
                                %lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                                %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %a = tensor.extract_slice %lhs[0, 0] [%m, %c2] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>
  %b = tensor.extract_slice %rhs[0, 0] [%n, %c3] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>
  // expected-error @+1 {{incompatible broadcast dimensions 2 and 3}}
  %r = hip.where(%ctx)
    ins(%cond, %a, %b : tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%out : tensor<?x?xf32>) : tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
