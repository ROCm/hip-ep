// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s > %t.before
// RUN: hip-mlir-opt --hip-infer-shapes --verify-diagnostics %s > %t.after
// RUN: diff %t.before %t.after

// Types hide a conflict on the second batch axis. The failed reification must
// discard the dimensions and broadcast merge emitted for the first axis.
func.func @late_batch_conflict(%ctx: !hip.context, %m: index, %n: index,
                               %lhs: tensor<?x?x4x8xf32>,
                               %rhs: tensor<?x?x8x16xf32>,
                               %out: tensor<?x?x4x16xf32>) -> tensor<?x?x4x16xf32> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %a = tensor.extract_slice %lhs[0, 0, 0, 0] [%m, %c2, 4, 8] [1, 1, 1, 1]
    : tensor<?x?x4x8xf32> to tensor<?x?x4x8xf32>
  %b = tensor.extract_slice %rhs[0, 0, 0, 0] [%n, %c3, 8, 16] [1, 1, 1, 1]
    : tensor<?x?x8x16xf32> to tensor<?x?x8x16xf32>
  // expected-error @+1 {{incompatible broadcast dimensions 2 and 3}}
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<?x?x4x8xf32>, tensor<?x?x8x16xf32>)
    outs(%out : tensor<?x?x4x16xf32>) : tensor<?x?x4x16xf32>
  return %r : tensor<?x?x4x16xf32>
}
