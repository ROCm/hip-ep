// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Shape-region verifier, exercised through hipsr.cast. Covered here:
//   - positive: region omitted round-trips; populated region reads its input
//     via the entry-block arg
//   - isolation: IsolatedFromAbove requires reads to go through entry-block args
//   - structure: 0-or-1 blocks, non-empty, hipsr.shape_yield terminator
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Positive round-trip: a cast with the shape region omitted parses, verifies,
// and prints back with no `shape_region` keyword. This is the form the
// onnx->hipsr conversion emits.
// CHECK-LABEL: func.func @cast_no_shape_region
// CHECK:     hipsr.cast(%{{.+}}) ins(%{{.+}} : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NOT: shape_region
func.func @cast_no_shape_region(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                                %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}

// -----

// Positive: a populated region reads its input via the entry-block arg (arg 0),
// not the op's operand (IsolatedFromAbove).
// CHECK-LABEL: func.func @cast_uses_block_arg
// CHECK:       hipsr.cast
// CHECK:         ^bb0(%[[IN:.+]]: tensor<?x8xf32>):
// CHECK:         tensor.dim %[[IN]]
func.func @cast_uses_block_arg(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                               %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
                  shape_region {
  ^bb0(%in: tensor<?x8xf32>):
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %in, %c0 : tensor<?x8xf32>
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%d0, %c8) : [f16]
  }
  return %0 : tensor<?x8xf16>
}

// -----

// Isolation: IsolatedFromAbove rejects referencing an enclosing-scope value --
// not even the op's own operand %input (must use the entry-block arg).
func.func @cast_scoping_disallowed_outer(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                                         %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // expected-note@+1 {{required by region isolation constraints}}
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
                  shape_region {
  ^bb0(%in: tensor<?x8xf32>):
    %c0 = arith.constant 0 : index
    // Capturing the op operand %input (an outside value) is illegal.
    // expected-error@+1 {{using value defined outside the region}}
    %d0 = tensor.dim %input, %c0 : tensor<?x8xf32>
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%d0, %c8) : [f16]
  }
  return %0 : tensor<?x8xf16>
}

// -----

// A present shape region must not be a lone empty block (absence is the omitted
// region, not an empty block).
func.func @cast_empty_block(%ctx: !hipsr.context, %input: tensor<4x8xf32>,
                            %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{expects a non-empty block}}
  %0 = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) : tensor<4x8xf16> shape_region {
  }
  return %0 : tensor<4x8xf16>
}

// -----

// The shape region may hold at most one block.
func.func @cast_two_blocks(%ctx: !hipsr.context, %input: tensor<4x8xf32>,
                           %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  %0 = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) : tensor<4x8xf16> shape_region {
    hipsr.shape_yield () : [f16]
  ^bb1:
    hipsr.shape_yield () : [f16]
  }
  return %0 : tensor<4x8xf16>
}

// -----

// A non-empty shape region block must end with a terminator.
func.func @cast_missing_terminator(%ctx: !hipsr.context, %input: tensor<4x8xf32>,
                                   %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{shape region block must end with a terminator}}
  %0 = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) : tensor<4x8xf16> shape_region {
    %c0 = arith.constant 0 : index
  }
  return %0 : tensor<4x8xf16>
}

// -----

// A non-empty shape region must terminate with hipsr.shape_yield, not some
// other terminator.
func.func @cast_wrong_terminator(%ctx: !hipsr.context, %input: tensor<4x8xf32>,
                                 %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{shape region must terminate with hipsr.shape_yield, got 'cf.br'}}
  %0 = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) : tensor<4x8xf16> shape_region {
  ^bb0:
    cf.br ^bb0
  }
  return %0 : tensor<4x8xf16>
}

// -----

// The entry-block args must match the op's shape-region operand list. cast is a
// normal op, so it takes only its data input -- an extra leading ctx arg is
// rejected.
func.func @cast_wrong_arg_count(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                                %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // expected-error@+1 {{shape region block must have one argument per shape-region operand (expected 1, got 2)}}
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
  ^bb0(%ctxarg: !hipsr.context, %in: tensor<?x8xf32>):
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %in, %c0 : tensor<?x8xf32>
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%d0, %c8) : [f16]
  }
  return %0 : tensor<?x8xf16>
}
