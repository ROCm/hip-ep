// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests for the hipsr shape region, exercised through hipsr.cast. The region is
// optional; when present it must be a single, non-empty block ending in
// hipsr.shape_yield. Covered here:
//   - positive: a cast with the region omitted round-trips
//   - scoping: the region may only use the op's operands
//     (IsolatedFromAboveButAllowOperands)
//   - structure: 0-or-1 blocks, non-empty, hipsr.shape_yield terminator
//     (SingleBlock trait + ShapeRegionInterface verifier)
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Positive round-trip: a cast with the shape region OMITTED entirely parses,
// verifies, and prints back with no `shape_region` keyword (the optional region
// group prints nothing when the region has zero blocks). This is the exact form
// the onnx->hipsr conversion emits, so IR without a shape region must read back
// correctly.
// CHECK-LABEL: func.func @cast_no_shape_region
func.func @cast_no_shape_region(%input: tensor<?x8xf32>,
                                %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // CHECK: hipsr.cast ins(%{{.+}} : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) -> tensor<?x8xf16>
  // CHECK-NOT: shape_region
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}

// -----

// A value defined in the enclosing function that is NOT one of the op's
// operands is a disallowed outer capture: the trait verifier rejects it.
func.func @cast_scoping_disallowed_outer(%input: tensor<?x8xf32>,
                                         %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %outer = arith.constant 8 : index
  // expected-note@+1 {{may only use values defined in its regions or the op's operands}}
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
                  shape_region {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %input, %c0 : tensor<?x8xf32>
    // expected-error@+1 {{using value defined outside the region}}
    hipsr.shape_yield (%d0, %outer) : [f16]
  }
  return %0 : tensor<?x8xf16>
}

// -----

// A present shape region must not be a lone empty block. (An absent shape
// region is expressed by omitting the region entirely, not by an empty block.)
func.func @cast_empty_block(%input: tensor<4x8xf32>,
                            %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{expects a non-empty block}}
  %0 = hipsr.cast ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) -> tensor<4x8xf16> shape_region {
  }
  return %0 : tensor<4x8xf16>
}

// -----

// The shape region may hold at most one block.
func.func @cast_two_blocks(%input: tensor<4x8xf32>,
                           %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  %0 = hipsr.cast ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) -> tensor<4x8xf16> shape_region {
    hipsr.shape_yield () : [f16]
  ^bb1:
    hipsr.shape_yield () : [f16]
  }
  return %0 : tensor<4x8xf16>
}

// -----

// A non-empty shape region block must end with a terminator.
func.func @cast_missing_terminator(%input: tensor<4x8xf32>,
                                   %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{shape region block must end with a terminator}}
  %0 = hipsr.cast ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) -> tensor<4x8xf16> shape_region {
    %c0 = arith.constant 0 : index
  }
  return %0 : tensor<4x8xf16>
}

// -----

// A non-empty shape region must terminate with hipsr.shape_yield, not some
// other terminator.
func.func @cast_wrong_terminator(%input: tensor<4x8xf32>,
                                 %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // expected-error@+1 {{shape region must terminate with hipsr.shape_yield, got 'cf.br'}}
  %0 = hipsr.cast ins(%input : tensor<4x8xf32>)
                  outs(%init : tensor<4x8xf16>) -> tensor<4x8xf16> shape_region {
  ^bb0:
    cf.br ^bb0
  }
  return %0 : tensor<4x8xf16>
}
