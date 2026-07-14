// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Negative test for ShapeRegionInterface's scoping verifier
// (verifyShapeRegionScoping), exercised through hipsr.cast. The verifier only
// runs on interface-implementing ops, so a real op is required to trigger it.
// Valid ops are covered by the round-trip test (cast.mlir).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -verify-diagnostics

// A value defined in the enclosing function that is NOT one of the op's
// operands is a disallowed outer capture: verifyShapeRegionScoping rejects it.
func.func @cast_scoping_disallowed_outer(%input: tensor<?x8xf32>,
                                         %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %outer = arith.constant 8 : index
  // expected-error@+1 {{shape region references disallowed outer value}}
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
                  shape_region {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %input, %c0 : tensor<?x8xf32>
    // expected-note@+1 {{used here by 'hipsr.shape_yield'}}
    hipsr.shape_yield %d0, %outer
  }
  return %0 : tensor<?x8xf16>
}
