// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Exercises ShapeRegionInterface's scoping verifier (verifyShapeRegionScoping)
// through hipsr.cast. The verifier only runs on interface-implementing ops, so
// a real op is required to trigger it.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// A shape region may reference the op's own operands (%input) and values
// defined inside the region.

// CHECK-LABEL: func.func @cast_scoping_operands_ok
func.func @cast_scoping_operands_ok(%input: tensor<?x8xf32>,
                                    %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // CHECK: hipsr.cast
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
                  shape_region {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %d0 = tensor.dim %input, %c0 : tensor<?x8xf32>
    hipsr.shape_yield %d0, %c8
  }
  return %0 : tensor<?x8xf16>
}

// -----

// Values defined in a region nested inside the shape region (scf.if) are
// allowed: the shape region is an ancestor of the nested region.

// CHECK-LABEL: func.func @cast_scoping_nested_region_ok
func.func @cast_scoping_nested_region_ok(%input: tensor<?x8xf32>,
                                         %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // CHECK: hipsr.cast
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
                  shape_region {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %pred = arith.constant true
    %d0 = scf.if %pred -> index {
      %d = tensor.dim %input, %c0 : tensor<?x8xf32>
      scf.yield %d : index
    } else {
      scf.yield %c0 : index
    }
    hipsr.shape_yield %d0, %c8
  }
  return %0 : tensor<?x8xf16>
}

// -----

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
