// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Exercises ShapeRegionInterface's scoping verifier (verifyShapeRegionScoping)
// through hipsr.matmul, the first op implementing the interface. The verifier
// only runs on interface-implementing ops, so a real op (not an unregistered
// placeholder) is required to trigger it.
//
// verifyShapeRegionStructure is a defensive backstop for ops that forget the
// SingleBlockImplicitTerminator<"ShapeYieldOp"> trait; on a real op that trait
// (and SizedRegion<1>) verify structure first, so it is not separately
// exercised here.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// A shape region may reference the op's own operands (%lhs, %rhs) and values
// defined inside the region. This is the allowed baseline.

// CHECK-LABEL: func.func @scoping_operands_ok
func.func @scoping_operands_ok(%lhs: tensor<?x256xf16>, %rhs: tensor<256x512xf16>,
                               %init: tensor<?x512xf16>) -> tensor<?x512xf16> {
  // CHECK: hipsr.matmul
  %0 = hipsr.matmul ins(%lhs, %rhs : tensor<?x256xf16>, tensor<256x512xf16>)
                    outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16>
                    shape_region {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x256xf16>
    %n = tensor.dim %rhs, %c1 : tensor<256x512xf16>
    hipsr.shape_yield %m, %n
  }
  return %0 : tensor<?x512xf16>
}

// -----

// Values defined in a region nested inside the shape region (here an scf.if)
// are allowed: the shape region is an ancestor of the nested region.

// CHECK-LABEL: func.func @scoping_nested_region_ok
func.func @scoping_nested_region_ok(%lhs: tensor<?x256xf16>, %rhs: tensor<256x512xf16>,
                                    %init: tensor<?x512xf16>) -> tensor<?x512xf16> {
  // CHECK: hipsr.matmul
  %0 = hipsr.matmul ins(%lhs, %rhs : tensor<?x256xf16>, tensor<256x512xf16>)
                    outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16>
                    shape_region {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pred = arith.constant true
    %m = scf.if %pred -> index {
      %d = tensor.dim %lhs, %c0 : tensor<?x256xf16>
      scf.yield %d : index
    } else {
      scf.yield %c0 : index
    }
    %n = tensor.dim %rhs, %c1 : tensor<256x512xf16>
    hipsr.shape_yield %m, %n
  }
  return %0 : tensor<?x512xf16>
}

// -----

// A value defined in the enclosing function that is NOT one of the op's
// operands is a disallowed outer capture: verifyShapeRegionScoping rejects it.

func.func @scoping_disallowed_outer(%lhs: tensor<?x256xf16>, %rhs: tensor<256x512xf16>,
                                    %init: tensor<?x512xf16>) -> tensor<?x512xf16> {
  %outer = arith.constant 512 : index
  // expected-error@+1 {{shape region references disallowed outer value}}
  %0 = hipsr.matmul ins(%lhs, %rhs : tensor<?x256xf16>, tensor<256x512xf16>)
                    outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16>
                    shape_region {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x256xf16>
    // expected-note@+1 {{used here by 'hipsr.shape_yield'}}
    hipsr.shape_yield %m, %outer
  }
  return %0 : tensor<?x512xf16>
}
