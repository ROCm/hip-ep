// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pass-level behavior of -hipsr-populate-shape-region: what the pass does over a
// function, independent of any single op's shape math (that is checked per op in
// cast.mlir, matmul.mlir, ...). Two behaviors:
//   - idempotency: a region that is already populated is left untouched;
//   - whole-function walk: every empty ShapeRegionInterface region in the
//     function, across op kinds, is filled in a single run.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-populate-shape-region | FileCheck %s

// Idempotency: the pass fills only empty regions, so an op that already carries a
// (here hand-written) shape region is left byte-for-byte untouched -- the
// generated shape.shape_of form must never be emitted over it.

// CHECK-LABEL: func.func @already_populated
func.func @already_populated(%ctx: !hipsr.context, %input: tensor<4x8xf32>, %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // CHECK: hipsr.cast
  // CHECK: shape_region {
  // CHECK:   %[[C4:.+]] = arith.constant 4 : index
  // CHECK:   %[[C8:.+]] = arith.constant 8 : index
  // CHECK:   hipsr.shape_yield (%[[C4]], %[[C8]]) : [f16]
  // CHECK-NOT: shape.shape_of
  %0 = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>) outs(%init : tensor<4x8xf16>) : tensor<4x8xf16> shape_region {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%c4, %c8) : [f16]
  }
  return %0 : tensor<4x8xf16>
}

// -----

// Whole-function walk: the pass visits every ShapeRegionInterface op in the
// function and fills each empty region in one run, regardless of op kind. Here a
// cast feeds a matmul; both are populated, each emitting its own op-specific
// shape computation (cast: shape_of + get_extent; matmul: the K-equality guard).

// CHECK-LABEL: func.func @whole_function_walk
func.func @whole_function_walk(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                               %cinit: tensor<?x8xf16>, %b: tensor<8x16xf16>,
                               %minit: tensor<?x16xf16>) -> tensor<?x16xf16> {
  // CHECK: hipsr.cast(%{{.+}}) ins(%{{.+}} : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
  // CHECK:   shape.shape_of
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f16]
  // CHECK: }
  // CHECK: hipsr.matmul(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x8xf16>, tensor<8x16xf16>) outs(%{{.+}} : tensor<?x16xf16>) : tensor<?x16xf16> shape_region {
  // CHECK:   shape.cstr_eq
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f16]
  // CHECK: }
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>) outs(%cinit : tensor<?x8xf16>) : tensor<?x8xf16>
  %1 = hipsr.matmul(%ctx) ins(%0, %b : tensor<?x8xf16>, tensor<8x16xf16>) outs(%minit : tensor<?x16xf16>) : tensor<?x16xf16>
  return %1 : tensor<?x16xf16>
}
