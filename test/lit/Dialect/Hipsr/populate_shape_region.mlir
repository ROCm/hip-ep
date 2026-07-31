// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: hipsr
//
//===----------------------------------------------------------------------===//
// Pass-level behavior of -hipsr-populate-shape-region (per-op shape math is
// checked in cast.mlir, matmul.mlir, ...):
//   - idempotency: an already-populated region is left untouched;
//   - whole-function walk: every empty region is filled in a single run.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-populate-shape-region | FileCheck %s

// The pass fills only empty regions, so this hand-written region is left
// untouched (no generated shape.shape_of).
// CHECK-LABEL: func.func @already_populated
// CHECK:       hipsr.cast
// CHECK:       shape_region {
// CHECK:         %[[C4:.+]] = arith.constant 4 : index
// CHECK:         %[[C8:.+]] = arith.constant 8 : index
// CHECK:         hipsr.shape_yield (%[[C4]], %[[C8]]) : [f16]
// CHECK-NOT:   shape.shape_of
func.func @already_populated(%ctx: !hipsr.context, %input: tensor<4x8xf32>, %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %0 = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>) outs(%init : tensor<4x8xf16>) : tensor<4x8xf16> shape_region {
  ^bb0(%in: tensor<4x8xf32>):
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%c4, %c8) : [f16]
  }
  return %0 : tensor<4x8xf16>
}

// -----

// A cast feeds a matmul; the pass fills both empty regions in one run, each
// with its own shape computation (cast: shape_of; matmul: K-equality guard).
// CHECK-LABEL: func.func @whole_function_walk
// CHECK:       hipsr.cast(%{{.+}}) ins(%{{.+}} : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
// CHECK:         shape.shape_of
// CHECK:         hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f16]
// CHECK:       }
// CHECK:       hipsr.matmul(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x8xf16>, tensor<8x16xf16>) outs(%{{.+}} : tensor<?x16xf16>) : tensor<?x16xf16> shape_region {
// CHECK:         shape.cstr_eq
// CHECK:         hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f16]
// CHECK:       }
func.func @whole_function_walk(%ctx: !hipsr.context, %input: tensor<?x8xf32>,
                               %cinit: tensor<?x8xf16>, %b: tensor<8x16xf16>,
                               %minit: tensor<?x16xf16>) -> tensor<?x16xf16> {
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>) outs(%cinit : tensor<?x8xf16>) : tensor<?x8xf16>
  %1 = hipsr.matmul(%ctx) ins(%0, %b : tensor<?x8xf16>, tensor<8x16xf16>) outs(%minit : tensor<?x16xf16>) : tensor<?x16xf16>
  return %1 : tensor<?x16xf16>
}
