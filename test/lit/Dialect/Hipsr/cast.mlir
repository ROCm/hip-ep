// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// hipsr.cast shape region, exercised through -hipsr-populate-shape-region: the
// op's ShapeRegionInterface::populateShapeRegion() emits the shape computation
// end-to-end (generated, not hand-written). Pass-level behavior (idempotency,
// whole-function walk) lives in populate_shape_region.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-populate-shape-region | FileCheck %s

// A hipsr.cast entered with an empty (omitted) shape region gets its region
// populated: shape.shape_of + one shape.get_extent per dim, yielded as a single
// group with the output element type.

// CHECK-LABEL: func.func @cast_tensor
func.func @cast_tensor(%ctx: !hipsr.context, %input: tensor<?x8xf32>, %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // CHECK: hipsr.cast(%{{.+}}) ins(%[[IN:.+]] : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
  // CHECK:   %[[SHAPE:.+]] = shape.shape_of %[[IN]]
  // CHECK:   %[[D0:.+]] = shape.get_extent %[[SHAPE]]
  // CHECK:   %[[D1:.+]] = shape.get_extent %[[SHAPE]]
  // CHECK:   hipsr.shape_yield (%[[D0]], %[[D1]]) : [f16]
  // CHECK: }
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>) outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}
