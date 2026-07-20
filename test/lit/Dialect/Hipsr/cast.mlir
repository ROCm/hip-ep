// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// hipsr.cast shape region, exercised through -hipsr-populate-shape-region: the
// op's ShapeRegionInterface::populateShapeRegion() emits the shape computation
// end-to-end (generated, not hand-written). Also checks the pass leaves an
// already-populated region untouched (idempotent).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-populate-shape-region | FileCheck %s

// A hipsr.cast entered with an empty (omitted) shape region gets its region
// populated: shape.shape_of + one shape.get_extent per dim, yielded as a single
// group with the output element type.

// CHECK-LABEL: func.func @cast_tensor
func.func @cast_tensor(%input: tensor<?x8xf32>, %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // CHECK: hipsr.cast ins(%[[IN:.+]] : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) -> tensor<?x8xf16> shape_region {
  // CHECK:   %[[SHAPE:.+]] = shape.shape_of %[[IN]]
  // CHECK:   %[[D0:.+]] = shape.get_extent %[[SHAPE]]
  // CHECK:   %[[D1:.+]] = shape.get_extent %[[SHAPE]]
  // CHECK:   hipsr.shape_yield (%[[D0]], %[[D1]]) : [f16]
  // CHECK: }
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>) outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}

// -----

// Idempotent: an already-populated region is left untouched (the pass only
// fills empty regions).

// CHECK-LABEL: func.func @cast_already_populated
func.func @cast_already_populated(%input: tensor<4x8xf32>, %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // CHECK: hipsr.cast
  // CHECK: shape_region {
  // CHECK:   %[[C4:.+]] = arith.constant 4 : index
  // CHECK:   %[[C8:.+]] = arith.constant 8 : index
  // CHECK:   hipsr.shape_yield (%[[C4]], %[[C8]]) : [f16]
  // The pass must not re-emit the generated `shape.shape_of` form over the
  // hand-written region.
  // CHECK-NOT: shape.shape_of
  %0 = hipsr.cast ins(%input : tensor<4x8xf32>) outs(%init : tensor<4x8xf16>) -> tensor<4x8xf16> shape_region {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    hipsr.shape_yield (%c4, %c8) : [f16]
  }
  return %0 : tensor<4x8xf16>
}

// -----

// The pass walks the whole function: every ShapeRegionInterface op gets its own
// empty region filled in a single run. Here two chained casts (f32->f16->f32)
// are both populated, each yielding its own output element type.

// CHECK-LABEL: func.func @cast_chain
func.func @cast_chain(%input: tensor<?x8xf32>, %init0: tensor<?x8xf16>,
                      %init1: tensor<?x8xf32>) -> tensor<?x8xf32> {
  // CHECK: hipsr.cast ins(%{{.+}} : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) -> tensor<?x8xf16> shape_region {
  // CHECK:   shape.shape_of
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f16]
  // CHECK: }
  // CHECK: hipsr.cast ins(%{{.+}} : tensor<?x8xf16>) outs(%{{.+}} : tensor<?x8xf32>) -> tensor<?x8xf32> shape_region {
  // CHECK:   shape.shape_of
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f32]
  // CHECK: }
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>) outs(%init0 : tensor<?x8xf16>) -> tensor<?x8xf16>
  %1 = hipsr.cast ins(%0 : tensor<?x8xf16>) outs(%init1 : tensor<?x8xf32>) -> tensor<?x8xf32>
  return %1 : tensor<?x8xf32>
}
