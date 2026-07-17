// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// hipsr.cast shape region, exercised through -hipsr-populate-shape-region: the
// op's ShapeRegionInterface::populateShapeRegion() emits the shape computation
// end-to-end (generated, not hand-written), for both tensor and device-memref
// inputs. Also checks the pass leaves an already-populated region untouched
// (idempotent).
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

// Bufferized DPS form: all-memref operands, no tensor result. The result clause
// is optional, so the op parses with no `->`, and populateShapeRegion fills its
// empty region -- taking the output element type from the `init` operand (there
// is no result to read).

// CHECK-LABEL: func.func @cast_bufferized
func.func @cast_bufferized(%input: memref<4x8xf32, #hipsr.mem<device>>,
                           %init: memref<4x8xf16, #hipsr.mem<device>>) {
  // CHECK: hipsr.cast ins(%[[IN:.+]] : memref<4x8xf32, #hipsr.mem<device>>) outs(%{{.+}} : memref<4x8xf16, #hipsr.mem<device>>) shape_region {
  // CHECK:   %[[SHAPE:.+]] = shape.shape_of %[[IN]] : memref<4x8xf32, #hipsr.mem<device>>
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f16]
  hipsr.cast ins(%input : memref<4x8xf32, #hipsr.mem<device>>)
             outs(%init : memref<4x8xf16, #hipsr.mem<device>>)
  return
}

// -----

// Idempotent: an already-populated region is left untouched (the pass only
// fills empty regions).

// CHECK-LABEL: func.func @cast_already_populated
func.func @cast_already_populated(%input: tensor<4x8xf32>, %init: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // CHECK: hipsr.cast
  // CHECK: shape_region {
  // CHECK: hipsr.shape_yield () : [f16]
  %0 = hipsr.cast ins(%input : tensor<4x8xf32>) outs(%init : tensor<4x8xf16>) -> tensor<4x8xf16> shape_region {
    hipsr.shape_yield () : [f16]
  }
  return %0 : tensor<4x8xf16>
}
