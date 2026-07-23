// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// hipsr.cast's populateShapeRegion() emits the shape computation, checked via
// -hipsr-populate-shape-region. Pass-level behavior lives in
// populate_shape_region.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -hipsr-populate-shape-region | FileCheck %s

// Empty region populated: shape.shape_of (over the entry-block arg) + one
// get_extent per dim, yielded with the output element type.
// CHECK-LABEL: func.func @cast_tensor
// CHECK:       hipsr.cast(%{{.+}}) ins(%{{.+}} : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) : tensor<?x8xf16> shape_region {
// CHECK:         ^bb0(%[[IN:.+]]: tensor<?x8xf32>):
// CHECK:         %[[SHAPE:.+]] = shape.shape_of %[[IN]]
// CHECK:         %[[D0:.+]] = shape.get_extent %[[SHAPE]]
// CHECK:         %[[D1:.+]] = shape.get_extent %[[SHAPE]]
// CHECK:         hipsr.shape_yield (%[[D0]], %[[D1]]) : [f16]
// CHECK:       }
func.func @cast_tensor(%ctx: !hipsr.context, %input: tensor<?x8xf32>, %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>) outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}
