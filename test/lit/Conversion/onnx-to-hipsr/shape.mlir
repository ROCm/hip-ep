// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Shape to a hipsr.compute whose body reads the input extents.
// Unlike a DPS conversion, this one populates the placeholder's shape region
// itself. Rejected forms live in shape-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// A dynamic input queries only the dynamic axes; the static one is a constant.
// The shape region is a constant too, because the result length is the number
// of selected axes, which the input's rank fixes.
// CHECK-LABEL: func.func @shape_full(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?x4096xf16>) -> tensor<3xi64> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<3xi64> shape_region {
// CHECK-NEXT:      %[[LEN:.*]] = arith.constant 3 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x?x4096xf16>) outs(%[[INIT]] : tensor<3xi64>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4096xf16>, %{{.*}}: tensor<3xi64>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?x4096xf16>
// CHECK-NEXT:      %[[EXT0:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x4096xf16>
// CHECK-NEXT:      %[[EXT1:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK-NEXT:      %[[EXT2:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]], %[[EXT2]] : tensor<3xi64>
// CHECK-NEXT:      hipsr.compute_yield %[[EXTENTS]] : tensor<3xi64>
// CHECK-NEXT:    } : tensor<3xi64>
// CHECK-NEXT:    return %[[RESULT]] : tensor<3xi64>
func.func @shape_full(%ctx: !hipsr.context, %input: tensor<?x?x4096xf16>)
    -> tensor<3xi64> {
  %0 = "onnx.Shape"(%input) {start = 0 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// -----

// A fully static input needs no extent query at all: both extents are
// literals and no tensor.dim appears in either region.
// CHECK-LABEL: func.func @shape_static(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<2x3xf16>) -> tensor<2xi64> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64> shape_region {
// CHECK-NEXT:      %[[LEN:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<2x3xf16>) outs(%[[INIT]] : tensor<2xi64>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %{{.*}}: tensor<2x3xf16>, %{{.*}}: tensor<2xi64>):
// CHECK-NEXT:      %[[EXT0:.*]] = arith.constant 2 : i64
// CHECK-NEXT:      %[[EXT1:.*]] = arith.constant 3 : i64
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]] : tensor<2xi64>
// CHECK-NEXT:      hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64>
// CHECK-NEXT:    } : tensor<2xi64>
// CHECK-NEXT:    return %[[RESULT]] : tensor<2xi64>
func.func @shape_static(%ctx: !hipsr.context, %input: tensor<2x3xf16>)
    -> tensor<2xi64> {
  %0 = "onnx.Shape"(%input) : (tensor<2x3xf16>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

// start drops the leading axes, so the window opens at axis 1.
// CHECK-LABEL: func.func @shape_start(
// CHECK:         hipsr.placeholder
// CHECK-SAME:      : tensor<2xi64> shape_region {
// CHECK-NEXT:      arith.constant 2 : index
// CHECK:         ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4096xf16>, %{{.*}}: tensor<2xi64>):
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      tensor.dim %[[IN]], %[[C1]]
func.func @shape_start(%ctx: !hipsr.context, %input: tensor<?x?x4096xf16>)
    -> tensor<2xi64> {
  %0 = "onnx.Shape"(%input) {start = 1 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

// A negative start counts back from the end, selecting the last axis alone.
// CHECK-LABEL: func.func @shape_negative_start(
// CHECK:         ^bb0(
// CHECK-NEXT:      %[[EXT:.*]] = arith.constant 4096 : i64
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[EXT]] : tensor<1xi64>
func.func @shape_negative_start(%ctx: !hipsr.context,
                                %input: tensor<?x?x4096xf16>)
    -> tensor<1xi64> {
  %0 = "onnx.Shape"(%input) {start = -1 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}

// -----

// end truncates the window, so the trailing static axis is left out: the body
// reads axes 0 and 1 and never mentions the 4096 extent.
// CHECK-LABEL: func.func @shape_end(
// CHECK:         hipsr.placeholder
// CHECK-SAME:      : tensor<2xi64> shape_region {
// CHECK-NEXT:      %[[LEN:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[LEN]] : index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x?x4096xf16>, %{{.*}}: tensor<2xi64>):
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?x4096xf16>
// CHECK-NEXT:      %[[EXT0:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x4096xf16>
// CHECK-NEXT:      %[[EXT1:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK-NEXT:      %[[EXTENTS:.*]] = tensor.from_elements %[[EXT0]], %[[EXT1]] : tensor<2xi64>
// CHECK-NEXT:      hipsr.compute_yield %[[EXTENTS]] : tensor<2xi64>
// CHECK-NEXT:    } : tensor<2xi64>
// CHECK-NEXT:    return %[[RESULT]] : tensor<2xi64>
func.func @shape_end(%ctx: !hipsr.context, %input: tensor<?x?x4096xf16>)
    -> tensor<2xi64> {
  %0 = "onnx.Shape"(%input) {end = 2 : si64}
      : (tensor<?x?x4096xf16>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}
