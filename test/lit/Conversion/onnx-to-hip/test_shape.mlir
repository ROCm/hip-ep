// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the ShapeToTensorDims pattern (lib/Conversion/OnnxToHip/
// ShapeConversion.cpp) lowers `onnx.Shape` into a `tensor.dim` chain
// combined via `tensor.from_elements`, per the ONNX Shape-15 spec.
//
// Coverage:
//   1. Static shape, full range  -> i64 constants in tensor.from_elements
//   2. Dynamic + static dims     -> mixed tensor.dim + i64 constants
//   3. start/end attributes      -> sub-range of dims (positive bounds)
//   4. Negative start/end        -> normalized via (idx + rank), per spec
//   5. Empty range (start>=end)  -> 0-element tensor<0xi64> constant
//
// The CHECK patterns are deliberately structural (no SSA-name captures)
// because the post-`--convert-onnx-to-hip` pipeline runs the canonicalizer
// inline, which hoists/renames constants. We assert that:
//   - the `onnx.Shape` op no longer exists,
//   - the lowered form contains the right structural ops (tensor.dim for
//     dynamic dims, arith.constant for static dims, tensor.from_elements
//     to assemble the result), and
//   - empty-range Shape becomes an `arith.constant dense<>` on tensor<0xi64>.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Placeholder @main_graph -- required by the metadata-generation step
  // inside --convert-onnx-to-hip. The ShapeToTensorDims pattern walks every
  // func.func in the module, so the named test functions below all get
  // exercised regardless of which func is the entry point.
  func.func @main_graph(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    return %arg0 : tensor<2x3x4xf32>
  }

  // --- Case 1: fully static input, full range ---
  // Three static dims (2, 3, 4) -> three i64 constants assembled via
  // tensor.from_elements into a 3xi64 result.
  // CHECK-LABEL: func.func @shape_static_full
  // CHECK-NOT:   onnx.Shape
  // CHECK-DAG:   arith.constant 2 : i64
  // CHECK-DAG:   arith.constant 3 : i64
  // CHECK-DAG:   arith.constant 4 : i64
  // CHECK:       tensor.from_elements {{.*}} : tensor<3xi64>
  func.func @shape_static_full(%x: tensor<2x3x4xf32>) -> tensor<3xi64> {
    %s = "onnx.Shape"(%x) : (tensor<2x3x4xf32>) -> tensor<3xi64>
    return %s : tensor<3xi64>
  }

  // --- Case 2: mixed dynamic + static dims, full range ---
  // Dynamic dim 0 -> tensor.dim + arith.index_cast; static dim 1 -> i64 const.
  // CHECK-LABEL: func.func @shape_dynamic_full
  // CHECK-NOT:   onnx.Shape
  // CHECK-DAG:   tensor.dim {{.*}} : tensor<?x16xf32>
  // CHECK-DAG:   arith.index_cast {{.*}} : index to i64
  // CHECK-DAG:   arith.constant 16 : i64
  // CHECK:       tensor.from_elements {{.*}} : tensor<2xi64>
  func.func @shape_dynamic_full(%x: tensor<?x16xf32>) -> tensor<2xi64> {
    %s = "onnx.Shape"(%x) : (tensor<?x16xf32>) -> tensor<2xi64>
    return %s : tensor<2xi64>
  }

  // --- Case 3: start/end sub-range (positive bounds) ---
  // start=1, end=3 picks dims [1..3) i.e. (3, 4).
  // CHECK-LABEL: func.func @shape_subrange_positive
  // CHECK-NOT:   onnx.Shape
  // CHECK-DAG:   arith.constant 3 : i64
  // CHECK-DAG:   arith.constant 4 : i64
  // CHECK:       tensor.from_elements {{.*}} : tensor<2xi64>
  func.func @shape_subrange_positive(%x: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
    %s = "onnx.Shape"(%x) {start = 1 : si64, end = 3 : si64}
       : (tensor<2x3x4x5xf32>) -> tensor<2xi64>
    return %s : tensor<2xi64>
  }

  // --- Case 4: negative start/end normalized via (idx + rank) ---
  // start=-2, end=-1 on rank-4 input -> start=2, end=3 -> dim[2] = 4.
  // CHECK-LABEL: func.func @shape_subrange_negative
  // CHECK-NOT:   onnx.Shape
  // CHECK:       arith.constant 4 : i64
  // CHECK:       tensor.from_elements {{.*}} : tensor<1xi64>
  func.func @shape_subrange_negative(%x: tensor<2x3x4x5xf32>) -> tensor<1xi64> {
    %s = "onnx.Shape"(%x) {start = -2 : si64, end = -1 : si64}
       : (tensor<2x3x4x5xf32>) -> tensor<1xi64>
    return %s : tensor<1xi64>
  }

  // --- Case 5: empty range (start >= end after normalization) ---
  // Per ONNX spec, Shape returns a 1-D tensor of length 0.
  // CHECK-LABEL: func.func @shape_empty_range
  // CHECK-NOT:   onnx.Shape
  // CHECK:       arith.constant dense<> : tensor<0xi64>
  func.func @shape_empty_range(%x: tensor<2x3xf32>) -> tensor<0xi64> {
    %s = "onnx.Shape"(%x) {start = 1 : si64, end = 1 : si64}
       : (tensor<2x3xf32>) -> tensor<0xi64>
    return %s : tensor<0xi64>
  }
}
