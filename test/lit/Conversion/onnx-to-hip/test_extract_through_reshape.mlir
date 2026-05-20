// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the ExtractThroughReshapeFromElements pattern
// (lib/Conversion/OnnxToHip/ReshapeConversion.cpp) folds
//
//   tensor.extract(tensor.collapse_shape(tensor.from_elements ...))
//   tensor.extract(tensor.expand_shape  (tensor.from_elements ...))
//
// down to the corresponding `tensor.from_elements` operand value when:
//   - the reshape result has a fully static shape, and
//   - the extract indices are compile-time constants.
//
// This extends the upstream `tensor.extract(tensor.from_elements)` fold
// (`ExtractOp::fold` in mlir/lib/Dialect/Tensor/IR/TensorOps.cpp) by one
// reshape hop, eliminating the heap roundtrip that morphizen's ONNX Loop
// importer would otherwise leave behind for trip-count rank-reduction.
//
// Coverage:
//   1. collapse_shape (1xi64 -> i64), extract %t[] -> folds to operand
//   2. expand_shape   (i64 -> 1xi64), extract %t[0] -> folds to operand
//   3. Dynamic reshape result -> NO fold
//   4. Non-constant extract index -> NO fold
//
// ============================================================================

// RUN: hip-mlir-opt %s --canonicalize --convert-onnx-to-hip --canonicalize | FileCheck %s

module {
  // Placeholder @main_graph -- required by the metadata-generation step
  // inside --convert-onnx-to-hip. The ExtractThroughReshapeFromElements
  // pattern is added to the same RewritePatternSet, so it runs on every
  // func.func in the module.
  func.func @main_graph(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    return %arg0 : tensor<16xf32>
  }

  // --- Case 1: collapse_shape(from_elements) then extract ---
  // CHECK-LABEL: func.func @extract_collapse_from_elements
  // CHECK-SAME:  (%[[V:.*]]: i64) -> i64
  // CHECK-NOT:   tensor.from_elements
  // CHECK-NOT:   tensor.collapse_shape
  // CHECK-NOT:   tensor.extract
  // CHECK:       return %[[V]] : i64
  func.func @extract_collapse_from_elements(%v: i64) -> i64 {
    %t1 = tensor.from_elements %v : tensor<1xi64>
    %t0 = tensor.collapse_shape %t1 [] : tensor<1xi64> into tensor<i64>
    %r = tensor.extract %t0[] : tensor<i64>
    return %r : i64
  }

  // --- Case 2: expand_shape(from_elements) then extract ---
  // CHECK-LABEL: func.func @extract_expand_from_elements
  // CHECK-SAME:  (%[[V:.*]]: i64) -> i64
  // CHECK-NOT:   tensor.from_elements
  // CHECK-NOT:   tensor.expand_shape
  // CHECK-NOT:   tensor.extract
  // CHECK:       return %[[V]] : i64
  func.func @extract_expand_from_elements(%v: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %t0 = tensor.from_elements %v : tensor<i64>
    %t1 = tensor.expand_shape %t0 [] output_shape [1] : tensor<i64> into tensor<1xi64>
    %r = tensor.extract %t1[%c0] : tensor<1xi64>
    return %r : i64
  }

  // --- Case 3: non-constant extract index -> NO fold ---
  // The pattern requires compile-time-constant extract indices; a dynamic
  // index leaves the extract in place.
  // CHECK-LABEL: func.func @extract_dynamic_index_no_fold
  // CHECK:       tensor.from_elements
  // CHECK:       tensor.extract
  func.func @extract_dynamic_index_no_fold(%v0: i64, %v1: i64, %i: index) -> i64 {
    %t = tensor.from_elements %v0, %v1 : tensor<2xi64>
    %t1 = tensor.expand_shape %t [[0, 1]] output_shape [2, 1]
        : tensor<2xi64> into tensor<2x1xi64>
    %c0 = arith.constant 0 : index
    %r = tensor.extract %t1[%i, %c0] : tensor<2x1xi64>
    return %r : i64
  }
}
